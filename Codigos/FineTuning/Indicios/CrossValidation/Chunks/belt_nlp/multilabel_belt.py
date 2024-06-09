from __future__ import annotations
from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import Tensor
from tqdm.auto import tqdm
from torch.nn import BCEWithLogitsLoss, DataParallel, Module, Linear, Sigmoid
from focalloss import FocalLoss
from  distrib_balanced_loss import ResampleLoss
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from transformers import AutoModel, AutoTokenizer, BatchEncoding, BertModel, PreTrainedTokenizerBase
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss
from statistics import mean
import logging
from belt_nlp.earlyStopping import EarlyStopping
from belt_nlp.modelSaves import SaveBestModel
from hurry.filesize import size
from sys import getsizeof
import os

class MultilabelBertClassifier(ABC):
    """
    The "device" parameter can have the following values:
        - "cpu" - The model will be loaded on CPU.
        - "cuda" - The model will be loaded on single GPU.
        - "cuda:i" - The model will be loaded on the specific single GPU with the index i.

    It is also possible to use multiple GPUs. In order to do this:
        - Set device to "cuda".
        - Set many_gpu flag to True.
        - As default it will use all of them.

    To use only selected GPUs - set the environmental variable CUDA_VISIBLE_DEVICES.
    """

    @abstractmethod
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        num_classes: int,
        patience: int = 7,
        logging_status:bool=True,
        class_freq: Optional[list[int]] = None,
        train_num: Optional[int] = None,
        save_dir: Optional[str] = "BELT_Model_Saves",
        accumulation_steps: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neural_network: Optional[Module] = None,
        pretrained_model_name_or_path: Optional[str] = "bert-base-uncased",
        device: str = "cuda",
        many_gpus: bool = False,
    ):
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if not neural_network:
            bert = AutoModel.from_pretrained(pretrained_model_name_or_path)
            #bert.gradient_checkpointing_enable()
            #modules = [bert.embeddings,*bert.encoder.layer[:12]] 
            #for module in modules:
            #    for param in module.parameters():
            #        param.requires_grad = False
            #for layer_index in range(12):
            #    layer = bert.encoder.layer[layer_index]
            #    for param in layer.parameters():
            #        param.requires_grad = False
            for param in bert.parameters():
                param.requires_grad = False
            neural_network = MultilabelBertClassifierNN(bert,num_classes,logging_status=logging)

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_dir = save_dir
        self.patience = patience
        self.logging_status=logging_status
        self.accumulation_steps = accumulation_steps
        self._params = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "accumulation_steps": self.accumulation_steps
        }
        self.device = device
        self.many_gpus = many_gpus
        self.tokenizer = tokenizer
        self.neural_network = neural_network
        self.collate_fn = None
        self.class_freq=class_freq
        self.train_num=train_num
        
        if not self.logging_status:
            logging.disable(logging.WARNING)

        self.neural_network.to(device)
        if device.startswith("cuda") and many_gpus:
            self.neural_network = DataParallel(self.neural_network)

    def print_non_zeros(self, x: list[Tensor]) -> list[list]:
        ids_nonzero=[]
        for element in x:
            elem_list=element.tolist()
            elem_list=[[ele for ele in sublist if ele!=0] for sublist in elem_list]
            ids_nonzero.append(elem_list)
        return ids_nonzero
    
    def check_trainable_parameters(self,model):
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)

        return trainable_params

    def fit(self, x_train: list[str], y_train: list[list[bool]],x_val:list[str],y_val:list[list[bool]], epochs: Optional[int] = None) -> None:
        if not epochs:
            epochs = self.epochs
        optimizer = AdamW(self.neural_network.parameters(), lr=self.learning_rate)
        tokens = self._tokenize(x_train)
        dataset = TokenizedDataset(tokens, y_train,logging_status=self.logging_status)
        tokens_val=self._tokenize(x_val)
        dataset_val=TokenizedDataset(tokens_val,y_val,logging_status=self.logging_status)
        early_stopping = EarlyStopping(patience=self.patience, path=os.path.join(self.save_dir), trace_func=self.loggert.critical)
        save_best_model = SaveBestModel()
        self.loggerf.debug("[fit] x_train shape: %s | content: %s ",len(x_train),x_train)
        self.loggerf.debug("[fit] y_train: %s | shape: %s ",y_train,len(y_train))
        self.loggerf.debug("[fit] tokens keys: %s", tokens.keys())
        self.loggerf.debug("[fit] tokens[input_ids] shape: %s | shape each doc: %s ",len(tokens['input_ids']),[i.shape for i in tokens['input_ids']])
        self.loggerf.debug("[fit] input_ids: %s | \n attention_mask: %s",self.print_non_zeros(tokens['input_ids']) ,self.print_non_zeros(tokens['attention_mask']))
        
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset), batch_size=self.batch_size, collate_fn=self.collate_fn
        )
        dataloader_val = DataLoader(
            dataset_val, sampler=RandomSampler(dataset_val), batch_size=self.batch_size, collate_fn=self.collate_fn
        )
        
        l=tqdm(range(epochs), leave=True, colour='red')
        self.loggerf.debug("[fit] memory begin : %s",size(torch.cuda.memory_allocated()))
        for epoch in l:
            l.set_description(f'Epoch {epoch}')
            trainLoss=self._train_single_epoch(dataloader, optimizer)
            self.loggerf.debug("[fit] memory after train single epoch : %s",size(torch.cuda.memory_allocated()))
            valMetrics=self._eval_single_epoch(dataloader_val)
            self.loggerf.debug("[fit] memory after eval single epoch : %s",size(torch.cuda.memory_allocated()))
            self.loggert.critical("\n| Epoca | Train Loss | Val Loss | Precisão | Recall |   F1   | Acurácia | Hamming Loss |")
            self.loggert.critical("--------------------------------------------------------------------------------------")
            self.loggert.critical("|  %s  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |\n", epoch+1, trainLoss, valMetrics['val_losses'], valMetrics['precision'], valMetrics['recall'], valMetrics['f1'], valMetrics['accuracy'], valMetrics['hloss'])
            early_stopping(valMetrics['val_losses'], self.neural_network, optimizer, epoch+1)        
            if early_stopping.early_stop:
                self.loggert.critical("Early stopping")
                break            
            save_best_model(valMetrics['val_losses'],self.neural_network,self.save_dir,'BELT')

    def predict(self, x: list[str], batch_size: Optional[int] = None) -> list[tuple[bool, float]]:
        if not batch_size:
            batch_size = self.batch_size
        scores = self.predict_scores(x, batch_size)
        classes = [[int(i >= 0.5) for i in list] for list in scores]
        return list(zip(classes, scores))

    def predict_classes(self, x: list[str], batch_size: Optional[int] = None) -> list[bool]:
        if not batch_size:
            batch_size = self.batch_size
        scores = self.predict_scores(x, batch_size)
        self.loggerf.debug("[predict_classes] scores: %s", scores)
        classes = [[int(i >= 0.5) for i in list] for list in scores]
        self.loggerf.debug("[predict_classes] classes: %s", scores)
        return classes

    def predict_scores(self, x: list[str], batch_size: Optional[int] = None) -> list[float]:
        if not batch_size:
            batch_size = self.batch_size
        tokens = self._tokenize(x)
        dataset = TokenizedDataset(tokens,logging_status=self.logging_status)
        dataloader = DataLoader(
            dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, collate_fn=self.collate_fn
        )
        total_predictions = []

        # deactivate dropout layers
        self.neural_network.eval()
        loop2=tqdm(dataloader, leave=True, colour='green')
        for batch in loop2:
            loop2.set_description(f'Predict score')
            # deactivate autograd
            with torch.no_grad():
                predictions = self._evaluate_single_batch(batch)
                total_predictions.extend(predictions.tolist())
        return total_predictions

    @abstractmethod
    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        pass
    
    def _eval_single_epoch(self, dataloader: DataLoader) -> None:
        self.neural_network.eval()
        #cross_entropy = BCEWithLogitsLoss()
        #cross_entropy = FocalLoss(gamma=1)
        cross_entropy = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                         focal=dict(focal=True, alpha=0.5, gamma=2),
                         logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                         map_param=dict(alpha=2.5, beta=10.0, gamma=0.9),
                         class_freq=self.class_freq, train_num=self.train_num)
        preds=[]
        trues=[]
        val_losses=[]
        loop = tqdm(dataloader, leave=True, colour='magenta')
        self.loggerf.debug("***************[eval_single_epoch - START] *******************")
        for step, batch in enumerate(loop):
            self.loggerf.debug("***************[eval_single_epoch - BATCH %s] *******************",step)
            loop.set_description(f'Validation | Step: {step}')
            labels = batch[-1].to(self.device)
            logits = self._evaluate_single_batch(batch).to(self.device)
            self.loggerf.debug("[_eval_single_epoch] logits: %s", logits)
            self.loggerf.debug("[_eval_single_epoch] labels: %s", labels)
            loss = cross_entropy(logits, labels)
            probs=torch.sigmoid(logits.detach().cpu())
            self.loggerf.debug("[_eval_single_epoch] probs: %s", probs)
            predictions=torch.clone(probs).to(self.device)
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            self.loggerf.debug("[_eval_single_epoch] after predictions: %s", predictions)
            val_losses.append(float(loss.detach().cpu().numpy()))
            preds.append(torch.tensor(predictions.detach().cpu().numpy()))
            trues.append(torch.tensor(labels.detach().cpu().numpy()))     
        y_true=torch.cat(trues,0)
        y_pred=torch.cat(preds,0)
        self.loggerf.debug("[_eval_single_epoch] y_true: %s", y_true)
        self.loggerf.debug("[_eval_single_epoch] y_pred: %s", y_pred)
        precisao=precision_score(y_true, y_pred,average='weighted',zero_division=0)
        recall=recall_score(y_true, y_pred,average='weighted')
        f1=f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        acuracia=accuracy_score(y_true, y_pred)
        hl=hamming_loss(y_true, y_pred)
        self.loggerf.debug("***************[eval_single_epoch - END] *******************")
        return {'val_losses':mean(val_losses),'precision':precisao,'recall':recall,'f1':f1,'accuracy':acuracia,'hloss':hl}


    def _train_single_epoch(self, dataloader: DataLoader, optimizer: Optimizer) -> None:
        self.neural_network.train()
        losses=[]
        #cross_entropy = BCEWithLogitsLoss()
        #cross_entropy = FocalLoss(gamma=1)
        cross_entropy = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                    focal=dict(focal=True, alpha=0.5, gamma=2),
                    logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                    map_param=dict(alpha=2.5, beta=10.0, gamma=0.9),
                    class_freq=self.class_freq, train_num=self.train_num)
        loop = tqdm(dataloader, leave=True, colour='green')
        self.loggerf.debug("***************[train_single_epoch - START] *******************")
        for step, batch in enumerate(loop):
            self.loggerf.debug("***************[train_single_epoch - BATCH %s] *******************",step)
            loop.set_description(f'Treinamento | Step: {step}')
            labels = batch[-1].to(self.device)            
            self.loggerf.debug("[train_single_epoch] memory before eval single batch: %s",size(torch.cuda.memory_allocated()))
            predictions = self._evaluate_single_batch(batch).to(self.device)
            self.loggerf.debug("[train_single_epoch] memory after eval single batch: %s",size(torch.cuda.memory_allocated()))
            self.loggerf.debug("[train_single_epoch] labels: %s", labels)
            self.loggerf.debug("[train_single_epoch] logits: %s", predictions)
            p=torch.sigmoid(predictions.clone().detach().cpu())
            self.loggerf.debug("[train_single_epoch] probabilities: %s", p)
            p[p >= 0.5] = 1
            p[p < 0.5] = 0
            self.loggerf.debug("[train_single_epoch] predictions: %s", p)
            #labels=torch.tensor(labels,dtype=float)
            loss = cross_entropy(predictions, labels) / self.accumulation_steps
            self.loggerf.debug("[train_single_epoch] loss: %s", loss)
            nt_params=self.check_trainable_parameters(self.neural_network)
            self.loggerf.debug("[train_single_epoch] Number parameters not tranaibles: %s| List: %s",len(nt_params), nt_params)
            loop.set_postfix(loss=loss.item())
            loss.backward()
            if ((step + 1) % self.accumulation_steps == 0) or (step + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            losses.append(float(loss.detach().cpu().numpy()))
            self.loggerf.debug("***************[train_single_epoch - END] *******************")
        return mean(losses)
       

    @abstractmethod
    def _evaluate_single_batch(self, batch: tuple[Tensor]) -> Tensor:
        pass

    def save(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(file=model_dir / "params.json", mode="w", encoding="utf-8") as file:
            json.dump(self._params, file)
        self.tokenizer.save_pretrained(model_dir)
        if self.many_gpus:
            torch.save(self.neural_network.module, model_dir / "model.bin")
        else:
            torch.save(self.neural_network, model_dir / "model.bin")

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda", many_gpus: bool = False) -> MultilabelBertClassifier:
        model_dir = Path(model_dir)
        with open(file=model_dir / "params.json", mode="r", encoding="utf-8") as file:
            params = json.load(file)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        neural_network = torch.load(f=model_dir / "model.bin", map_location=device)
        return cls(
            **params,
            tokenizer=tokenizer,
            neural_network=neural_network,
            pretrained_model_name_or_path=None,
            device=device,
            many_gpus=many_gpus,
        )


class MultilabelBertClassifierNN(Module):
    def __init__(self, model: BertModel, output_size,logging_status=False):
        super().__init__()
        self.model = model
        self.trans = torch.nn.TransformerEncoderLayer(d_model=768, nhead=2)
        self.dropout=torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(768, 30)
        self.classifier = torch.nn.Linear(30, output_size)

        self.loggerf = logging.getLogger(name='arq3')
        self.loggerf.propagate = False
        self.loggerf.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        handler_file =  logging.FileHandler(filename='BELT-MultilabelBertClassifierNN.log',mode='w')
        handler_file.setFormatter(formatter)
        handler_file.setLevel(logging.DEBUG)
        self.loggerf.addHandler(handler_file)

        if not logging_status:
            logging.disable(logging.WARNING)
        
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        #with torch.no_grad():
        #print("[forward] memory before bert:",size(torch.cuda.memory_allocated()))
        self.loggerf.debug("[forward] input_ids shape: %s",input_ids.shape)
        self.loggerf.debug("[forward] attention_mask shape: %s",attention_mask.shape)
        outputs = self.model(input_ids, attention_mask)
        self.loggerf.debug("[forward] outputs: %s",size(getsizeof(outputs)))
        self.loggerf.debug("[forward] memory after bert: %s",size(torch.cuda.memory_allocated()))
        self.loggerf.debug("outputs: %s | shape: %s", outputs[1],outputs[1].shape)
        #x = x[0][:, 0, :]
        #Usa a media dos embeddings de todos os tokens para representá-lo
        #x=torch.mean(x[0], dim=1)
        #x=outputs[1]
        x=torch.sum(outputs[0], dim=1)
        x=self.dropout(x)
        self.loggerf.debug("dropout x: %s | shape: %s",x, x.shape)
        x = self.trans(x.unsqueeze(0))
        self.loggerf.debug("trans x: %s | shape: %s",x, x.shape)
        x = self.fc(x)
        self.loggerf.debug("fc x: %s | shape: %s",x, x.shape)
        x = self.classifier(x)
        self.loggerf.debug("classifier x: %s | shape: %s",x, x.shape)
        return x

class TokenizedDataset(Dataset):
    """Dataset for tokens with optional labels."""

    def __init__(self, tokens: BatchEncoding, labels: Optional[list] = None,logging_status=False):
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.labels = labels
        
        self.loggerf = logging.getLogger(name='arq2')
        self.loggerf.propagate = False
        self.loggerf.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        handler_file =  logging.FileHandler(filename='BELT-Tokenized.log',mode='w')
        handler_file.setFormatter(formatter)
        handler_file.setLevel(logging.DEBUG)
        self.loggerf.addHandler(handler_file)

        if not logging_status:
            logging.disable(logging.WARNING)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Union[tuple[Tensor, Tensor, Any], tuple[Tensor, Tensor]]:
        if self.labels:
            self.loggerf.debug("[getitem] idx: %s", idx)
            self.loggerf.debug("[getitem] input_ids shape: %s", self.input_ids[idx].shape)
            self.loggerf.debug("[getitem] attention_mask: %s", self.attention_mask[idx].shape)
            self.loggerf.debug("[getitem] labels: %s", self.labels[idx])
            return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]
        return self.input_ids[idx], self.attention_mask[idx]
