from __future__ import annotations
from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import Tensor
from tqdm.auto import tqdm
from torch.nn import BCELoss, DataParallel, Module, Linear, Sigmoid
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from transformers import AutoModel, AutoTokenizer, BatchEncoding, BertModel, PreTrainedTokenizerBase, RobertaModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss
from statistics import mean
import logging

class BertClassifier(ABC):
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
        accumulation_steps: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neural_network: Optional[Module] = None,
        pretrained_model_name_or_path: Optional[str] = "bert-base-uncased",
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if not neural_network:
            bert = AutoModel.from_pretrained(pretrained_model_name_or_path)
            neural_network = BertClassifierNN(bert)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
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


        self.neural_network.to(device)
        if device.startswith("cuda") and many_gpus:
            self.neural_network = DataParallel(self.neural_network)

    def fit(self, x_train: list[str], y_train: list[bool],x_val:list[str],y_val:list[bool], epochs: Optional[int] = None) -> None:
        if not epochs:
            epochs = self.epochs
        optimizer = AdamW(self.neural_network.parameters(), lr=self.learning_rate)
        tokens = self._tokenize(x_train)
        dataset = TokenizedDataset(tokens, y_train)
        tokens_val=self._tokenize(x_val)
        dataset_val=TokenizedDataset(tokens_val,y_val)
        #self.loggerf.debug("[fit] x_train shape: %s | text: %s",len(x_train), x_train)
        #self.loggerf.debug("[fit] tokens keys: %s | tokens[inputs ids shape]: %s | input_ids shape: %s", tokens.keys(),len(tokens['input_ids']),tokens['input_ids'][0].shape)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset), batch_size=self.batch_size, collate_fn=self.collate_fn
        )
        dataloader_val = DataLoader(
            dataset_val, sampler=RandomSampler(dataset_val), batch_size=self.batch_size, collate_fn=self.collate_fn
        )

        l=tqdm(range(epochs), leave=True, colour='red')
        for epoch in l:
            l.set_description(f'Epoch {epoch}')
            trainLoss=self._train_single_epoch(dataloader, optimizer)
            valMetrics=self._eval_single_epoch(dataloader_val)
            self.loggert.debug("\n| Epoca | Train Loss | Val Loss | Precisão | Recall |   F1   | Acurácia | Hamming Loss |")
            self.loggert.debug("--------------------------------------------------------------------------------------")
            self.loggert.debug("|  %s  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |\n", epoch+1, trainLoss, valMetrics['val_losses'], valMetrics['precision'], valMetrics['recall'], valMetrics['f1'], valMetrics['accuracy'], valMetrics['hloss'])

    def predict(self, x: list[str], batch_size: Optional[int] = None) -> list[tuple[bool, float]]:
        if not batch_size:
            batch_size = self.batch_size
        scores = self.predict_scores(x, batch_size)
        classes = [i >= 0.5 for i in scores]
        return list(zip(classes, scores))

    def predict_classes(self, x: list[str], batch_size: Optional[int] = None) -> list[bool]:
        if not batch_size:
            batch_size = self.batch_size
        scores = self.predict_scores(x, batch_size)
        #self.loggerf.debug("[predict_classes] scores: %s", scores)
        classes = [i >= 0.5 for i in scores]
        #self.loggerf.debug("[predict_classes] classes: %s", scores)
        return classes

    def predict_scores(self, x: list[str], batch_size: Optional[int] = None) -> list[float]:
        if not batch_size:
            batch_size = self.batch_size
        tokens = self._tokenize(x)
        dataset = TokenizedDataset(tokens)
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
                predictions = self._evaluate_single_batch(batch,None)
                total_predictions.extend(predictions.tolist())
        return total_predictions

    @abstractmethod
    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        pass
    
    def _eval_single_epoch(self, dataloader: DataLoader) -> None:
        self.neural_network.eval()
        cross_entropy = BCELoss()
        preds=[]
        trues=[]
        val_losses=[]
        loop = tqdm(dataloader, leave=True, colour='magenta')
        for step, batch in enumerate(loop):
            loop.set_description(f'Validation | Step: {step}')
            labels = batch[-1].float().cpu()
            predictions = self._evaluate_single_batch(batch,None)
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            loss = cross_entropy(predictions, labels)
            val_losses.append(float(loss.detach().cpu().numpy()))
            preds.append(torch.tensor(predictions.cpu().detach().numpy()))
            trues.append(torch.tensor(labels.detach().numpy()))     
        y_true=torch.cat(trues,0)
        y_pred=torch.cat(preds,0)
        precisao=precision_score(y_true, y_pred,average='weighted',zero_division=0)
        recall=recall_score(y_true, y_pred,average='weighted')
        f1=f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        acuracia=accuracy_score(y_true, y_pred)
        hl=hamming_loss(y_true, y_pred)
        return {'val_losses':mean(val_losses),'precision':precisao,'recall':recall,'f1':f1,'accuracy':acuracia,'hloss':hl}


    def _train_single_epoch(self, dataloader: DataLoader, optimizer: Optimizer) -> None:
        self.neural_network.train()
        losses=[]
        cross_entropy = BCELoss()
        with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1,repeat=0),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('../torch-profiler'),
        record_shapes=False,
        profile_memory=True,
        with_stack=False) as prof:
            loop = tqdm(dataloader, leave=True, colour='green')
            for step, batch in enumerate(loop):
                loop.set_description(f'Treinamento | Step: {step}')
                labels = batch[-1].float().cpu()
                #self.loggerf.debug("[train_single_epoch] labels: %s", labels)
                #self.loggerf.debug("[train_single_epoch] memory before evaluate : %s",torch.cuda.memory_allocated())
                predictions = self._evaluate_single_batch(batch,optimizer)
                #self.loggerf.debug("[train_single_epoch] memory after evaluate : %s",torch.cuda.memory_allocated())
                loss = cross_entropy(predictions, labels) / self.accumulation_steps
                losses.append(float(loss.detach().cpu().numpy()))
                #self.loggerf.debug("[train_single_epoch] loss: %s", loss)
                loop.set_postfix(loss=loss.item())
                loss.backward()
                if ((step + 1) % self.accumulation_steps == 0) or (step + 1 == len(dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
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
    def load(cls, model_dir: str, device: str = "cuda:0", many_gpus: bool = False) -> BertClassifier:
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


class BertClassifierNN(Module):
    def __init__(self, model: Union[BertModel, RobertaModel]):
        super().__init__()
        self.model = model

        # classification head
        self.linear = Linear(768, 1)
        self.sigmoid = Sigmoid()

        #self.loggerf = logging.getLogger(name='arq1')
        #self.loggerf.propagate = False
        #self.loggerf.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        handler_file =  logging.FileHandler(filename='BELT-ClassifierNN.log',mode='w')
        handler_file.setFormatter(formatter)
        handler_file.setLevel(logging.DEBUG)
        #self.loggerf.addHandler(handler_file)


    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        with torch.no_grad():
            
            x = self.model(input_ids, attention_mask)
            #self.loggerf.debug("[forward] BEFORE x: %s|shape: %s", x,x[0].shape)
            #Usa a media dos embeddings de todos os tokens para representá-lo
            x=torch.mean(x[0], dim=1)
            #x = x[0][:, 0, :]  # take <s> token (equiv. to [CLS])
            #self.loggerf.debug("[forward] AFTER x: %s|shape: %s", x,x.shape)

        # classification head
        x = self.linear(x)
        #self.loggerf.debug("[forward] linear: %s", x)
        x = self.sigmoid(x)
        #self.loggerf.debug("[forward] sigmoid: %s", x)
        return x


class TokenizedDataset(Dataset):
    """Dataset for tokens with optional labels."""

    def __init__(self, tokens: BatchEncoding, labels: Optional[list] = None):
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.labels = labels
        #self.loggerf = logging.getLogger(name='arq2')
        #self.loggerf.propagate = False
        #self.loggerf.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        handler_file =  logging.FileHandler(filename='BELT-Tokenized.log',mode='w')
        handler_file.setFormatter(formatter)
        handler_file.setLevel(logging.DEBUG)
        #self.loggerf.addHandler(handler_file)


    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Union[tuple[Tensor, Tensor, Any], tuple[Tensor, Tensor]]:
        if self.labels:
            #self.loggerf.debug("[getitem] idx: %s", idx)
            #self.loggerf.debug("[getitem] input_ids shape: %s", self.input_ids[idx].shape)
            #self.loggerf.debug("[getitem] attention_mask: %s", self.attention_mask[idx].shape)
            #self.loggerf.debug("[getitem] labels: %s", self.labels[idx])
            return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]
        return self.input_ids[idx], self.attention_mask[idx]
