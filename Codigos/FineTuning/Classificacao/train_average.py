import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from earlyStopping import EarlyStopping
from statistics import mean 
from datasets import load_dataset, DatasetDict, disable_caching
from save_models import SaveBestModel, SaveBestMetrics
from  distrib_balanced_loss import ResampleLoss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import logging
from sklearn.metrics import hamming_loss, classification_report, accuracy_score
from classifier import Classifier
from tqdm.auto import tqdm
import time
import pandas as pd
import numpy as np
from belt_nlp.splitting import split_tokens_into_smaller_chunks, add_special_tokens_at_beginning_and_end, add_padding_tokens, stack_tokens_from_all_chunks


class TrainModel:
    def __init__(self,
                 dataset: DatasetDict,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 patience: int,
                 dir_save_models: str,
                 dir_save_metrics: str,
                 gradient_accumulation_steps: int,
                 method: int,
                 max_chunks: int,
                 max_length: int,
                 coluna: str,
                 modelo: str,
                 model_name: str,
                 metrics_name: str,
                 device: str,
                 cross_validation: bool,
                 file_folds: str,
                 tipo_classificacao: str,
                 tipo_estrategia: str,
                 funcao_perda_ponderada: str,
                 huggingface_dataset: bool,
                 token: str                
                 ) -> None:
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.dir_save_models = dir_save_models
        self.dir_save_metrics = dir_save_metrics
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.method = method
        self.max_chunks = max_chunks
        self.max_length = max_length
        self.coluna = coluna
        self.modelo = modelo
        self.model_name = model_name
        self.metrics_name = metrics_name
        self.cross_validation = cross_validation
        self.file_folds = file_folds
        self.tipo_classificacao = tipo_classificacao
        self.tipo_estrategia = tipo_estrategia
        self.funcao_perda_ponderada = funcao_perda_ponderada
        self.huggingface_dataset = huggingface_dataset
        self.token = token
        self.device = torch.device(device)
        
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Importando o tokenizador...")
        self.tokenizador = AutoTokenizer.from_pretrained(self.modelo)
        
        if huggingface_dataset:
            self.logger.info("Carregando o dataset do Hugging Face...")
            # Caso seja um subset de outro dataset, isso é separado por vírgula.
            if len(dataset.split(",")) > 1:
                dataset_name, subset_name = dataset.split(",")
                self.dataset = load_dataset(dataset_name, subset_name, token=self.token)
            else:
                self.dataset = load_dataset(dataset, token=self.token)
        else:
            self.logger.info("Carregando o dataset do diretório local...")
            self.dataset = load_dataset(dataset)
            
        # Mapeamento dos rótulos
        if self.tipo_classificacao == 'multilabel':
            self.labels = list(self.dataset['train'].features.keys())[1:]
        elif self.tipo_classificacao == 'multiclasse':
            self.labels = list(set(self.dataset['train']['label']))
        else:
            raise ValueError("Tipo de classificação não reconhecido.")
        
        self.id2label = {idx:label for idx, label in enumerate(self.labels)}
        self.label2id = {label:idx for idx, label in enumerate(self.labels)}
            
        self.encoded_dataset = self.dataset.map(self.preprocess_data, batched=True)
        self.encoded_dataset.set_format("torch")
    
    
    def preprocess_data(self, examples):        
        text = examples["text"]
        if self.tipo_estrategia == 'average':
            encoding=self.tokenizador(text, add_special_tokens=False, truncation=False)
        elif self.tipo_estrategia == 'max_length': 
            encoding=self.tokenizador(text, truncation=True, padding='max_length', max_length=self.max_length)
        else:
            raise ValueError("Estratégia de treinamento não reconhecida.")
        
        if self.tipo_classificacao == 'multilabel':
            labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
            labels_matrix = np.zeros((len(examples["text"]), len(self.labels)))
            for idx, label in enumerate(self.labels):
                labels_matrix[:, idx] = labels_batch[label]
            encoding["labels"] = labels_matrix.tolist()
        elif self.tipo_classificacao == 'multiclasse':
            encoding["labels"] = examples["label"]
        else:
            raise ValueError("Tipo de classificação não reconhecido.")
        
        return encoding
    
    
    def calculate_mean_global_metrics(self, metricas: list) -> dict:
        """
        Calcula a média das métricas para o classification_report.
        """
        mean_metrics = {}
        for key in metricas[0].keys():
            if key == 'accuracy':
                mean_metrics[key] = np.mean([metrica[key] for metrica in metricas])
            else:
                precision = np.mean([metrica[key]['precision'] for metrica in metricas])
                recall = np.mean([metrica[key]['recall'] for metrica in metricas])
                f1_score = np.mean([metrica[key]['f1-score'] for metrica in metricas])
                support = np.mean([metrica[key]['support'] for metrica in metricas])
                mean_metrics[key] = {'precision': precision, 'recall': recall, 'f1-score': f1_score, 'support': support}
                
        return mean_metrics
    
    
    def truncate_tensor(self,tensor, max_length):
        if tensor.size(1) <= max_length:
            return tensor
        else:
            return tensor[:, :max_length]
    
    
    def get_info_dataset(self, freq_cutoff=0):
        if self.huggingface_dataset:
            dataset = pd.DataFrame(self.dataset['train'])
        else:
            dataset = pd.read_csv(self.dataset)
        # Conta a frequência de cada rótulo
        term2count = {label: sum(dataset[label]) for label in self.labels}
        # Filtra termos com frequência acima do corte
        term_freq = sorted([term for term, count in term2count.items() if count >= freq_cutoff])
        # Frequência das classes
        class_freq = [term2count[term] for term in term_freq]
        # Número total de amostras no dataset
        train_num = len(dataset)
        return class_freq, train_num
    
    
    def get_loss_function(self):
        if self.tipo_classificacao == 'multilabel' and self.funcao_perda_ponderada == True:
            class_freq, train_num = self.get_info_dataset(freq_cutoff=0)
            return ResampleLoss(reweight_func='rebalance', 
                        loss_weight=1.0, focal=dict(focal=True, alpha=0.5, gamma=2),
                        logit_reg=dict(init_bias=0.05, neg_scale=2.0), map_param=dict(alpha=2.5, beta=10.0, gamma=0.9),
                        class_freq=class_freq, train_num=train_num)
        elif self.tipo_classificacao == 'multilabel' and self.funcao_perda_ponderada == False:
            return BCEWithLogitsLoss()
        elif self.tipo_classificacao == 'multiclasse':
            return CrossEntropyLoss()
        else:
            raise ValueError("Tipo de classificação não reconhecido.")
    
    
    def compute_metrics(self, pred, labels):
        acuracia=accuracy_score(labels, pred)
        hl=hamming_loss(labels, pred)
        cf_report = classification_report(labels, pred, output_dict=True, zero_division=1)
        return cf_report, acuracia, hl
    

    def process_batch(self, inputs: dict) -> tuple:
        trunc_size=int(510*(self.max_chunks/self.batch_size))
        labels = inputs.get("labels").to(self.device)
        input_ids_padded=inputs['input_ids']
        indexes = torch.nonzero(input_ids_padded)
        input_ids_orig = self.truncate_tensor(input_ids_padded[indexes].squeeze().unsqueeze(dim=0).to("cpu"), trunc_size)
        att_mask_padded=inputs['attention_mask']
        indexes = torch.nonzero(att_mask_padded)
        att_maks_orig = self.truncate_tensor(att_mask_padded[indexes].squeeze().unsqueeze(dim=0).to("cpu"), trunc_size)
        tensor_dict = {'input_ids': input_ids_orig, 'attention_mask': att_maks_orig}
        input_id_chunks, mask_chunks = split_tokens_into_smaller_chunks(tensor_dict, chunk_size=510, stride=500, minimal_chunk_length=50)
        add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
        add_padding_tokens(input_id_chunks, mask_chunks)
        input_ids, attention_mask = stack_tokens_from_all_chunks(input_id_chunks, mask_chunks)
        
        return input_ids, attention_mask, labels        
        

    def train_step(self, train_loader: DataLoader, fold: int, epoca: int) -> float:
        """
        Função que realiza o treinamento do modelo.
        Parâmetros:
            train_loader: dataloader de treino
            device: dispositivo a ser utilizado
        Retorno:
            loss: loss média do treino
        """
        self.classifier.train()        
        losses=[]
        loop = tqdm(train_loader, leave=True, colour='green')
        for batch_idx, inputs in enumerate(loop):    
            if self.tipo_estrategia == 'average':
                self.logger.info("Treinamento com estratégia de average.")
                input_ids, attention_mask, labels = self.process_batch(inputs)
                logits=self.classifier(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device))         
            elif self.tipo_estrategia == 'max_length':
                self.logger.info("Treinamento com estratégia de max_length.")
                input_ids, attention_mask, labels = inputs.get("input_ids"), inputs.get('attention_mask'), inputs.get("labels")
                logits=self.classifier(input_ids = input_ids.unsqueeze(0).to(self.device), attention_mask = attention_mask.unsqueeze(0).to(self.device))
            else:
                raise ValueError("Estratégia de treinamento não reconhecida.")
            
            if self.tipo_classificacao == 'multilabel' and self.funcao_perda_ponderada == True:
                loss=self.criterion(logits, labels.unsqueeze(0).to(self.device))
            elif (self.tipo_classificacao == 'multilabel' and self.funcao_perda_ponderada == False) or (self.tipo_classificacao == 'multiclasse'):
                loss=self.criterion(logits, labels.to(self.device))
            else:
                raise ValueError("Tipo de classificação não reconhecido.")
            
            if self.cross_validation:
                loop.set_description(f'Treinamento - {self.model_name} | Fold: {fold} | Época: {epoca}')
            else:
                loop.set_description(f'Treinamento - {self.model_name} | Época: {epoca}')
                
            loop.set_postfix(loss=loss.item())
            losses.append(float(loss.detach().cpu().numpy()))
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if ((batch_idx + 1) % self.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
        return mean(losses)
    
    
    def val_step(self, val_loader: DataLoader, fold: int, epoca: int) -> dict:
        """
        Função que realiza a validação do modelo.
        Parâmetros:
            val_loader: dataloader de validação
            device: dispositivo a ser utilizado
            epoch: época atual
            fold: fold atual
        Retorno:
            metricas: dicionário com as métricas de validação
        """
        self.classifier.eval()        
        preds, trues, val_losses = [], [], []
        loop = tqdm(val_loader, leave=True, colour='blue')
        
        for inputs in loop:
            if self.tipo_estrategia == 'average':
                input_ids, attention_mask, labels = self.process_batch(inputs)
                with torch.no_grad():
                    logits=self.classifier(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device)) 
            elif self.tipo_estrategia == 'max_length':
                input_ids, attention_mask, labels = inputs.get("input_ids"), inputs.get('attention_mask'), inputs.get("labels")
                with torch.no_grad():
                    logits=self.classifier(input_ids = input_ids.unsqueeze(0).to(self.device), attention_mask = attention_mask.unsqueeze(0).to(self.device))
            else:
                raise ValueError("Estratégia de treinamento não reconhecida.")
                    
            if self.tipo_classificacao == 'multilabel' and self.funcao_perda_ponderada == True:
                loss=self.criterion(logits, labels.unsqueeze(0))
            elif (self.tipo_classificacao == 'multilabel' and self.funcao_perda_ponderada == False) or (self.tipo_classificacao == 'multiclasse'):
                loss=self.criterion(logits, labels)
            else:
                raise ValueError("Tipo de classificação não reconhecido.")
                
            if self.cross_validation:
                loop.set_description(f'Validação - {self.model_name} | Fold: {fold} | Época: {epoca}')
            else:
                loop.set_description(f'Validação - {self.model_name} | Época: {epoca}')
                
            loop.set_postfix(loss=loss.item())
            val_losses.append(float(loss.detach().cpu().numpy()))
            
            if self.tipo_classificacao == 'multiclasse':
                probs = torch.argmax(torch.softmax(logits.unsqueeze(0), dim=1), dim=1)
                predictions=torch.clone(probs)
                preds.append(torch.tensor(predictions.squeeze(0).cpu().detach().numpy()))
                trues.append(torch.tensor(labels.cpu().detach().numpy()))
            elif self.tipo_classificacao == 'multilabel':
                probs=torch.sigmoid(logits)
                predictions=torch.clone(probs)
                predictions[predictions >= 0.5] = 1
                predictions[predictions < 0.5] = 0
                preds.append(torch.tensor(predictions.cpu().detach().numpy()))
                trues.append(torch.tensor(labels.cpu().detach().numpy()))
            else:
                raise ValueError("Tipo de classificação não reconhecido.")
                
        y_true = torch.stack(trues).cpu().detach().numpy()
        y_pred = torch.stack(preds).cpu().detach().numpy()
        cf_report, acuracia, hl = self.compute_metrics(y_pred, y_true)
        
        return {'fold': fold, 'epoca':epoca, 'val_losses':mean(val_losses), 'cf_report': cf_report,'accuracy':acuracia, 'hloss': hl}
    
    
    def loggerr_validacao(self, fold: int, epoch: int, trainLoss: float, valMetrics: dict) -> None:
        original_formatter = self.logger.handlers[0].formatter
        simple_formatter = logging.Formatter('%(message)s')
        self.logger.handlers[0].setFormatter(simple_formatter)
        
        try:
            self.logger.info("\n| Fold | Epoca | Train Loss | Val Loss | Precisão | Recall |   F1   | Acurácia | Hamming Loss |")
            self.logger.info("-----------------------------------------------------------------------------------------------")
            if self.cross_validation:
                self.logger.info("| {:^4} | {:^5} | {:^10.4f} | {:^8.4f} | {:^8.4f} | {:^6.4f} | {:^6.4f} | {:^8.4f} | {:^12.4f} |\n".format(
                    fold, epoch, trainLoss, valMetrics['val_losses'],
                    valMetrics['cf_report']['weighted avg']['precision'],
                    valMetrics['cf_report']['weighted avg']['recall'],
                    valMetrics['cf_report']['weighted avg']['f1-score'],
                    valMetrics['accuracy'], valMetrics['hloss']))
            else:
                self.logger.info("| {:^4} | {:^10.4f} | {:^8.4f} | {:^8.4f} | {:^6.4f} | {:^6.4f} | {:^8.4f} | {:^12.4f} |\n".format(
                    epoch, trainLoss, valMetrics['val_losses'],
                    valMetrics['cf_report']['weighted avg']['precision'],
                    valMetrics['cf_report']['weighted avg']['recall'],
                    valMetrics['cf_report']['weighted avg']['f1-score'],
                    valMetrics['accuracy'], valMetrics['hloss']))
        finally:
            self.logger.handlers[0].setFormatter(original_formatter)


    def train(self):
        self.logger.info("Iniciando o treinamento...")
        # Tipos de função de perda
        self.criterion = self.get_loss_function()
        if self.cross_validation:
            self.logger.info("Treinamento com Cross-Validation...")
            folds = np.load(self.file_folds, allow_pickle=True)
            history={'train_losses': [],'val_losses': [], 'accuracy': [], 'cf_report': [], 'f1_curve': [], 'train_loss_curve': [], 'val_loss_curve': [], 'hloss': [], 'time_per_epoch': [], 'epochs_per_fold': []}
            save_best_model = SaveBestModel()
            save_best_metrics = SaveBestMetrics()
            # Para treinamentos Cross-Validation
            for fold, (train_idx, val_idx) in enumerate(folds):
                train_dataset = self.encoded_dataset['train'].select(train_idx)
                val_dataset = self.encoded_dataset['train'].select(val_idx)
                
                f1_curve, train_loss_curve, val_loss_curve  = [], [], []
                self.classifier = Classifier(input_size = 768, output_size=len(self.labels), method=self.method, model=self.modelo, tokenizer=self.tokenizador).to(self.device)
                self.optimizer = AdamW(self.classifier.parameters(), lr=self.learning_rate)
                early_stopping = EarlyStopping(self.patience, os.path.join(self.dir_save_models, "checkpoint.pth"), trace_func=self.logger.info)
                inicio = time.time()
                
                for epoch in range(self.epochs):
                    trainLoss = self.train_step(train_dataset, fold+1, epoch+1)
                    valMetrics = self.val_step(val_dataset, fold+1, epoch+1)
                    f1_curve.append(valMetrics['cf_report']['weighted avg']['f1-score'])
                    train_loss_curve.append(trainLoss)
                    val_loss_curve.append(valMetrics['val_losses'])
                    self.loggerr_validacao(fold+1, epoch+1, trainLoss, valMetrics)
                    early_stopping(valMetrics['val_losses'], self.classifier, self.optimizer, epoch+1)        
                    if early_stopping.early_stop:
                        self.logger.info("Early stopping")
                        break
                    bestMetrics = save_best_metrics(valMetrics['cf_report']['weighted avg']['f1-score'], valMetrics, trainLoss)
                    save_best_model(valMetrics['val_losses'], self.classifier, self.dir_save_models, self.model_name, self.coluna)
                fim = time.time()
                
                history['train_losses'].append(bestMetrics['train_loss'])
                history['val_losses'].append(bestMetrics['val_losses'])
                history['cf_report'].append(bestMetrics['cf_report'])
                history['hloss'].append(bestMetrics['hloss'])
                history['f1_curve'].append(f1_curve)
                history['train_loss_curve'].append(train_loss_curve)
                history['val_loss_curve'].append(val_loss_curve)
                history['time_per_epoch'].append((fim - inicio)/epoch)
                history['epochs_per_fold'].append(epoch)
                    
            metricas = {
                "model_name": self.model_name,
                "train_loss": np.mean(history['train_losses']),
                "train_loss_curve": history['train_loss_curve'],
                "val_loss": np.mean(history['val_losses']),
                "val_loss_curve": history['val_loss_curve'],
                "cf_report": self.calculate_mean_global_metrics(history['cf_report']),
                "hloss": np.mean(history['hloss']),
                "training_time_epoch": history['time_per_epoch'],
                "mean_training_time_epoch": np.mean(history['time_per_epoch']),
                "f1_curve": history['f1_curve'],
                "epochs_per_fold": history['epochs_per_fold'],
                "mean_epochs": np.mean(history['epochs_per_fold'])
            }
            self.logger.removeHandler(self.logger.handlers[0])
            return metricas
            
        else:
            self.logger.info("Treinamento particionado...")
            train_dataset = self.encoded_dataset['train']
            val_dataset = self.encoded_dataset['validation']
            test_dataset = self.encoded_dataset['test']
                
            f1_curve, train_loss_curve, val_loss_curve  = [], [], []
            self.classifier = Classifier(input_size = 768, output_size=len(self.labels), method=self.method, model=self.modelo, tokenizer=self.tokenizador).to(self.device)
            self.optimizer = AdamW(self.classifier.parameters(), lr=self.learning_rate)
            early_stopping = EarlyStopping(self.patience, os.path.join(self.dir_save_models, "checkpoint.pth"), trace_func=self.logger.info)
            inicio = time.time()
            
            for epoch in range(self.epochs):
                trainLoss = self.train_step(train_dataset, None, epoch+1)
                valMetrics = self.val_step(val_dataset, None, epoch+1)
                f1_curve.append(valMetrics['cf_report']['weighted avg']['f1-score'])
                train_loss_curve.append(trainLoss)
                val_loss_curve.append(valMetrics['val_losses'])
                self.loggerr_validacao(None, epoch, trainLoss, valMetrics)
                early_stopping(valMetrics['val_losses'], self.classifier, self.optimizer, epoch+1)        
                if early_stopping.early_stop:
                    self.logger.info("Early stopping")
                    break
                
            fim = time.time()
            test_metrics = self.val_step(test_dataset, fold+1, epoch+1)
            
            metricas = {
                "model_name": self.model_name,
                "train_loss": trainLoss,
                "train_loss_curve": train_loss_curve,
                "val_loss": test_metrics['val_losses'],
                "val_loss_curve": val_loss_curve,
                "cf_report": test_metrics['cf_report'],
                "hloss": test_metrics['hloss'],
                "training_time_epoch": (fim - inicio)/epoch,
                "f1_curve": f1_curve,
            }
            self.logger.removeHandler(self.logger.handlers[0])
            return metricas              