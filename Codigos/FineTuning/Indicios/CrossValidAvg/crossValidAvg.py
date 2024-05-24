import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
import sys
from datasets import load_dataset
import os
from transformers import AutoTokenizer, logging
from torch.utils.data import DataLoader
from statistics import mean 
from  distrib_balanced_loss import ResampleLoss
import logging
import time
import pandas as pd
from sklearn.metrics import hamming_loss, classification_report
from belt_nlp.splitting import split_tokens_into_smaller_chunks, add_special_tokens_at_beginning_and_end, add_padding_tokens, stack_tokens_from_all_chunks
sys.path.insert(0, '../../../')
from utils.earlyStopping import EarlyStopping
from utils.classifier import Classifier
from utils.modelSaves import SaveBestMetrics, SaveBestModel


class ClassificaIndicios:
    def __init__(self, batch_size: int, epochs: int, patience: int, model_name: str, dir_save_metrics: str,
                  dir_save_models: str, learning_rate: float, gradient_accumulation_steps: int, modelo: str, max_chunks: int, layer: int, tokenizer: str, dataset: str, coluna: str) -> None:
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model_name = model_name
        self.dir_save_metrics = dir_save_metrics
        self.dir_save_models = dir_save_models
        self.learning_rate = learning_rate
        self.coluna = coluna
        self.max_chunks = max_chunks
        self.layer = layer
        self.modelo = modelo
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.dataset = dataset
        self.device = torch.device('cuda')

        print('Carregando o tokenizador...')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        self.data = load_dataset("csv", data_files={"dataset": dataset})
        self.labels = list(self.data['dataset'].features.keys())[1:]
        self.id2label = {idx:label for idx, label in enumerate(self.labels)}
        self.label2id = {label:idx for idx, label in enumerate(self.labels)}
        
        self.encoded_dataset = self.data.map(self.preprocess_data, batched=True, remove_columns=self.data['dataset'].column_names)
        self.encoded_dataset.set_format("torch")

        self.loggert = logging.getLogger(name='tela')
        self.loggert.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        handler_screen = logging.StreamHandler()
        handler_screen.setFormatter(formatter)
        handler_screen.setLevel(logging.DEBUG)
        self.loggert.addHandler(handler_screen)
    
    
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
        
    
    def preprocess_data(self, examples):
        text = examples["text"]
        encoding=self.tokenizer(text, add_special_tokens=False, truncation=False)
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
        labels_matrix = np.zeros((len(examples["text"]), len(self.labels)))
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]
        encoding["labels"] = labels_matrix.tolist()
        return encoding
    
    
    def truncate_tensor(self,tensor, max_length):
        if tensor.size(1) <= max_length:
            return tensor
        else:
            return tensor[:, :max_length]


    def train_step(self, train_loader: DataLoader, fold: list, epoca: int) -> float:
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
        trunc_size=int(510*(self.max_chunks/self.batch_size))
        loop = tqdm(train_loader, leave=True, colour='green')
        for batch_idx, inputs in enumerate(loop):
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
            # Passando os dados para o modelo
            logits=self.classifier(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device))
            loss=self.criterion(logits, labels.unsqueeze(0))
            loop.set_description(f'Treinamento - {self.model_name} | Fold: {fold} | Época: {epoca}')
            loop.set_postfix(loss=loss.item())
            losses.append(float(loss.detach().cpu().numpy()))
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            if ((batch_idx + 1) % self.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        return mean(losses)
    

    def val_step(self, val_loader: DataLoader, fold: int, epoch: int) -> dict:
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
        preds=[]
        trues=[]
        val_losses=[]
        trunc_size=int(510*(self.max_chunks/self.batch_size))
        loop = tqdm(val_loader, leave=True, colour='blue')
        for inputs in loop:
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
            with torch.no_grad():
                # Obtendo os embeddings
                logits=self.classifier(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device))
            loss=self.criterion(logits, labels.unsqueeze(0))
            loop.set_description(f'Validação - {self.model_name} | Fold: {fold} | Época: {epoch}')
            loop.set_postfix(loss=loss.item())
            val_losses.append(float(loss.detach().cpu().numpy()))
            probs=torch.sigmoid(logits)
            predictions=torch.clone(probs)
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            preds.append(torch.tensor(predictions.cpu().detach().numpy()).squeeze(0))
            trues.append(torch.tensor(labels.cpu().detach().numpy()))     
        y_true=torch.cat(trues,0)
        y_pred=torch.cat(preds,0)
        cf_report = classification_report(y_true, y_pred, output_dict=True)
        hl = hamming_loss(y_true, y_pred)
        return {'fold': fold, 'epoca':epoch, 'val_losses':mean(val_losses), 'cf_report': cf_report, 'hloss': hl}        


    def modelTraining(self, folds: list) -> None:
        """
        Função que realiza o treinamento do modelo.
        Parâmetros:
            folds: folds de treino e validação
            train_dataset: dataset de treino
        """
        history={'train_losses': [],'val_losses': [], 'cf_report': [], 'f1_curve': [], 'hloss': [], 'time_per_epoch': [], 'epochs_per_fold': []}
        save_best_model = SaveBestModel()
        save_best_metrics = SaveBestMetrics()
        for fold,(train_idx, val_idx) in enumerate(folds):
            metrics_f1_curve = []
            train_df = pd.read_csv(self.dataset).iloc[train_idx]
            term2count=dict()
            for l in self.labels:
                term2count[l]=len(train_df[train_df[l]==1])
            FREQ_CUTOFF = 0 
            term_freq = sorted([term for term, count in term2count.items() if count>=FREQ_CUTOFF])
            class_freq = [term2count[x] for x in term_freq]
            train_num = len(train_df)
            
            self.criterion = ResampleLoss(reweight_func='rebalance', 
                        loss_weight=1.0, focal=dict(focal=True, alpha=0.5, gamma=2),
                        logit_reg=dict(init_bias=0.05, neg_scale=2.0), map_param=dict(alpha=2.5, beta=10.0, gamma=0.9),
                        class_freq=class_freq, train_num=train_num)
            self.classifier = Classifier(input_size = 768, output_size=7, layer=self.layer, model=self.modelo, tokenizer=self.tokenizer).to(self.device)
            self.optimizer = AdamW(self.classifier.parameters(), lr=self.learning_rate)
            early_stopping = EarlyStopping(self.patience, os.path.join(self.dir_save_models, "checkpoint.pth"), trace_func=self.loggert.debug)
            train_dataset = self.encoded_dataset['dataset'].select(train_idx)
            val_dataset = self.encoded_dataset['dataset'].select(val_idx)
            bestMetrics = {}
            inicio = time.time()
            for epoch in range(self.epochs):
                trainLoss = self.train_step(train_dataset, fold+1, epoch+1)
                valMetrics = self.val_step(val_dataset, fold+1, epoch+1)
                metrics_f1_curve.append(valMetrics['cf_report']['weighted avg']['f1-score'])
                self.loggert.debug("\n| Fold | Epoca | Train Loss | Val Loss | Precisão | Recall |   F1   | Acurácia | Hamming Loss |")
                self.loggert.debug("--------------------------------------------------------------------------------------")
                self.loggert.debug("|  %s  |  %s  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |\n", fold+1, epoch+1,
                                   trainLoss, valMetrics['val_losses'], valMetrics['cf_report']['weighted avg']['precision'],
                                   valMetrics['cf_report']['weighted avg']['recall'], valMetrics['cf_report']['weighted avg']['f1-score'],
                                   valMetrics['cf_report']['accuracy'], valMetrics['hloss'])
                early_stopping(valMetrics['val_losses'], self.classifier, self.optimizer, epoch+1)        
                if early_stopping.early_stop:
                    self.loggert.debug("Early stopping")
                    break
                bestMetrics = save_best_metrics(valMetrics['cf_report']['weighted avg']['f1-score'], valMetrics, trainLoss)
                save_best_model(valMetrics['val_losses'], self.classifier, self.dir_save_models, self.model_name)
            fim = time.time()

            history['train_losses'].append(bestMetrics['train_loss'])
            history['val_losses'].append(bestMetrics['val_losses'])
            history['cf_report'].append(bestMetrics['cf_report'])
            history['hloss'].append(bestMetrics['hloss'])
            history['f1_curve'].append(metrics_f1_curve)
            history['time_per_epoch'].append((fim - inicio)/epoch)
            history['epochs_per_fold'].append(epoch)

        metricas = {
            "model_name": self.model_name,
            "train_loss": np.mean(history['train_losses']),
            "train_loss_curve": history['train_losses'],
            "val_loss": np.mean(history['val_losses']),
            "val_loss_curve": history['val_losses'],
            "cf_report": self.calculate_mean_global_metrics(history['cf_report']),
            "hloss": np.mean(history['hloss']),
            "training_time_epoch": history['time_per_epoch'],
            "mean_training_time_epoch": np.mean(history['time_per_epoch']),
            "f1_curve": history['f1_curve'],
            "epochs_per_fold": history['epochs_per_fold'],
            "mean_epochs": np.mean(history['epochs_per_fold'])
        }
        self.loggert.removeHandler(self.loggert.handlers[0])
        return metricas