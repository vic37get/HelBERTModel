import pandas as pd
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
import sys
import os
from transformers import AutoModel, AutoTokenizer, logging
from torch.utils.data import DataLoader, SubsetRandomSampler
from statistics import mean 
import logging
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss
sys.path.insert(0, '../../../../')
from utils.earlyStopping import EarlyStopping
from utils.classifier_secoes import Classifier
from utils.myDataset import MyDataset
from utils.modelSaves import SaveBestMetrics


class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss: float, model: AutoModel,
                  dir_save_models: str, nameModel: str, coluna: str):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            model_save = os.path.join(dir_save_models, 'best_model-{}-{}.pth'.format(nameModel, coluna))
            if os.path.exists(model_save):
                os.remove(model_save)
            torch.save({
                'model_state_dict': model.state_dict(),
                }, model_save)


class ClassificaSecoes:
    def __init__(self, batch_size: int, epochs: int, patience: int, model_name: str, dir_save_metrics: str,
                  dir_save_models: str, learning_rate: float, modelo: str, tokenizer: str, treino: str, coluna: str) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model_name = model_name
        self.dir_save_metrics = dir_save_metrics
        self.dir_save_models = dir_save_models
        self.learning_rate = learning_rate
        self.coluna = coluna
        self.device = torch.device('cuda')

        print('Carregando dados...')
        self.treino = pd.read_csv(treino)

        print('Carregando modelo e tokenizador...')

        self.modelo = AutoModel.from_pretrained(modelo, output_hidden_states=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.modelo.resize_token_embeddings(len(self.tokenizer))

        self.loggert = logging.getLogger(name='tela')
        self.loggert.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        handler_screen = logging.StreamHandler()
        handler_screen.setFormatter(formatter)
        handler_screen.setLevel(logging.DEBUG)
        self.loggert.addHandler(handler_screen)


    def collate_func(self, batch: list) -> list:
        """
        Função que prepara o batch para ser passado para o modelo.
        Parâmetros:
            batch: batch de dados
        Retorno:
            X: lista com os textos
            Y: lista com os labels
        """
        X = [x[0] for x in batch]
        Y = [x[1] for x in batch]
        return [X,Y]
    

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
        loop = tqdm(train_loader, leave=True, colour='green')
        for embeddings, labels in loop:
            embeddings=torch.stack(embeddings).to(self.device)
            logits=self.classifier(embeddings)
            loss=None
            labels=torch.tensor(labels, dtype=torch.long).to(self.device)
            loss=self.criterion(logits.squeeze(0), labels)
            self.optimizer.zero_grad()
            loss.backward()
            loop.set_description(f'Treinamento - {self.model_name} | Fold: {fold} | Época: {epoca}')
            loop.set_postfix(loss=loss.item())
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().numpy()))
        return mean(losses)
    

    def val_step(self, val_loader: DataLoader, epoch: int, fold: int):
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
        loop = tqdm(val_loader, leave=True, colour='blue')
        for embeddings, labels in loop:
            embeddings=torch.stack(embeddings).to(self.device)
            logits=self.classifier(embeddings)
            loss=None
            labels=torch.tensor(labels,dtype=torch.long).to(self.device)
            loss=self.criterion(logits.squeeze(0), labels)
            loop.set_description(f'Validação - {self.model_name} | Fold: {fold} | Época: {epoch}')
            loop.set_postfix(loss=loss.item())
            val_losses.append(float(loss.detach().cpu().numpy()))
            probs = torch.argmax(torch.softmax(logits.squeeze(0), dim=1), dim=1)
            predictions=torch.clone(probs)
            preds.append(torch.tensor(predictions.cpu().detach().numpy()))
            trues.append(torch.tensor(labels.cpu().detach().numpy()))
        y_true=torch.cat(trues, 0)
        y_pred=torch.cat(preds, 0)
        precisao=precision_score(y_true, y_pred,average='macro',zero_division=0)
        recall=recall_score(y_true, y_pred,average='macro')
        f1=f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        acuracia=accuracy_score(y_true, y_pred)
        hl=hamming_loss(y_true, y_pred)
        return {'fold': fold, 'epoca':epoch, 'val_losses':mean(val_losses),'precision':precisao,'recall':recall,'f1':f1,'accuracy':acuracia,'hloss':hl}

    
    def modelTraining(self, folds: list, train_dataset: MyDataset) -> None:
        """
        Função que realiza o treinamento do modelo.
        Parâmetros:
            folds: folds de treino e validação
            train_dataset: dataset de treino
        """
        history={'train_losses':[],'val_losses':[],'precision':[],'recall':[],'f1':[],'accuracy':[],'hloss':[], 'time_per_epoch':[], 'epochs_per_fold':[]}
        save_best_model = SaveBestModel()
        save_best_metrics = SaveBestMetrics()
        for fold,(train_idx, val_idx) in enumerate(folds):
            self.criterion = CrossEntropyLoss()
            self.classifier = Classifier(input_size = 768, output_size=5).to(self.device)
            self.optimizer = AdamW(self.classifier.parameters(), lr=self.learning_rate)
            early_stopping = EarlyStopping(self.patience, os.path.join(self.dir_save_models, "checkpoint.pth"), trace_func=self.loggert.debug)
            train_sampler=SubsetRandomSampler(train_idx)
            val_sampler=SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, collate_fn=self.collate_func, sampler=train_sampler)
            val_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, collate_fn=self.collate_func, sampler=val_sampler)
            bestMetrics = {}
            inicio = time.time()
            for epoch in range(self.epochs):
                trainLoss = self.train_step(train_loader, fold+1, epoch+1)
                valMetrics = self.val_step(val_loader, epoch+1, fold+1)
                self.loggert.debug("\n| Fold | Epoca | Train Loss | Val Loss | Precisão | Recall |   F1   | Acurácia | Hamming Loss |")
                self.loggert.debug("--------------------------------------------------------------------------------------")
                self.loggert.debug("|  %s  |  %s  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |\n", fold+1, epoch+1, trainLoss, valMetrics['val_losses'], valMetrics['precision'], valMetrics['recall'], valMetrics['f1'], valMetrics['accuracy'], valMetrics['hloss'])
                early_stopping(valMetrics['val_losses'], self.classifier, self.optimizer, epoch+1)        
                if early_stopping.early_stop:
                    self.loggert.debug("Early stopping")
                    break
                bestMetrics = save_best_metrics(valMetrics['f1'], valMetrics, trainLoss)
                save_best_model(valMetrics['val_losses'], self.classifier, self.dir_save_models, self.model_name, self.coluna)
            fim = time.time()
          
            history['train_losses'].append(bestMetrics['train_loss'])
            history['val_losses'].append(bestMetrics['val_losses'])
            history['precision'].append(bestMetrics['precision'])
            history['recall'].append(bestMetrics['recall'])
            history['f1'].append(bestMetrics['f1'])
            history['accuracy'].append(bestMetrics['accuracy'])
            history['hloss'].append(bestMetrics['hloss'])
            history['time_per_epoch'].append((fim - inicio)/epoch)
            history['epochs_per_fold'].append(epoch)

        metricas = {
            "model_name": self.model_name,
            "train_loss": np.mean(history['train_losses']),
            "val_loss": np.mean(history['val_losses']),
            "precisao": np.mean(history['precision']),
            "recall": np.mean(history['recall']),
            "f1": np.mean(history['f1']),
            "acuracia": np.mean(history['accuracy']),
            "hloss": np.mean(history['hloss']),
            "training_time_epoch": history['time_per_epoch'],
            "mean_training_time_epoch": np.mean(history['time_per_epoch']),
            "epochs_per_fold": history['epochs_per_fold'],
            "mean_epochs": np.mean(history['epochs_per_fold'])
        }
        self.loggert.removeHandler(self.loggert.handlers[0])
        return metricas