import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from tqdm.auto import tqdm
import sys
import os
from transformers import AutoModel, AutoTokenizer, logging
from torch.utils.data import DataLoader
from statistics import mean 
import logging
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
sys.path.insert(0, '../../../')
from utils.earlyStopping import EarlyStopping
from utils.classifier import Classifier
from utils.myDataset import MyDataset


class ClassificaIndicios:
    def __init__(self, batch_size: int, epochs: int, patience: int, model_name: str, dir_save_metrics: str,
                  dir_save_models: str, learning_rate: float, modelo: str, tokenizer: str, treino: str, 
                  validacao: str, teste: str, coluna: str) -> None:
        
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
        self.validacao = pd.read_csv(validacao)
        self.teste = pd.read_csv(teste)

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
        return [X, Y]
        

    def train_step(self, train_loader: DataLoader, epoca: int) -> float:
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
            self.optimizer.zero_grad(set_to_none=True)
            loss=None
            labels=torch.tensor(labels, dtype=float).to(self.device)
            loss=self.criterion(logits.squeeze(0), labels)
            loss.backward()
            loop.set_description(f'Treinamento - {self.model_name} | Época: {epoca}')
            loop.set_postfix(loss=loss.item())
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().numpy()))
        return mean(losses)
    

    def val_step(self, val_loader: DataLoader, epoch: int):
        """
        Função que realiza a validação do modelo.
        Parâmetros:
            val_loader: dataloader de validação
            device: dispositivo a ser utilizado
            epoch: época atual
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
            labels=torch.tensor(labels,dtype=float).to(self.device)
            loss=self.criterion(logits.squeeze(0),labels)
            loop.set_description(f'Validação - {self.model_name} | Época: {epoch}')
            loop.set_postfix(loss=loss.item())
            val_losses.append(float(loss.detach().cpu().numpy()))
            probs=torch.sigmoid(logits)
            predictions=torch.clone(probs)
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            preds.append(torch.tensor(predictions.cpu().detach().numpy()).squeeze(0))
            trues.append(torch.tensor(labels.cpu().detach().numpy()))     
        y_true = torch.cat(trues,0)
        y_pred = torch.cat(preds,0)
        precisao = precision_score(y_true, y_pred,average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred,average='weighted', zero_division=0)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
        acuracia = accuracy_score(y_true, y_pred)
        cf_report = classification_report(y_true, y_pred, output_dict=True)
        return {'epoca': epoch, 'val_losses': mean(val_losses), 'precision': precisao, 'recall': recall, 'f1': f1, 'accuracy': acuracia, 'cf_report': cf_report}


    def test_step(self, test_loader: DataLoader):
        """
        Função que realiza o teste do modelo.
        Parâmetros:
            test_loader: dataloader de validação
            device: dispositivo a ser utilizado
            epoch: época atual
        Retorno:
            metricas: dicionário com as métricas de teste.
        """
        self.classifier.eval()        
        test_losses, trues, preds= [], [], []
        loop = tqdm(test_loader, leave=True, colour='yellow')
        for embeddings, labels in loop:
            embeddings=torch.stack(embeddings).to(self.device)
            logits=self.classifier(embeddings)
            loss=None
            labels=torch.tensor(labels,dtype=float).to(self.device)
            loss=self.criterion(logits.squeeze(0),labels)
            loop.set_description(f'Executando teste - {self.model_name}')
            loop.set_postfix(loss=loss.item())
            test_losses.append(float(loss.detach().cpu().numpy()))
            probs=torch.sigmoid(logits)
            predictions=torch.clone(probs)
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            preds.append(torch.tensor(predictions.cpu().detach().numpy()).squeeze(0))
            trues.append(torch.tensor(labels.cpu().detach().numpy()))     
        y_true = torch.cat(trues,0)
        y_pred = torch.cat(preds,0)
        precisao = precision_score(y_true, y_pred,average='weighted',zero_division=0)
        recall = recall_score(y_true, y_pred,average='weighted', zero_division=0)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
        acuracia = accuracy_score(y_true, y_pred)
        cf_report = classification_report(y_true, y_pred, output_dict=True)
        return {'test_losses': mean(test_losses),'precision': precisao,'recall': recall,'f1': f1,'accuracy': acuracia, 'cf_report': cf_report}
    

    def modelTraining(self, train_dataset: MyDataset, val_dataset: MyDataset, test_dataset: MyDataset) -> None:
        """
        Função que realiza o treinamento do modelo.
        Parâmetros:
            train_dataset: dataset de treino
        """ 
        self.criterion = BCEWithLogitsLoss()
        self.classifier = Classifier(input_size = 768, output_size=7).to(self.device)
        self.optimizer = AdamW(self.classifier.parameters(), lr=self.learning_rate)
        early_stopping = EarlyStopping(self.patience, os.path.join(self.dir_save_models, f"{self.model_name}.pth"), trace_func=self.loggert.debug)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, collate_fn=self.collate_func)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, collate_fn=self.collate_func)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, collate_fn=self.collate_func)

        inicio = time.time()
        metrics_curve = []
        for epoch in range(self.epochs):
            trainLoss = self.train_step(train_loader, epoch+1)
            valMetrics = self.val_step(val_loader, epoch+1)
            metrics_curve.append({'epoch': epoch+1, 'f1': valMetrics['f1'], 'val_loss': valMetrics['val_losses'], 'cf_report': valMetrics['cf_report']})
            self.loggert.debug("\n| Epoca | Train Loss | Val Loss | Precisão | Recall |   F1   | Acurácia |")
            self.loggert.debug("--------------------------------------------------------------------------------------")
            self.loggert.debug("|  %s  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |\n", epoch+1, trainLoss, valMetrics['val_losses'], valMetrics['precision'], valMetrics['recall'], valMetrics['f1'], valMetrics['accuracy'])
            early_stopping(valMetrics['val_losses'], self.modelo, self.optimizer, epoch+1)        
            if early_stopping.early_stop:
                self.loggert.debug("Early stopping")
                break

        fim = time.time()
        testMetrics = self.test_step(test_loader) 

        metricas = {
            "model_name": self.model_name,
            "test_loss": testMetrics['test_losses'],
            "precisao": testMetrics['precision'],
            "recall": testMetrics['recall'],
            "f1": testMetrics['f1'],
            "acuracia": testMetrics['accuracy'],
            "cf_report": testMetrics['cf_report'],
            "training_epoch_time": (fim - inicio)/epoch+1,
            "metrics_curve": metrics_curve,
            "epochs": epoch+1
        }
        # Salvando o modelo e suas configurações
        model_save = os.path.join(self.dir_save_models, f"{self.model_name}-model.pth")
        torch.save({'model_state_dict': self.classifier.state_dict()}, model_save)
        self.loggert.removeHandler(self.loggert.handlers[0])
        return metricas