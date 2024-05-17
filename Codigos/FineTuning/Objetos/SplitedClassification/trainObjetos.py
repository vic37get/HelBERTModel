import pandas as pd
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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss, classification_report
sys.path.insert(0, '../../../')
from utils.earlyStopping import EarlyStopping
from utils.classifier import Classifier
from utils.myDataset import MyDataset


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


class ClassificaTiposObjetos:
    def __init__(self, batch_size: int, epochs: int, patience: int, model_name: str, dir_save_metrics: str,
                  dir_save_models: str, learning_rate: float, modelo: str, tokenizer: str, treino: str, teste: str, validacao: str, coluna: str) -> None:
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
        self.teste = pd.read_csv(teste)
        self.validacao = pd.read_csv(validacao)

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
            loss=None
            labels=torch.tensor(labels, dtype=torch.long).to(self.device)
            loss=self.criterion(logits.squeeze(0), labels)
            self.optimizer.zero_grad()
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
            labels=torch.tensor(labels,dtype=torch.long).to(self.device)
            loss=self.criterion(logits.squeeze(0), labels)
            loop.set_description(f'Validação - {self.model_name} | Época: {epoch}')
            loop.set_postfix(loss=loss.item())
            val_losses.append(float(loss.detach().cpu().numpy()))
            probs = torch.argmax(torch.softmax(logits.squeeze(0), dim=1), dim=1)
            predictions=torch.clone(probs)
            preds.append(torch.tensor(predictions.cpu().detach().numpy()))
            trues.append(torch.tensor(labels.cpu().detach().numpy()))
        y_true=torch.cat(trues, 0)
        y_pred=torch.cat(preds, 0)
        precisao=precision_score(y_true, y_pred,average='weighted',zero_division=0)
        recall=recall_score(y_true, y_pred,average='weighted', zero_division=0)
        f1=f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
        acuracia=accuracy_score(y_true, y_pred)
        return {'epoca': epoch, 'val_losses':mean(val_losses),'precision':precisao,'recall':recall,'f1':f1,'accuracy':acuracia}


    def test_setp(self, test_loader: DataLoader):
        """
        Função que realiza a validação do modelo.
        Parâmetros:
            test_loader: dataloader de teste
        """
        self.classifier.eval()        
        preds=[]
        trues=[]
        test_losses=[]
        loop = tqdm(test_loader, leave=True, colour='yellow')
        for embeddings, labels in loop:
            embeddings=torch.stack(embeddings).to(self.device)
            logits=self.classifier(embeddings)
            loss=None
            labels=torch.tensor(labels,dtype=torch.long).to(self.device)
            loss=self.criterion(logits.squeeze(0), labels)
            loop.set_description(f'Teste - {self.model_name}')
            loop.set_postfix(loss=loss.item())
            test_losses.append(float(loss.detach().cpu().numpy()))
            probs = torch.argmax(torch.softmax(logits.squeeze(0), dim=1), dim=1)
            predictions=torch.clone(probs)
            preds.append(torch.tensor(predictions.cpu().detach().numpy()))
            trues.append(torch.tensor(labels.cpu().detach().numpy()))
        y_true=torch.cat(trues, 0)
        y_pred=torch.cat(preds, 0)
        precisao=precision_score(y_true, y_pred,average='weighted',zero_division=0)
        recall=recall_score(y_true, y_pred,average='weighted', zero_division=0)
        f1=f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
        acuracia=accuracy_score(y_true, y_pred)
        cf_report = classification_report(y_true, y_pred, output_dict=True)
        return {'test_losses':mean(test_losses),'precision':precisao,'recall':recall,'f1':f1,'accuracy':acuracia, 'cf_report':cf_report}
    
    def modelTraining(self, train_dataset: MyDataset, val_dataset: MyDataset, test_dataset: MyDataset) -> dict:
        """
        Função que realiza o treinamento do modelo.
        Parâmetros:
            train_dataset: dataset de treino
            val_dataset: dataset de validação
            test_dataset: dataset de teste
        """
        metrics_f1_curve = []

        self.criterion = CrossEntropyLoss()
        self.classifier = Classifier(input_size = 768, output_size=4).to(self.device)
        self.optimizer = AdamW(self.classifier.parameters(), lr=self.learning_rate)
        early_stopping = EarlyStopping(self.patience, os.path.join(self.dir_save_models, "checkpoint.pth"), trace_func=self.loggert.debug)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, collate_fn=self.collate_func)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, collate_fn=self.collate_func)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, collate_fn=self.collate_func)
    
        inicio = time.time()
        for epoch in range(self.epochs):
            trainLoss = self.train_step(train_loader, epoch+1)
            valMetrics = self.val_step(val_loader, epoch+1)
            metrics_f1_curve.append(valMetrics['f1'])
            self.loggert.debug("\n| Epoca | Train Loss | Val Loss | Precisão | Recall |   F1   | Acurácia |")
            self.loggert.debug("--------------------------------------------------------------------------------------")
            self.loggert.debug("|  %s  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |  %.4f  |\n", epoch+1, trainLoss, valMetrics['val_losses'], valMetrics['precision'], valMetrics['recall'], valMetrics['f1'], valMetrics['accuracy'])
            early_stopping(valMetrics['val_losses'], self.classifier, self.optimizer, epoch+1)        
            if early_stopping.early_stop:
                self.loggert.debug("Early stopping")
                break
        fim = time.time()
        testMetrics = self.test_setp(test_loader)
        metricas = {
            "model_name": self.model_name,
            "test_loss": testMetrics['test_losses'],
            "precisao": testMetrics['precision'],
            "recall": testMetrics['recall'],
            "f1": testMetrics['f1'],
            "acuracia": testMetrics['accuracy'],
            "cf_report": testMetrics['cf_report'],
            "training_epoch_time": (fim - inicio)/epoch+1,
            "f1_curve": metrics_f1_curve,
            "epochs": epoch+1            
        }

        torch.save({'model_state_dict': self.classifier.state_dict()}, os.path.join(self.dir_save_models, '{}-model-{}.pth'.format(self.model_name, self.coluna)))
        self.loggert.removeHandler(self.loggert.handlers[0])
        return metricas