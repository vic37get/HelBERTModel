import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")


class Dataset(torch.utils.data.Dataset):    
    def __init__(self, encodings, labels=None):          
        self.encodings = encodings        
        self.labels = labels
     
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["label"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])


class ClassificaTipoObjetos:
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

        print('Carregando modelo e tokenizador...')
        self.modelo = AutoModelForSequenceClassification.from_pretrained(modelo, num_labels=4).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.modelo.resize_token_embeddings(len(self.tokenizer))

        print('Carregando dados...')
        self.treino = pd.read_csv(treino)
        self.teste = pd.read_csv(teste)
        self.validacao = pd.read_csv(validacao)


    def tokeniza_texto(self, texto: str) -> torch.Tensor:
        return self.tokenizer(texto, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    
    def compute_metrics(self, p):    
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average='macro',zero_division=0)
        precision = precision_score(y_true=labels, y_pred=pred, average='macro', zero_division=0)
        f1 = f1_score(y_true=labels, y_pred=pred, average='macro', zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


    def fit(self) -> None:
        tokenized_train = self.tokeniza_texto(self.treino[self.coluna].tolist())
        tokenized_test = self.tokeniza_texto(self.teste[self.coluna].tolist())
        tokenized_val = self.tokeniza_texto(self.validacao[self.coluna].tolist())

        train_dataset = Dataset(tokenized_train, self.treino['label'].tolist())
        val_dataset = Dataset(tokenized_test, self.teste['label'].tolist())
        test_dataset = Dataset(tokenized_val, self.validacao['label'].tolist())

        args = TrainingArguments(
            output_dir=self.dir_save_models,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            load_best_model_at_end=True,
            save_total_limit=1,
            fp16=True,
            seed = 0,
            metric_for_best_model = 'eval_loss',
            greater_is_better = False,
            gradient_checkpointing = True,
            do_train = True,
            do_eval = True,
            evaluation_strategy = 'epoch',
            logging_strategy = 'epoch',
            save_strategy = 'epoch',
        )

        trainer = Trainer(
            model=self.modelo,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)]
        )
        
        trainer.train()

        print("Realizando a predição para a base de teste.")
        metrics = trainer.predict(test_dataset).metrics
        metrics['name_model'] = self.model_name

        print("Salvando o modelo...")
        self.modelo.save_pretrained(os.path.join(self.dir_save_models, '{}'.format(self.model_name)))

        return metrics
    
    def predict(self, texto: str) -> int:
        tokenized_test = self.tokeniza_texto(self.teste[self.coluna].tolist())
        dataset_test = Dataset(tokenized_test)
        trainer = Trainer(model=self.modelo)
        metrics = trainer.predict(dataset_test).metrics
        return metrics
