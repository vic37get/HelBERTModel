from datasets import load_dataset
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
import datasets
datasets.disable_caching()
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback, Trainer
import time
import os
import numpy as np


class TrainLsgObjetos:
    def __init__(self, batch_size: int, epochs: int, patience: int, model_name: str, dir_save_metrics: str,
                dir_save_models: str, learning_rate: float, modelo: str, tokenizer: str, dataset: str,
                gradient_accumulation_steps: int) -> None:
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model_name = model_name
        self.dir_save_metrics = dir_save_metrics
        self.dir_save_models = dir_save_models
        self.learning_rate = learning_rate
        self.modelo = modelo
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.data = load_dataset(self.dataset, token="")
        self.labels = list(set(self.data['train']['label']))
        self.id2label = {idx:label for idx, label in enumerate(self.labels)}
        self.label2id = {label:idx for idx, label in enumerate(self.labels)}
        
        print("Carregando tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        
        
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
    
    def train(self, folds: list):
        history={'accuracy':[], 'cf_report':[], 'hammingLoss':[]}
        for fold, (train_idx, val_idx) in enumerate(folds):
            
            self.encoded_dataset = self.data.map(self.preprocess_data, batched=True, remove_columns=self.data['train'].column_names)
            self.encoded_dataset.set_format("torch")
            
            train_dataset = self.encoded_dataset['train'].select(train_idx)
            val_dataset = self.encoded_dataset['train'].select(val_idx)
            model = AutoModelForSequenceClassification.from_pretrained(self.modelo, num_labels=len(self.labels), id2label=self.id2label,
                                                                       label2id=self.label2id, trust_remote_code=True)
            model.resize_token_embeddings(len(self.tokenizer))

            args = TrainingArguments(
                output_dir=self.dir_save_models,
                evaluation_strategy = "epoch",
                save_strategy = "epoch",
                save_total_limit = 1,
                greater_is_better = False,
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=self.epochs,
                fp16=True,
                weight_decay=0,
                do_train = True,
                do_eval = True,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                gradient_accumulation_steps=self.gradient_accumulation_steps
            )
            
            trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)]
        )
            
            inicio = time.time()
            trainer.train()
            fim = time.time()
            metrics = trainer.evaluate(eval_dataset=val_dataset)
            trainer.save_model(os.path.join(self.dir_save_models, self.model_name))
            history['accuracy'].append(float(metrics['eval_accuracy']))
            history['cf_report'].append(metrics['eval_cf_report'])
            history['hammingLoss'].append(float(metrics['eval_hammingLoss']))
            
        metricas = {
            "model_name": self.model_name,
            "accuracy": np.mean(history['accuracy']),
            "cf_report": self.calculate_mean_global_metrics(history['cf_report']),
            "hammingLoss": np.mean(history['hammingLoss']),
            "training_time": fim - inicio
        }
        return metricas

    def preprocess_data(self, examples):
        text = examples["text"]
        encoding=self.tokenizer(text, padding="max_length", truncation=True, max_length=4096, return_tensors='pt')
        encoding['labels'] = examples['label']
        return encoding
    
    def compute_metrics(self, p):    
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        cf_report = classification_report(labels, pred, output_dict=True, zero_division=0)
        accuracy = accuracy_score(labels, pred)
        hloss = hamming_loss(labels, pred)
        metrics = {'accuracy': accuracy,
                   'cf_report': cf_report,
                   'hammingLoss': hloss}
        return metrics