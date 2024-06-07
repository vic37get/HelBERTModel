import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback, Trainer, EvalPrediction
import time
import numpy as np
from  distrib_balanced_loss import ResampleLoss


class FineTunningTrainer:
    def __init__(self, batch_size: int, epochs: int, patience: int, model_name: str, dir_save_metrics: str,
                dir_save_models: str, learning_rate: float, modelo: str, tokenizer: str, dataset: str) -> None:
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model_name = model_name
        self.dir_save_metrics = dir_save_metrics
        self.dir_save_models = dir_save_models
        self.learning_rate = learning_rate
        self.modelo = modelo
        self.tokenizer = tokenizer
        self.dir_dataset = dataset
        self.dataset = dataset

        self.dataset = load_dataset("csv", data_files={"dataset": self.dataset})
        self.labels = list(self.dataset['dataset'].features.keys())[1:]
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
        for fold, (train_idx, val_idx) in enumerate(folds):
            
            self.encoded_dataset = self.dataset.map(self.preprocess_data, batched=True, remove_columns=self.dataset['dataset'].column_names)
            self.encoded_dataset.set_format("torch")
            
            train_dataset = self.encoded_dataset['dataset'].select(train_idx)
            val_dataset = self.encoded_dataset['dataset'].select(val_idx)
            model = AutoModelForSequenceClassification.from_pretrained(self.modelo, problem_type="multi_label_classification", num_labels=len(self.labels), id2label=self.id2label, label2id=self.label2id)

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
                weight_decay=0,
                do_train = True,
                do_eval = True,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss"
            )

            train_df = pd.read_csv(self.dir_dataset)
            term2count=dict()
            for l in self.labels:
                term2count[l]=len(train_df[train_df[l]==1])
            FREQ_CUTOFF = 0 
            term_freq = sorted([term for term, count in term2count.items() if count>=FREQ_CUTOFF])
            class_freq = [term2count[x] for x in term_freq]
            train_num = len(train_df)
            
            trainer = WesTrainer(
                model,
                args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience=self.patience)],
                class_freq=class_freq,
                train_num=train_num
            )
            inicio = time.time()
            trainer.train()
            fim = time.time()
            metrics = trainer.evaluate(eval_dataset=val_dataset)
            trainer.save_model('{}-{}'.format(self.dir_save_models, self.model_name))
            
            metricas = {
                "model_name": self.model_name,
                "accuracy": metrics['eval_accuracy'],
                "cf_report": metrics['eval_cf_report'],
                "hammingLoss": float(metrics['eval_hammingLoss']),
                "training_time": fim - inicio
            }
        return metricas

    def preprocess_data(self, examples):
        text = examples["text"]
        for indice in text:
            if not isinstance(indice, str):
                print(indice)
        encoding=self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
        labels_matrix = np.zeros((len(examples["text"]), len(self.labels)))
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]
        encoding["labels"] = labels_matrix.tolist()
        return encoding

    def multi_label_metrics(self, predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        y_true = labels
        # finally, compute metrics
        cf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        hloss = hamming_loss(y_true, y_pred)
        # return as dictionary
        metrics = {
                'accuracy': accuracy,
                'cf_report': cf_report,
                'hammingLoss': hloss}
        return metrics

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = self.multi_label_metrics(predictions=preds.reshape(-1, len(self.labels)), labels=p.label_ids)
        return result

class WesTrainer(Trainer):
    def __init__(self, *args,**kwargs) -> None:
        super().__init__(*args, train_dataset=kwargs.get('train_dataset'), 
        eval_dataset=kwargs.get('eval_dataset'), tokenizer=kwargs.get('tokenizer'), compute_metrics=kwargs.get('compute_metrics'),
        callbacks=kwargs.get('callbacks'))
        self.class_freq = kwargs.get('class_freq')
        self.train_num = kwargs.get('train_num')
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                         focal=dict(focal=True, alpha=0.5, gamma=2),
                         logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                         map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
                         class_freq=self.class_freq, train_num=self.train_num)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1,self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss