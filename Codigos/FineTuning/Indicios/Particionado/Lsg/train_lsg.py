import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
import torch
import datasets
datasets.disable_caching()
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback, Trainer, EvalPrediction
import time
import os
import numpy as np
from  distrib_balanced_loss import ResampleLoss


class TrainLsgIndicios:
    def __init__(self, batch_size: int, epochs: int, patience: int, model_name: str, dir_save_metrics: str,
                dir_save_models: str, learning_rate: float, modelo: str, tokenizer: str, train: str, test: str, val: str,
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
        self.dataset = train
        self.gradient_accumulation_steps = gradient_accumulation_steps       

        self.data = load_dataset("csv", data_files={"train": train, "test": test, "val": val})
        # self.data = load_dataset('tcepi/bidCorpus_raw', 'bidCorpus_gold')
        self.labels = list(self.data['train'].features.keys())[1:]
        self.id2label = {idx:label for idx, label in enumerate(self.labels)}
        self.label2id = {label:idx for idx, label in enumerate(self.labels)}
        
        print("Carregando modelo e tokenizador..")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(self.modelo, problem_type="multi_label_classification",
                                                                    num_labels=len(self.labels), id2label=self.id2label,
                                                                    label2id=self.label2id, trust_remote_code=True)
        self.modelo.resize_token_embeddings(len(self.tokenizer))
        
        self.encoded_dataset = self.data.map(self.preprocess_data, batched=True, remove_columns=self.data['train'].column_names)
        self.encoded_dataset.set_format("torch")  
        
        
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
    
    
    def get_info_dataset(self, freq_cutoff=0):
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
    
    
    def train(self):
        
        train_dataset = self.encoded_dataset['train']
        test_dataset = self.encoded_dataset['test']
        class_freq, train_num = self.get_info_dataset(freq_cutoff=0)

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
        
        trainer = WesTrainer(
            self.modelo,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=self.patience)],
            class_freq=class_freq,
            train_num=train_num
        )
        inicio = time.time()
        trainer.train()
        fim = time.time()
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.save_model(os.path.join(self.dir_save_models, self.model_name))
        trainer.save_metrics(os.path.join(self.dir_save_metrics, "{}.json".format(self.model_name)), metrics)
            
        metricas = {
            "model_name": self.model_name,
            "accuracy": float(metrics['eval_accuracy']),
            "cf_report": metrics['eval_cf_report'],
            "hammingLoss": float(metrics['eval_hammingLoss']),
            "training_time": fim - inicio
        }
        return metricas

    def preprocess_data(self, examples):
        text = examples["text"]
        encoding=self.tokenizer(text, padding="max_length", truncation=True, max_length=4096, return_tensors='pt')
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