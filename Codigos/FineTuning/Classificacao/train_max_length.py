import pandas as pd
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback, Trainer
from  distrib_balanced_loss import ResampleLoss
import time
import os
import numpy as np


class TrainMaxLength:
    def __init__(self,
                 dataset: DatasetDict,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 patience: int,
                 dir_save_models: str,
                 dir_save_metrics: str,
                 gradient_accumulation_steps: int,
                 max_length: int,
                 modelo: str,
                 model_name: str,
                 metrics_name: str,
                 cross_validation: bool,
                 file_folds: str,
                 tipo_classificacao: str,
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
        self.max_length = max_length
        self.modelo = modelo
        self.model_name = model_name
        self.metrics_name = metrics_name
        self.cross_validation = cross_validation
        self.file_folds = file_folds
        self.tipo_classificacao = tipo_classificacao
        self.funcao_perda_ponderada = funcao_perda_ponderada
        self.huggingface_dataset = huggingface_dataset
        self.token = token
        
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
        dataset = pd.DataFrame(self.dataset['train'])
        term2count = {label: sum(dataset[label]) for label in self.labels}
        term_freq = sorted([term for term, count in term2count.items() if count >= freq_cutoff])
        class_freq = [term2count[term] for term in term_freq]
        train_num = len(dataset)
        return class_freq, train_num
    
    
    def train(self):
        
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
        
        if self.cross_validation:
            self.logger.info("Treinamento com Cross-Validation...")
            self.logger.info("Tipo de classificação: {}".format(self.tipo_classificacao))
            folds = np.load(self.file_folds, allow_pickle=True)
            history = {'accuracy': [], 'cf_report': [], 'hammingLoss': []}
            
            for fold, (train_idx, val_idx) in enumerate(folds):
                
                train_dataset = self.encoded_dataset['train'].select(train_idx)
                val_dataset = self.encoded_dataset['train'].select(val_idx)

                if self.tipo_classificacao == 'multiclasse':
                    self.logger.info(f"Treinando o modelo no {fold+1}º fold...")
                    model = AutoModelForSequenceClassification.from_pretrained(self.modelo, num_labels=len(self.labels), id2label=self.id2label,
                                                                               label2id=self.label2id, trust_remote_code=True)
                    model.resize_token_embeddings(len(self.tokenizador))
                    trainer = Trainer(
                        model=model,
                        args=args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        compute_metrics=self.compute_metrics_multiclasse,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)]
                    )
                    
                elif self.tipo_classificacao == 'multilabel':
                    self.logger.info(f"Treinando o modelo no {fold+1}º fold...")
                    model = AutoModelForSequenceClassification.from_pretrained(self.modelo, problem_type="multi_label_classification",
                                                                               num_labels=len(self.labels), id2label=self.id2label,
                                                                               label2id=self.label2id, trust_remote_code=True)
                    model.resize_token_embeddings(len(self.tokenizador))
                    class_freq, train_num = self.get_info_dataset(freq_cutoff=0)
                    trainer = WesTrainer(
                        model=model,
                        args=args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        tokenizer=self.tokenizador,
                        compute_metrics=self.compute_metrics_multilabel,
                        callbacks = [EarlyStoppingCallback(early_stopping_patience=self.patience)],
                        class_freq=class_freq,
                        train_num=train_num
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
        
        else:
            self.logger.info("Treinamento sem Cross-Validation...")
            self.logger.info("Tipo de classificação: {}".format(self.tipo_classificacao))
            train_dataset = self.encoded_dataset['train']
            val_dataset = self.encoded_dataset['test']
            
            if self.tipo_classificacao == 'multiclasse':
                model = AutoModelForSequenceClassification.from_pretrained(self.modelo, num_labels=len(self.labels), id2label=self.id2label,
                                                                           label2id=self.label2id, trust_remote_code=True)
                model.resize_token_embeddings(len(self.tokenizador))
                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=self.compute_metrics_multiclasse,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)]
                )
                
            elif self.tipo_classificacao == 'multilabel':
                model = AutoModelForSequenceClassification.from_pretrained(self.modelo, problem_type="multi_label_classification",
                                                                           num_labels=len(self.labels), id2label=self.id2label,
                                                                           label2id=self.label2id, trust_remote_code=True)
                model.resize_token_embeddings(len(self.tokenizador))
                class_freq, train_num = self.get_info_dataset(freq_cutoff=0)
                trainer = WesTrainer(
                    model,
                    args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    tokenizer=self.tokenizador,
                    compute_metrics=self.compute_metrics_multilabel,
                    callbacks = [EarlyStoppingCallback(early_stopping_patience=self.patience)],
                    class_freq=class_freq,
                    train_num=train_num
                )
            
            inicio = time.time()
            trainer.train()
            fim = time.time()
            metrics = trainer.evaluate(eval_dataset=val_dataset)
            trainer.save_model(os.path.join(self.dir_save_models, self.model_name))
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
        encoding=self.tokenizador(text, truncation=True, padding='max_length', max_length=self.max_length)        
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

    def compute_metrics_multilabel(self, p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = self.multi_label_metrics(predictions=preds.reshape(-1, len(self.labels)), labels=p.label_ids)
        return result
    
    def compute_metrics_multiclasse(self, p):    
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        cf_report = classification_report(labels, pred, output_dict=True, zero_division=1)
        accuracy = accuracy_score(labels, pred)
        hloss = hamming_loss(labels, pred)
        metrics = {'accuracy': accuracy,
                   'cf_report': cf_report,
                   'hammingLoss': hloss}
        return metrics

class WesTrainer(Trainer):
    def __init__(self, *args,**kwargs) -> None:
        super().__init__(*args, train_dataset=kwargs.get('train_dataset'), 
        eval_dataset=kwargs.get('eval_dataset'), tokenizer=kwargs.get('tokenizer'), compute_metrics=kwargs.get('compute_metrics'),
        callbacks=kwargs.get('callbacks'))
        self.class_freq = kwargs.get('class_freq')
        self.train_num = kwargs.get('train_num')
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                         focal=dict(focal=True, alpha=0.5, gamma=2),
                         logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                         map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
                         class_freq=self.class_freq, train_num=self.train_num)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1,self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss