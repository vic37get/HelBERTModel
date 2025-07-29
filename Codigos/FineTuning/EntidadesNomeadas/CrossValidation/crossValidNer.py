from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import evaluate 
from transformers.trainer_callback import EarlyStoppingCallback
import numpy as np
from datasets import DatasetDict
import os
import torch
import shutil
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SaveBestMetrics:
    def __init__(self,best_f1=float('-inf')):
        self.best_f1 = best_f1
        self.best_metrics = None

    def __call__(self, current_f1: float, current_metrics: dict, train_loss: float):
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_metrics=current_metrics
            self.best_metrics['train_loss']=train_loss
        return self.best_metrics


class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss: float, model: AutoModelForTokenClassification, dir_save_models: str, nameModel: str, id2label: dict, label2id: dict):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            model_save = os.path.join(dir_save_models, 'best_model-{}'.format(nameModel))
            if os.path.exists(model_save):
                shutil.rmtree(model_save)
            model.save_pretrained(model_save)


class CrossValidNer:
    def __init__(self, batch_size: int, epochs: int, patience: int, dir_save_models: str, learning_rate: float) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.dir_save_models = dir_save_models
        self.learning_rate = learning_rate
        self.device = torch.device('cuda')

    
    def tokenize_and_align_labels(self, examples: DatasetDict) -> DatasetDict:
        """
        Função de tokenização e alinhamento das labels.
        """
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=1024, padding="max_length")
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs


    def compute_metrics(self, p) -> dict:
        """
        Função que calcula as métricas
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


    def train(self, folds: list, dataset: DatasetDict, nameModel: str, dirModel: str, dirTokenizer: str) -> dict:
        """
        Função de treinamento do modelo.
        """
        print('Iniciando treinamento...')    
        # Quando não está no formato do argilla
        # self.label_list = json.load(open("/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/EntidadesNomeadas/ner_pt/ner_pt_treino_dataset/ner_pt_treino_dataset_features.json"))
        # id2label = {i: label for i, label in enumerate(self.label_list)}
        # label2id = {label: i for i, label in enumerate(self.label_list)}

        self.label_list = dataset.features['ner_tags'].feature.names
        id2label = {i: label for i, label in enumerate(self.label_list)}
        label2id = {label: i for i, label in enumerate(self.label_list)}

        save_best_metrics = SaveBestMetrics()
        save_best_model = SaveBestModel()
        inicio = time.time()
        history={'train_losses':[], 'val_losses':[], 'precision':[], 'recall':[], 'f1':[], 'accuracy':[], 'hloss':[]}
        for fold,(train_idx, val_idx) in enumerate(folds):
            self.model = AutoModelForTokenClassification.from_pretrained(dirModel, num_labels=len(self.label_list), id2label=id2label, label2id=label2id).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(dirTokenizer)
            self.model.resize_token_embeddings(len(self.tokenizer))
            dataset = dataset.map(self.tokenize_and_align_labels, batched=True)
            data_collator = DataCollatorForTokenClassification(self.tokenizer)
            self.metric = evaluate.load("seqeval")

            # train_dataset = dataset['train'].select(train_idx)
            # val_dataset = dataset['train'].select(val_idx)

            train_dataset = dataset.select(train_idx)
            val_dataset = dataset.select(val_idx)

            args = TrainingArguments(
                output_dir=self.dir_save_models,
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=self.epochs,
                save_total_limit=1,
                load_best_model_at_end=True,
                fp16=True,
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
                model=self.model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)],
            )

            print(f'Treinando fold {fold+1}...')
            trainer.train()
            metrics = trainer.evaluate(eval_dataset=val_dataset)
            bestMetrics = save_best_metrics(metrics['eval_f1'], metrics, metrics['eval_loss'])
            save_best_model(metrics['eval_loss'], self.model, self.dir_save_models, nameModel, id2label, label2id)

            history['val_losses'].append(float(bestMetrics['eval_loss']))
            history['precision'].append(float(bestMetrics['eval_precision']))
            history['recall'].append(float(bestMetrics['eval_recall']))
            history['f1'].append(float(bestMetrics['eval_f1']))
            history['accuracy'].append(float(bestMetrics['eval_accuracy']))
        
        fim = time.time()
        metricas = {
            "model_name": nameModel,
            "val_loss": np.mean(history['val_losses']),
            "precisao": np.mean(history['precision']),
            "recall": np.mean(history['recall']),
            "f1": np.mean(history['f1']),
            "acuracia": np.mean(history['accuracy']),
            "training_time": fim - inicio
        }
        return metricas