from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import load_from_disk, load_dataset
import evaluate 
from transformers.trainer_callback import EarlyStoppingCallback
import numpy as np
import os
import shutil
import sys
sys.path.insert(0, '../../../')
from utils.manipulateFiles import writeJson


class NERIndicios:
    def __init__(self, dataset: dict, modelo: str, modelName: str, tokenizador: str, batch_size: int, learning_rate: float, epochs: int,
                dir_save_models: str, dir_save_metrics: str, patience: int) -> None:
        self.modelName = modelName
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dir_save_models = dir_save_models
        self.dir_save_metrics = dir_save_metrics
        self.patience = patience
        
        print("Carregando dataset")
        self.dataset = load_from_disk(dataset)

        self.label_list = self.dataset['train'].features['ner_tags'].feature.names
        id2label = {i: label for i, label in enumerate(self.label_list)}
        label2id = {label: i for i, label in enumerate(self.label_list)}

        print("Carregando modelo e tokenizador")
        self.modelo = AutoModelForTokenClassification.from_pretrained(modelo, num_labels=len(self.label_list), id2label=id2label, label2id=label2id)
        self.tokenizador = AutoTokenizer.from_pretrained(tokenizador)

        print("Tokenizando e alinhando labels")
        self.dataset = self.dataset.map(self.tokenize_and_align_labels, batched=True)

        data_collator = DataCollatorForTokenClassification(self.tokenizador)
        self.metric = evaluate.load("seqeval")


        args = TrainingArguments(
            output_dir=self.dir_save_models,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            save_total_limit=1,
            load_best_model_at_end=True,
            seed = 0,
            fp16=True,
            metric_for_best_model = 'eval_f1',
            greater_is_better = True,
            gradient_checkpointing = True,
            do_train = True,
            do_eval = True,
            evaluation_strategy = 'epoch',
            logging_strategy = 'epoch',
            save_strategy = 'epoch',
        )

        trainer = Trainer(
            self.modelo,
            args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizador,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)]
        )
        
        print("Treinando o modelo...")
        trainer.train()

        predictions, labels, _ = trainer.predict(self.dataset["test"])
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        print("Salvando as métricas...")
        writeJson(os.path.join(dir_save_metrics, 'metrics-{}.json'.format(modelName)), eval(str(self.metric.compute(predictions=true_predictions, references=true_labels))))

        print("Salvando o modelo...")
        trainer.save_model(os.path.join(dir_save_models, modelName))
        self.modelo = AutoModelForTokenClassification.from_pretrained(os.path.join(dir_save_models, modelName), id2label=id2label, label2id=label2id, num_labels = len(self.label_list))
        shutil.rmtree(os.path.join(dir_save_models, modelName))
        self.modelo.save_pretrained(os.path.join(dir_save_models, modelName))


    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizador(examples["tokens"], truncation=True, is_split_into_words=True, max_length=512, padding="max_length")
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
            "accuracy": results["overall_accuracy"]
        }