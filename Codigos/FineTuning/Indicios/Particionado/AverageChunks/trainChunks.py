import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback, Trainer, EvalPrediction
import time
import numpy as np
from  distrib_balanced_loss import ResampleLoss
from collections.abc import Mapping
from belt_nlp.splitting import split_tokens_into_smaller_chunks, add_special_tokens_at_beginning_and_end, add_padding_tokens, stack_tokens_from_all_chunks


class FineTunningBertAvgChunks:
    def __init__(self, batch_size: int, epochs: int, patience: int, model_name: str, dir_save_metrics: str,
                dir_save_models: str, learning_rate: float, modelo: str, max_chunks: int, tokenizer: str,
                train: str, test: str, val: str) -> None:
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model_name = model_name
        self.dir_save_metrics = dir_save_metrics
        self.dir_save_models = dir_save_models
        self.learning_rate = learning_rate
        self.modelo = modelo
        self.max_chunks = max_chunks
        self.tokenizer = tokenizer
        self.dataset = train

        dataset = load_dataset("csv", data_files={"train": train, "test": test, "val": val})
        self.labels = list(dataset['train'].features.keys())[1:]
        self.id2label = {idx:label for idx, label in enumerate(self.labels)}
        self.label2id = {label:idx for idx, label in enumerate(self.labels)}

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.encoded_dataset = dataset.map(self.preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
        self.encoded_dataset.set_format("torch")
    
    def train(self):
        inicio = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(self.modelo, problem_type="multi_label_classification", num_labels=len(self.labels), id2label=self.id2label, label2id=self.label2id)
        train_dataset = self.encoded_dataset['train']
        val_dataset = self.encoded_dataset['val']

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
            metric_for_best_model="eval_loss",
            gradient_accumulation_steps=8,
        )

        train_df = pd.read_csv(self.dataset)
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
            callbacks = [EarlyStoppingCallback(early_stopping_patience=4)],
            chunk_size=510,
            max_chunks=self.max_chunks,
            stride=500,
            minimal_chunk_length=50,
            class_freq=class_freq,
            train_num=train_num
        )

        trainer.train()
        metrics = trainer.evaluate(eval_dataset=val_dataset)
        trainer.save_model('{}-{}'.format(self.dir_save_models, self.model_name))

        fim = time.time()
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
        encoding=self.tokenizer(text, add_special_tokens=False, truncation=False)
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
        self.chunk_size = kwargs.get('chunk_size')
        self.max_chunks = kwargs.get('max_chunks')
        self.stride = kwargs.get('stride')
        self.minimal_chunk_length = kwargs.get('minimal_chunk_length')
        self.class_freq = kwargs.get('class_freq')
        self.train_num = kwargs.get('train_num')

        self.loss_fct = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                         focal=dict(focal=True, alpha=0.5, gamma=2),
                         logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                         map_param=dict(alpha=2.5, beta=10.0, gamma=0.9),
                         class_freq=self.class_freq, train_num=self.train_num)

    def truncate_tensor(self,tensor, max_length):
        if tensor.size(1) <= max_length:
            return tensor
        else:
            return tensor[:, :max_length]
   
    def nested_detach(self,tensors):
        "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
        if isinstance(tensors, (list, tuple)):
            return type(tensors)(self.nested_detach(t) for t in tensors)
        elif isinstance(tensors, Mapping):
            return type(tensors)({k: self.nested_detach(t) for k, t in tensors.items()})
        return tensors.detach()

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        self.accelerator.backward(loss)
        return loss.detach()/self.args.gradient_accumulation_steps

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        if has_labels or loss_without_labels:
            labels = self.nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None
        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss,predictions = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(predictions, dict):
                    logits = tuple(v for k, v in predictions.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = predictions[1:]
        if prediction_loss_only:
           return (loss, None, None)
        logits = self.nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        return (loss, predictions, labels)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        num_classes=self.model.config.num_labels
        batch_size=self.args.per_device_train_batch_size
        device="cuda"
        trunc_size=int(510*(self.max_chunks/batch_size))
        input_ids_chunkeds=[]
        attention_mask_chunkeds=[]
        # split input_ids and attention_mask of samples in batch into chunks
        for i in range(inputs['input_ids'].shape[0]):
            input_ids_padded=inputs['input_ids'][i]
            indexes = torch.nonzero(input_ids_padded)
            input_ids_orig = input_ids_padded[indexes].squeeze().unsqueeze(dim=0).to("cpu")
            input_ids_orig = self.truncate_tensor(input_ids_orig, trunc_size)
            att_mask_padded=inputs['attention_mask'][i]
            indexes = torch.nonzero(att_mask_padded)
            att_maks_orig = att_mask_padded[indexes].squeeze().unsqueeze(dim=0).to("cpu")
            att_maks_orig = self.truncate_tensor(att_maks_orig, trunc_size)
            tensor_dict = {'input_ids': input_ids_orig, 'attention_mask': att_maks_orig}
            input_id_chunks, mask_chunks = split_tokens_into_smaller_chunks(tensor_dict, self.chunk_size, self.stride, self.minimal_chunk_length)
            add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
            add_padding_tokens(input_id_chunks, mask_chunks)
            input_ids, attention_mask = stack_tokens_from_all_chunks(input_id_chunks, mask_chunks)
            #print("Input_ids:",input_ids.shape, "Attention_mask:",attention_mask.shape)
            input_ids_chunkeds.append(input_ids)
            attention_mask_chunkeds.append(attention_mask)
        labels = inputs.get("labels")
        number_of_chunks = [len(x) for x in input_ids_chunkeds]
        # concatenate all input_ids into one batch
        input_ids_combined = []
        for x in input_ids_chunkeds:
            input_ids_combined.extend(x.tolist())
        # concatenate all attention masks into one batch
        attention_mask_combined = []
        for x in attention_mask_chunkeds:
            attention_mask_combined.extend(x.tolist())
        preds=torch.empty(0, num_classes).to(device)
        # Processa os documentos do lote em partes menores baseadas em número de chunks
        batch_chunks=batch_size
        for i in range(0, len(input_ids_combined), batch_chunks):
            input_ids_temp=input_ids_combined[i:i + batch_chunks]
            input_ids_combined_tensors = torch.stack([torch.tensor(x).to(device) for x in input_ids_temp])
            attention_mask_temp=attention_mask_combined[i:i + batch_chunks]
            attention_mask_combined_tensors = torch.stack([torch.tensor(x).to(device) for x in attention_mask_temp])
            # get model predictions for the combined batch            
            preds_chunks = model(input_ids_combined_tensors, attention_mask_combined_tensors)
            preds_chunks = preds_chunks["logits"]
            preds=torch.cat([preds,preds_chunks])
        # split result preds into chunks
        preds_split = preds.split(number_of_chunks,dim=0)     
        # pooling
        pooled_preds = torch.stack([torch.mean(x,dim=0) for x in preds_split])
        # compute loss
        loss = self.loss_fct(pooled_preds, labels)
        return (loss,pooled_preds) if return_outputs else loss