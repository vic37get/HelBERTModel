from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import torch.nn.functional as F

# Carregar os datasets de treino e teste
dataset = load_dataset("csv", data_files={
    "train": "/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/PreTreinamento/HelBERT-uncased-fs/treino_distilled.csv",
    "test": "/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/PreTreinamento/HelBERT-uncased-fs/teste_distilled.csv"
})

# Carregar os tokenizadores e modelos estudante e professor
student_tokenizer = BertTokenizer.from_pretrained('/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Modelos/PreTreinamento/distilHelBERT-base')
student_model = BertForMaskedLM.from_pretrained('/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Modelos/PreTreinamento/distilHelBERT-base').to('cuda')

teacher_tokenizer = BertTokenizer.from_pretrained('/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Modelos/PreTreinamento/HelBERT-uncased-fs/checkpoint-epoca-6')
teacher_model = BertForMaskedLM.from_pretrained('/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Modelos/PreTreinamento/HelBERT-uncased-fs/checkpoint-epoca-6').to('cuda')

# Tokenizar os datasets
def tokenize_function(examples):
    return student_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Definir o collator de dados para MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=student_tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Função de distilação para calcular a perda entre as previsões do estudante e do professor
def distillation_loss(student_outputs, teacher_outputs, temperature=2.0):
    student_logits = student_outputs.logits / temperature
    teacher_logits = teacher_outputs.logits / temperature
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return distillation_loss

# Definir o Trainer
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        student_outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
        loss = distillation_loss(student_outputs, teacher_outputs)
        return (loss, student_outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy='epoch',
    fp16=True,
    save_total_limit=1,
    logging_dir='./logs',
)

trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

# Treinar o modelo
trainer.train()
trainer.save_model('/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Modelos/PreTreinamento/distilFineTuned')