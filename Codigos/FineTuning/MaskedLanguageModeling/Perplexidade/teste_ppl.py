from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from tqdm import tqdm
from numpy import mean
import pandas as pd
import numpy as np
import json

dados = pd.read_csv('/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/MaskedLanguageModeling/LipSet/LipSet_chunk.csv')
device = torch.device('cuda')
list_models = [
        {
            "model_name": "HelBERT-uncased-fs-6",
            "modelo": "/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Modelos/PreTreinamento/HelBERT-uncased-fs/checkpoint-epoca-6",
            "tokenizador": "/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Modelos/PreTreinamento/HelBERT-uncased-fs/checkpoint-epoca-6"
        },
        {
            "model_name": "HelBERT-uncased-fs-16",
            "modelo": "/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Modelos/PreTreinamento/HelBERT-uncased-fs/checkpoint-epoca-16",
            "tokenizador": "/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Modelos/PreTreinamento/HelBERT-uncased-fs/checkpoint-epoca-16"
        },
        {
            "model_name": "HelBERT-base-uncased-scratch-30",
            "modelo": "/var/projetos/Jupyterhubstorage/victor.silva/brazilian-legal-text-bert/Modelos/epocas_HelBERT-base-uncased-scratch/checkpoint-epoca-30",
            "tokenizador": "/var/projetos/Jupyterhubstorage/victor.silva/brazilian-legal-text-bert/Modelos/epocas_HelBERT-base-uncased-scratch/checkpoint-epoca-30"
        },
        {
            "model_name": "LegalBERT-pt-sc",
            "modelo": "raquelsilveira/legalbertpt_sc",
            "tokenizador": "raquelsilveira/legalbertpt_sc"
        },
        {
            "model_name": "jurisBERT",
            "modelo": "alfaneo/jurisbert-base-portuguese-uncased",
            "tokenizador": "alfaneo/jurisbert-base-portuguese-uncased"
        },
        {
            "model_name": "BERTimbau",
            "modelo": "neuralmind/bert-base-portuguese-cased",
            "tokenizador": "neuralmind/bert-base-portuguese-cased"
        },
        {
            "model_name": "mBERT",
            "modelo": "bert-base-multilingual-cased",
            "tokenizador": "bert-base-multilingual-cased"
        }
    ]

def score(model, tokenizer, sentence):
  tensor_input = tokenizer.encode(sentence, return_tensors='pt', truncation=True, max_length=96)
  repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
  mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
  masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
  labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
  with torch.no_grad():
    output = model(input_ids = masked_input.to(device), labels=labels.to(device))
  result = np.exp(output.loss.item())
  torch.cuda.empty_cache()
  del tensor_input, repeat_input, mask, masked_input, labels, output
  return result

results = []
for model_name in list_models:
    print(f'Importando o modelo {model_name["model_name"]}')
    model = AutoModelForMaskedLM.from_pretrained(model_name['modelo']).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name['tokenizador'])
    model.resize_token_embeddings(len(tokenizer))
    result_model = {}
    perplexities = []
    for indice in tqdm(dados.index, desc='Calculando perplexidades para o modelo {}'.format(model_name['model_name']), colour='yellow'):
        sentence = dados['text'][indice]
        ppl = score(model, tokenizer, sentence)
        perplexities.append(ppl)
    result_model['model_name'] = model_name['model_name']
    result_model['perplexity'] = mean(perplexities)
    results.append(result_model)
    print(result_model)
json.dump(results, open('/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Metricas/MaskedLanguageModeling/Perplexidade/perplexidades-lipset.json', 'w'), indent=4, ensure_ascii=False)
