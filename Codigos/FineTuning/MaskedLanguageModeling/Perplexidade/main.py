from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from tqdm import tqdm
from numpy import mean
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import os
import re
import sys
sys.path.insert(0, '../../../')
from utils.manipulateFiles import writeJson, openJson 


class CustomDataset(Dataset):
    def __init__(self, masked_inputs, labels):
        self.masked_inputs = masked_inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.masked_inputs)
    
    def __getitem__(self, idx):
        x = self.masked_inputs[idx]
        y = self.labels[idx]
        return x, y
    
def filtrarNumeros(masked_input, labels, tokenizer):
    numeros = re.compile(r'(\b([0-9]+)\b)')
    numeros_romanos = re.compile(r"(\s|\.|\,|\;|\:|^)(?=[XVIΙ])(XC|XL|L?X{0,3})([IΙ]X|[IΙ]V|V?[IΙ]{0,3})(\s|\.|\,|\;|\:|$)")
    masked_finais, labels_finais = [], []
    for masked_input_i, labels_i in zip(masked_input, labels):
        token_masked = [i for i in labels_i if i != -100]
        token = tokenizer.convert_ids_to_tokens(token_masked)
        if re.search(numeros, token[0]) == None and re.search(numeros_romanos, token[0]) == None and token[0] != '[UNK]':
            masked_finais.append(masked_input_i)
            labels_finais.append(labels_i)
    masked_input = torch.stack(masked_finais, dim=0)
    labels = torch.stack(labels_finais, dim=0)
    return masked_input, labels


def score(model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, sentence: str, batch_size: int, device='cuda'):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt', max_length=256, truncation=True)
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    masked_input, labels = filtrarNumeros(masked_input, labels, tokenizer)    
    outputs = []
    dataset = CustomDataset(masked_input, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for masked_input_i, labels_i in dataloader:
        with torch.no_grad():
            output = model(input_ids = masked_input_i.to(device), labels=labels_i.to(device))
            outputs.append(output.loss.item())
    result = np.exp(mean(outputs))
    torch.cuda.empty_cache()
    del masked_input, labels, output
    return result


def main() -> None:
    params = openJson('configPerplexidade.json')
    dados = pd.read_csv(params['dataset'])
    device = torch.device('cpu')
    results = []
    
    for model_name in params['modelos']:
        print(f'Importando o modelo {model_name["model_name"]}')
        model = AutoModelForMaskedLM.from_pretrained(model_name['modelo']).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name['tokenizador'])
        model.resize_token_embeddings(len(tokenizer))
        result_model = {}
        perplexities = []
        inicio = time.time()
        for indice in tqdm(dados.index, desc='Calculando perplexidades para o modelo {}'.format(model_name['model_name']), colour='yellow'):
            sentence = dados['text'][indice]
            ppl = score(model, tokenizer, sentence, batch_size=params['batch_size'], device=device)
            perplexities.append(ppl)
        fim = time.time()
        result_model['model_name'] = model_name['model_name']
        result_model['perplexity'] = mean(perplexities)
        result_model['time'] = fim - inicio
        writeJson(os.path.join(params['dir_save_metrics'], '{}-ppl_sem_numeros.json'.format(model_name['model_name'])), result_model)
        results.append(result_model)
    writeJson(os.path.join(params['dir_save_metrics'], 'ppl_models_sem_numeros.json'), results)


if __name__ == '__main__':
    main()