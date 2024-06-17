from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch
from tqdm import tqdm
from numpy import mean
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from unidecode import unidecode
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
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


def calculateMetrics(true: list, predictions: list) -> dict:
    precision = precision_score(true, predictions, average='weighted', zero_division=1)
    recall = recall_score(true, predictions,average='weighted', zero_division=1)
    f1 = f1_score(true, predictions, average='weighted', zero_division=1)
    accuracy = accuracy_score(true, predictions)
    hl = hamming_loss(true, predictions)  
    return {'precisao': precision, 'recall': recall, 'f1': f1, 'acuracia': accuracy, 'hloss': hl}

def mask_words(sentence):
    masked_sentences = []
    words = sentence.split()
    for i in range(len(words)):
        if words[i].isalpha():
            new_sentence = words.copy()
            new_sentence[i] = '[MASK]'
            masked_sentences.append([' '.join(new_sentence), words[i]])
    return masked_sentences

def predicaoUnica(fill_sentence: list) -> str:
    return unidecode(fill_sentence[0]['token_str']).lower()

def get_perplexity(model, tokenizer, sentence, label):
    fill = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0)
    mlm_predictions = fill(sentence)
    prediction = predicaoUnica(mlm_predictions)
    label = unidecode(label).lower()
    return prediction, label


def main() -> None:
    params = openJson('configPerplexidade.json')
    dados = pd.read_csv(params['dataset'])
    device = torch.device('cuda')
    results = []
    indices_skip = []
    # O primeiro modelo a ser calculado é o mBERT, pois ele é o modelo que possui a maior fertilidade.
    # Com isso, sentenças maiores que 512 tokens não serão computadas para nenhum modelo.
    for model_name in params['modelos']:
        print(f'Importando o modelo {model_name["model_name"]}')
        model = AutoModelForMaskedLM.from_pretrained(model_name['modelo']).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name['tokenizador'])
        model.resize_token_embeddings(len(tokenizer))
        result_model = {}
        inicio = time.time()
        list_true, list_preds = [], []
        # Se for uma sentença dentro dos indices para passar, não calcula.
        dados = dados.drop(indices_skip, axis=0).reset_index(drop=True)
        indices_skip = []
        for indice in tqdm(dados.index, desc='Calculando perplexidades para o modelo {}'.format(model_name['model_name']), colour='yellow'):
            # Se a sentença tokenizada tiver mais de 512 tokens, não calcula.
            try:
                sentence = mask_words(dados.loc[indice, 'text'])
                for masked_sentence, label in sentence:
                    pred, true = get_perplexity(model, tokenizer, masked_sentence, label)
                    list_true.append(true)
                    list_preds.append(pred)
            except:
                print(f'Sentença com {len(tokenizer.tokenize(dados.loc[indice, 'text']))} tokens. Pulando para a próxima sentença.')
                indices_skip.append(indice)
                continue
        result_model['metrics'] = calculateMetrics(list_true, list_preds)
        fim = time.time()
        result_model['model_name'] = model_name['model_name']
        result_model['time'] = fim - inicio
        writeJson(os.path.join(params['dir_save_metrics'], '{}-ppl_sem_numeros.json'.format(model_name['model_name'])), result_model)
        results.append(result_model)
    writeJson(os.path.join(params['dir_save_metrics'], 'ppl_models_sem_numeros.json'), results)


# def main() -> None:
#     params = openJson('configPerplexidade.json')
#     dados = pd.read_csv(params['dataset'])
#     device = torch.device('cpu')
#     results = []
    
#     for model_name in params['modelos']:
#         print(f'Importando o modelo {model_name["model_name"]}')
#         model = AutoModelForMaskedLM.from_pretrained(model_name['modelo']).to(device)
#         tokenizer = AutoTokenizer.from_pretrained(model_name['tokenizador'])
#         model.resize_token_embeddings(len(tokenizer))
#         result_model = {}
#         perplexities = []
#         inicio = time.time()
#         for indice in tqdm(dados.index, desc='Calculando perplexidades para o modelo {}'.format(model_name['model_name']), colour='yellow'):
#             sentence = dados['text'][indice]
#             ppl = score(model, tokenizer, sentence, batch_size=params['batch_size'], device=device)
#             perplexities.append(ppl)
#         fim = time.time()
#         result_model['model_name'] = model_name['model_name']
#         result_model['perplexity'] = mean(perplexities)
#         result_model['time'] = fim - inicio
#         writeJson(os.path.join(params['dir_save_metrics'], '{}-ppl_sem_numeros.json'.format(model_name['model_name'])), result_model)
#         results.append(result_model)
#     writeJson(os.path.join(params['dir_save_metrics'], 'ppl_models_sem_numeros.json'), results)


if __name__ == '__main__':
    main()