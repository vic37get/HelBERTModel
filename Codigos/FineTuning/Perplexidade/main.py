from calculaPerplexidade import PerplexityPipeline, score
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm.auto import tqdm
import torch
import pandas as pd
from evaluate import load
perplexity = load("perplexity", module_type="metric")
import os
import sys
sys.path.insert(0, '../../')
from utils.manipulateFiles import openJson, writeJson


def main() -> None:
    params = openJson('configPerplexidade.json')
    dados = pd.read_csv(params['dataset'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_perplexity = []
    
    for infoModel in params['modelos']:
        model = AutoModelForMaskedLM.from_pretrained(infoModel['modelo']).to(device)
        tokenizer = AutoTokenizer.from_pretrained(infoModel['tokenizador'])
        list_results = []
        for indice in tqdm(dados.index, desc="Calculando perplexidade para o modelo {}".format(infoModel['model_name']), colour='blue'):
            results = score(model=model, tokenizer=tokenizer, sentence=dados.loc[indice, 'text'], device=device)
            list_results.append(results)
        results_model = {'model': infoModel['model_name'], 'perplexity': round(sum(list_results) / len(list_results), 2)}
        writeJson(os.path.join(params['dir_save_metrics'], '{}-perplexity.json'.format(infoModel['model_name'])), results_model)
        metrics_perplexity.append(results_model)
        del model
        torch.cuda.empty_cache()
    writeJson(os.path.join(params['dir_save_metrics'], 'modelsPerplexity.json'), metrics_perplexity)

        # results = perplexity.compute(model_id=infoModel['modelo'],
        #                         add_start_token=False,
        #                         predictions=dados)
        # results_model = {'model': infoModel['model_name'], 'perplexity': round(results['mean_perplexity'], 2)}
        # writeJson(os.path.join(params['dir_save_metrics'], '{}-perplexity.json'.format(infoModel['model_name'])), results_model)
        # metrics_perplexity.append(results_model)
    #     list_perplexidades = []
    #     modelo = AutoModelForMaskedLM.from_pretrained(infoModel['modelo']).to(device)
    #     tokenizador = AutoTokenizer.from_pretrained(infoModel['tokenizador'])
    #     perplexidade = PerplexityPipeline(model=modelo, tokenizer=tokenizador)
    #     for sentenca in tqdm(dados, desc="Calculando perplexidade para o modelo {}".format(infoModel['model_name']),  colour='blue'):
    #         perplexidade_pipeline = perplexidade(sentenca)
    #         list_perplexidades.append(perplexidade_pipeline['ppl'])
    #     perplexidade_media = sum(list_perplexidades) / len(list_perplexidades)
    #     results_model = {'model': infoModel['model_name'], 'perplexity': perplexidade_media}
    #     writeJson(os.path.join(params['dir_save_metrics'], '{}-perplexity.json'.format(infoModel['model_name'])), results_model)
    #     metrics_perplexity.append(results_model)
    # writeJson(os.path.join(params['dir_save_metrics'], 'modelsPerplexity.json'), metrics_perplexity)

    
if __name__ == "__main__":
    main()