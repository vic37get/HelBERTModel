import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import sys
sys.path.insert(0, '../../')
from utils.manipulateFiles import openJson, writeJson


def calculateFertility(sentence: str, tokenizer: AutoTokenizer) -> int:
    """
    Calculate the fertility of a sentence.
    """
    qtd_tokens = len(tokenizer.tokenize(sentence))
    words = len(sentence.split())
    return qtd_tokens / words


def main() -> None:
    params = openJson('configFertilidade.json')
    dados = pd.read_csv(params['dataset'])
    list_results = []
    for infoModel in params['modelos']:
        tokenizer = AutoTokenizer.from_pretrained(infoModel['tokenizador'])
        list_fertility = []
        for indice in tqdm(dados.index, desc="Calculando fertilidade para o modelo {}".format(infoModel['model_name']), colour='blue'):
            list_fertility.append(calculateFertility(dados.loc[indice, 'text'], tokenizer))
        results_model = {'model': infoModel['model_name'], 'fertility': round(sum(list_fertility) / len(list_fertility), 2)}
        writeJson(params['dir_save_metrics'] + '{}-fertility.json'.format(infoModel['model_name']), results_model)
        list_results.append(results_model)
    writeJson(params['dir_save_metrics'] + 'modelsFertility.json', list_results)


if __name__ == "__main__":
    main()