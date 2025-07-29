from crossValidNer import CrossValidNer
import json
import numpy as np
import os
import sys
from datasets import load_from_disk
sys.path.insert(0, '../../../')
from utils.manipulateFiles import writeJson


def main() -> None:
    params = json.load(open('configNer.json', 'r'))
    metrics_models = []
    for infoModel in params['models']:
        nerModel = CrossValidNer(batch_size=params['batch_size'], epochs=params['epochs'], patience=params['patience'],
                                dir_save_models=params['dir_save_models'], learning_rate=params['learning_rate'])
    
        folds = np.load(params['fileFolds'], allow_pickle=True)
        print('Carregando dados...')
        dataset = load_from_disk(params['dataset'])
        metrics = nerModel.train(folds, dataset, infoModel['model_name'], infoModel['modelo'], infoModel['tokenizador'])
        print("Salvando resultados do modelo...")
        writeJson(os.path.join(params['dir_save_metrics'], '{}-{}.json'.format(params['name_metrics'], infoModel['model_name'])), metrics)
        metrics_models.append(metrics)

    # print("Salvando resultados de todos os modelos...")
    # writeJson(os.path.join(params['dir_save_metrics'], '{}_models.json'.format(params['name_metrics'])), metrics_models)

if __name__ == '__main__':
    main()