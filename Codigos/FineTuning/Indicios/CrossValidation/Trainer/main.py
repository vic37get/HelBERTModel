from trainTrainer import FineTunningTrainer
import sys
from tqdm.auto import tqdm
sys.path.insert(0, '../../../../')
from utils.manipulateFiles import openJson, writeJson
import os
import numpy as np


def main() -> None:
    params = openJson('configTrainer.json')
    for setModel in tqdm(params['modelos'], desc="Realizando Cross Validation dos modelos..", colour='red'):
        print("Classificando indicios com o modelo: " + setModel['model_name'])
        classificador = FineTunningTrainer(batch_size=params['batch_size'], epochs=params['epochs'],
                        patience=params['patience'], model_name=setModel['model_name'], dir_save_metrics=params['dir_save_metrics'],
                        dir_save_models=params['dir_save_models'], learning_rate=params['learning_rate'], modelo=setModel['modelo'],
                        tokenizer=setModel['tokenizador'], dataset=params['dataset'])
        folds=np.load(params['fileFolds'], allow_pickle=True)
        metricas = classificador.train(folds)
        writeJson(os.path.join(params['dir_save_metrics'], setModel['model_name'] + '_metrics.json'), metricas)          

if __name__ == "__main__":
    main()