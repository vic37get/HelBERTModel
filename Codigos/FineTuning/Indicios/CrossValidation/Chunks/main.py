from crossValidAvgChunks import FineTunningBertAvgChunks
import sys
from tqdm.auto import tqdm
sys.path.insert(0, '../../../../')
from utils.manipulateFiles import openJson, writeJson
import numpy as np
import os


def main() -> None:
    metrics_models = []
    params = openJson('configIndiciosChunk.json')
    
    for setModel in tqdm(params['modelos'], desc="Realizando Cross Validation dos modelos..", colour='red'):
        folds = np.load(params['fileFolds'], allow_pickle=True)
        print("Classificando indicios com o modelo: " + setModel['model_name'])
        classificador = FineTunningBertAvgChunks(batch_size=params['batch_size'], epochs=params['epochs'],
                        patience=params['patience'], model_name=setModel['model_name'], dir_save_metrics=params['dir_save_metrics'],
                        dir_save_models=params['dir_save_models'], learning_rate=params['learning_rate'], modelo=setModel['modelo'],
                        max_chunks=setModel['max_chunks'],tokenizer=setModel['tokenizador'], datasetDir=params['datasetDir'],
                        gradient_accumulation_steps=params['gradient_accumulation_steps'], folds=folds)
        metricas = classificador.train()
        writeJson(os.path.join(params['dir_save_metrics'], setModel['model_name'] + '_metrics.json'), metricas)
        metrics_models.append(metricas)
    writeJson(os.path.join(params['dir_save_metrics'], 'metricsModels.json'), metrics_models)
          

if __name__ == "__main__":
    main()