from crossValidAvg import ClassificaIndicios
import sys
import numpy as np
import os
from tqdm.auto import tqdm
sys.path.insert(0, '../../')
from utils.manipulateFiles import openJson, writeJson


def main() -> None:
    #metrics_models = []
    params = openJson('configIndicios.json')
    for num_experimento in tqdm(range(1,2), desc='Executando experimentos', colour='red'):
        for setModel in params['modelos']:
            print("Classificando indicios com o modelo: " + setModel['model_name'])
            classificador = ClassificaIndicios(batch_size=params['batch_size'], epochs=params['epochs'],
                            patience=params['patience'], max_chunks=setModel['max_chunks'], method=setModel['method'],model_name=setModel['model_name'], dir_save_metrics=params['dir_save_metrics'],
                            dir_save_models=params['dir_save_models'], learning_rate=params['learning_rate'], gradient_accumulation_steps=params['gradient_accumulation_steps'],
                            modelo=setModel['modelo'], tokenizer=setModel['tokenizador'], dataset=params['dataset'], coluna=params['coluna'], frozen=params['frozen'])
        
            folds=np.load(params['fileFolds'], allow_pickle=True)
            print("Treinando modelo...")
            metricas = classificador.modelTraining(folds)
            print("Salvando resultados do modelo...")
            writeJson(os.path.join(params['dir_save_metrics'], '{}-{}-{}.json'.format(params['filenameMetrics'], num_experimento, setModel['model_name'])), metricas)
            #metrics_models.append(metricas)
        #print("Salvando resultados de todos os modelos...")
        #writeJson(os.path.join(params['dir_save_metrics'], '{}_models.json'.format(params['filenameMetrics'])), metrics_models)
    

if __name__ == "__main__":
    main()