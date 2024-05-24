from crossValidIndicios import ClassificaIndicios
import sys
sys.path.insert(0, '../../')
from utils.manipulateFiles import openJson, writeJson
from utils.myDataset import MyDataset
import numpy as np
import os


def main() -> None:
    metrics_models = []
    params = openJson('configIndicios.json')
    
    for setModel in params['modelos']:
        print("Classificando indicios com o modelo: " + setModel['model_name'])
        classificador = ClassificaIndicios(batch_size=params['batch_size'], epochs=params['epochs'],
                        patience=params['patience'], model_name=setModel['model_name'], dir_save_metrics=params['dir_save_metrics'],
                        dir_save_models=params['dir_save_models'], learning_rate=params['learning_rate'], modelo=setModel['modelo'],
                        tokenizer=setModel['tokenizador'], dataset=params['dataset'], coluna=params['coluna'])
    
        labels = classificador.dataset.columns.values.tolist()[1:]
        train_dataset = MyDataset(classificador.dataset, labels, classificador.coluna, classificador.tokenizer, classificador.device, classificador.modelo)
        folds=np.load(params['fileFolds'], allow_pickle=True)

        print("Treinando modelo...")
        metricas = classificador.modelTraining(folds, train_dataset)

        print("Salvando resultados do modelo...")
        writeJson(os.path.join(params['dir_save_metrics'], '{}-{}.json'.format(params['filenameMetrics'], setModel['model_name'])), metricas)
        metrics_models.append(metricas)

    print("Salvando resultados de todos os modelos...")
    writeJson(os.path.join(params['dir_save_metrics'], '{}_models.json'.format(params['filenameMetrics'])), metrics_models)
    

if __name__ == "__main__":
    main()