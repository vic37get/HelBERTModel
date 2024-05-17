from trainObjetos import ClassificaTiposObjetos
import sys
sys.path.insert(0, '../../')
from utils.manipulateFiles import openJson, writeJson
from utils.myDataset import MyDataset
import numpy as np
import os


def main() -> None:
    metrics_models = []
    params = openJson('configObjetos.json')
    
    for setModel in params['modelos']:
        print("Classificando Objetos com o modelo: " + setModel['model_name'])
        classificador = ClassificaTiposObjetos(batch_size=params['batch_size'], epochs=params['epochs'],
                        patience=params['patience'], model_name=setModel['model_name'], dir_save_metrics=params['dir_save_metrics'],
                        dir_save_models=params['dir_save_models'], learning_rate=params['learning_rate'], modelo=setModel['modelo'],
                        tokenizer=setModel['tokenizador'], treino=params['treino'], teste=params['teste'], validacao=params['validacao'], coluna=params['coluna'])
    
        train_dataset = MyDataset(classificador.treino, 'label', classificador.coluna, classificador.tokenizer, classificador.device, classificador.modelo)
        test_dataset = MyDataset(classificador.teste, 'label', classificador.coluna, classificador.tokenizer, classificador.device, classificador.modelo)
        valid_dataset = MyDataset(classificador.validacao, 'label', classificador.coluna, classificador.tokenizer, classificador.device, classificador.modelo)

        print("Treinando modelo...")
        metricas = classificador.modelTraining(train_dataset, valid_dataset, test_dataset)

        print("Salvando resultados do modelo...")
        writeJson(os.path.join(params['dir_save_metrics'], '{}-{}.json'.format(params['filenameMetrics'], setModel['model_name'])), metricas)
        metrics_models.append(metricas)

    print("Salvando resultados de todos os modelos...")
    writeJson(os.path.join(params['dir_save_metrics'], '{}_models.json'.format(params['filenameMetrics'])), metrics_models)
    

if __name__ == "__main__":
    main()