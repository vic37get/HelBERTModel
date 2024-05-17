from trainObjetos import ClassificaTipoObjetos
import sys
sys.path.insert(0, '../../../')
from utils.manipulateFiles import openJson, writeJson
import os


def main() -> None:
    metrics_models = []
    params = openJson('configObjetos.json')

    for infoModel in params['modelos']:  
        print("Classificando Tipos de Objetos com o modelo: {}".format(infoModel['model_name']))     
        bertObjetos = ClassificaTipoObjetos(batch_size=params['batch_size'], epochs=params['epochs'],
                        patience=params['patience'], model_name=infoModel['model_name'], dir_save_metrics=params['dir_save_metrics'],
                        dir_save_models=params['dir_save_models'], learning_rate=params['learning_rate'], modelo=infoModel['modelo'],
                        tokenizer=infoModel['tokenizador'], treino=params['treino'], validacao=params['validacao'],
                        teste=params['teste'], coluna=params['coluna'])
        metrics = bertObjetos.fit()

        print("Salvando resultados do modelo...")
        writeJson(os.path.join(params['dir_save_metrics'], '{}-{}.json'.format(params['filenameMetrics'], infoModel['model_name'])), metrics)
    

if __name__ == "__main__":
    main()