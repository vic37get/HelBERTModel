from trainNERIndicios import NERIndicios
import sys
sys.path.insert(0, '../../')
from utils.manipulateFiles import openJson
import json
import os


def main() -> None:
    params = openJson('configNERIndicios.json')
    
    for setModel in params['modelos']:
        print("Classificando entidades nomeadas com o modelo: " + setModel['model_name'])
        NERIndicios(batch_size=params['batch_size'], epochs=params['epochs'],
                        patience=params['patience'], modelName=setModel['model_name'], dir_save_metrics=params['dir_save_metrics'],
                        dir_save_models=params['dir_save_models'], learning_rate=params['learning_rate'], modelo=setModel['modelo'],
                        tokenizador=setModel['tokenizador'], dataset=params['dataset'])

    listMetrics = []    
    for file in os.listdir(params['dir_save_metrics']):
        if file.endswith(".json"):
            with open(os.path.join(params['dir_save_metrics'], file), 'r') as f:
                data = json.load(f)
                data['model_name'] = file.split('.')[0]
                listMetrics.append(data)

    with open(os.path.join(params['dir_save_metrics'], 'metrics.json'), 'w') as f:
        json.dump(listMetrics, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()