from trainAvg import ClassificaIndicios
import os
import json


def main() -> None:
    params = json.load(open('configIndicios.json'))
    for setModel in params['modelos']:
        print("Classificando indicios com o modelo: " + setModel['model_name'])
        classificador = ClassificaIndicios(num_labels=params['num_labels'], batch_size=params['batch_size'], epochs=params['epochs'],
                        patience=params['patience'], max_chunks=setModel['max_chunks'], method=setModel['method'],model_name=setModel['model_name'], dir_save_metrics=params['dir_save_metrics'],
                        dir_save_models=params['dir_save_models'], learning_rate=params['learning_rate'], gradient_accumulation_steps=params['gradient_accumulation_steps'],
                        modelo=setModel['modelo'], tokenizer=setModel['tokenizador'], train=params['train'], test=params['test'], val=params['val'], coluna=params['coluna'], frozen=params['frozen'])
    
        print("Treinando modelo...")
        metricas = classificador.modelTraining()
        print("Salvando resultados do modelo...")
        json.dump(metricas, open(os.path.join(params['dir_save_metrics'], '{}-{}.json'.format(params['filenameMetrics'], setModel['model_name'])), 'w'))
    

if __name__ == "__main__":
    main()