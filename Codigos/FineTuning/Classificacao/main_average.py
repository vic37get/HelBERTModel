from train_average import TrainModel
import json
import os
import logging


def main() -> None:
    
    params = json.load(open('config.json'))
    for setModel in params['modelos']:

        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info(f"Treinando um classificador com o modelo: {setModel['model_name']}")
        
        classificador = TrainModel(
            dataset=params['dataset'],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            patience=params['patience'],
            dir_save_models=params['dir_save_models'],
            dir_save_metrics=params['dir_save_metrics'],
            gradient_accumulation_steps=params['gradient_accumulation_steps'],
            method=params['method'], 
            max_chunks=params['max_chunks'],
            max_length=params['max_length'],
            coluna=params['coluna'], 
            modelo=setModel['modelo'],
            model_name=setModel['model_name'], 
            metrics_name=params['metrics_name'],
            device=params['device'], 
            cross_validation=params['cross_validation'], 
            file_folds=params['file_folds'],
            tipo_estrategia=params['tipo_estrategia'],
            tipo_classificacao=params['tipo_classificacao'],
            funcao_perda_ponderada=params['funcao_perda_ponderada'],
            huggingface_dataset=params['huggingface_dataset'],
            token=params['token'],
        )
    
        metricas = classificador.train()
        logger.info(f"Salvando resultados do modelo: {setModel['model_name']}")
        json.dump(metricas, open(os.path.join(params['dir_save_metrics'], '{}-{}.json'.format(params['metrics_name'], setModel['model_name'])), 'w'), indent=4, ensure_ascii=False)
        logger.removeHandler(logger.handlers[0])
    

if __name__ == "__main__":
    main()