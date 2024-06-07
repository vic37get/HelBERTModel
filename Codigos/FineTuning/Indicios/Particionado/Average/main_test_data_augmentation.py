from trainAvg import ClassificaIndicios
import sys
import os
from tqdm.auto import tqdm
sys.path.insert(0, '../../')
from utils.manipulateFiles import openJson, writeJson


def main() -> None:
    metrics_models = []
    params = openJson('configIndicios.json')
    lista_datasets = []
    # Base de dados weak com data augmentation.
    train_da = '/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/Indicios/DatasetsWeak/DataAugmentation/bid_notices_weak_sup_hab_clean_train.csv'
    test_da = '/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/Indicios/DatasetsWeak/DataAugmentation/bid_notices_weak_sup_hab_clean_test.csv'
    val_da = '/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/Indicios/DatasetsWeak/DataAugmentation/bid_notices_weak_sup_hab_clean_val.csv'
    nome_base = 'data_augmentation'
    lista_datasets.append([train_da, test_da, val_da, nome_base])
    # Base de dados weak sem data augmentation.
    train_weak = '/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/Indicios/DatasetsWeak/Stratified/bid_notices_weak_sup_hab_clean_train.csv'
    test_weak = '/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/Indicios/DatasetsWeak/Stratified/bid_notices_weak_sup_hab_clean_test.csv'
    val_weak = '/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/Indicios/DatasetsWeak/Stratified/bid_notices_weak_sup_hab_clean_val.csv'
    nome_base = 'weak'
    lista_datasets.append([train_weak, test_weak, val_weak, nome_base])
    
    for train, test, validacao, nome_base in tqdm(lista_datasets, desc='Executando experimentos', colour='red'):
        for setModel in params['modelos']:
            print("Classificando indicios com o modelo: " + setModel['model_name'])
            print("Base de dados: " + nome_base)
            classificador = ClassificaIndicios(batch_size=params['batch_size'], epochs=params['epochs'],
                            patience=params['patience'], max_chunks=setModel['max_chunks'], method=setModel['method'],model_name=setModel['model_name'], dir_save_metrics=params['dir_save_metrics'],
                            dir_save_models=params['dir_save_models'], learning_rate=params['learning_rate'], gradient_accumulation_steps=params['gradient_accumulation_steps'],
                            modelo=setModel['modelo'], tokenizer=setModel['tokenizador'], train=train, test=test, val=validacao, coluna=params['coluna'], frozen=params['frozen'])
        
            print("Treinando modelo...")
            metricas = classificador.modelTraining()
            print("Salvando resultados do modelo...")
            writeJson(os.path.join(params['dir_save_metrics'], 'metrics-{}-{}.json'.format(nome_base, setModel['model_name'])), metricas)
            metrics_models.append(metricas)
                

if __name__ == "__main__":
    main()