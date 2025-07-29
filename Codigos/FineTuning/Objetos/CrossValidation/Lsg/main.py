from train_lsg_objetos import TrainLsgObjetos
from tqdm.auto import tqdm
import numpy as np
import json
import os


def main() -> None:
    params = json.load(open('./configObjetosLsg.json'))
    for setModel in tqdm(params['modelos'], desc="Realizando Cross Validation dos modelos..", colour='red'):
        folds = np.load(params['fileFolds'], allow_pickle=True)
        print("Classificando com o modelo: " + setModel['model_name'])
        classificador = TrainLsgObjetos(batch_size=params['batch_size'], epochs=params['epochs'],
                    patience=params['patience'], model_name=setModel['model_name'], dir_save_metrics=params['dir_save_metrics'],
                    dir_save_models=params['dir_save_models'], learning_rate=params['learning_rate'], modelo=setModel['modelo'],
                    tokenizer=setModel['tokenizador'], dataset=params['dataset'],
                    gradient_accumulation_steps=params['gradient_accumulation_steps'])
        
        folds=np.load(params['fileFolds'], allow_pickle=True)
        metricas = classificador.train(folds)
        json.dump(metricas, open(os.path.join(params['dir_save_metrics'], f"{params['filenameMetrics']}-{setModel['model_name']}.json"), 'w'), indent=4, ensure_ascii=False)          


if __name__ == "__main__":
    main()