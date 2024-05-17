from testMLM import MLMPrediction
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
import os
import torch
sys.path.insert(0, '../../')
from utils.manipulateFiles import openJson, writeJson
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    device = torch.device("cuda")
    metrics = []
    data = openJson("configMLMPrediction.json")
    print("Importando dados...")
    dataset = pd.read_csv(data['dataset'])
    for setModel in data['modelos']:
        testMLM = MLMPrediction()
        modelo = AutoModelForMaskedLM.from_pretrained(setModel['modelo']).to(device)
        tokenizador = AutoTokenizer.from_pretrained(setModel['tokenizador'])
        metric = testMLM.testMLM(dataset, modelo, tokenizador, setModel['model_name'])
        metrics.append(metric)
        writeJson(os.path.join(data['dir_metrics'], "{}-metrics.json".format(setModel['model_name'])), metric)
    writeJson(os.path.join(data['dir_metrics'], "metricsMLM.json"), metrics)

    
if __name__ == "__main__":
    main()