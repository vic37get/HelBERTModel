from transformers import pipeline
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import warnings
from unidecode import unidecode
warnings.filterwarnings("ignore")


class MLMPrediction:
    def __init__(self) -> None:
        pass


    def calculateMetrics(self, true: list, predictions: list, modelName: str) -> dict:
        """
        Função que calcula as métricas do modelo.
        Parâmetros:
        ----------
        self: TrainMLM
            Objeto da classe TrainMLM
        true: list
            Lista com as palavras verdadeiras
        predictions: list
            Lista com as palavras preditas
        totalLoss: list
            Lista com as losses
        modelName: str
            Nome do modelo
        """
        precision = precision_score(true, predictions, average='weighted', zero_division=1)
        recall = recall_score(true, predictions,average='weighted', zero_division=1)
        f1 = f1_score(true, predictions, average='weighted', zero_division=1)
        accuracy = accuracy_score(true, predictions)
        hl = hamming_loss(true, predictions)  
        return {"modelName": modelName, 'precisao': precision, 'recall': recall, 'f1': f1, 'acuracia': accuracy, 'hloss': hl}


    def testMLM(self, val_loader: pd.DataFrame, model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, modelName: str) -> dict:
        """
        Função que testa o modelo de MLM.
        Parâmetros:
        ----------
        self: TrainMLM
            Objeto da classe TrainMLM
        val_loader: pd.DataFrame
            Dataloader de validação
        model: AutoModelForMaskedLM
            Modelo de MLM
        tokenizer: AutoTokenizer
            Tokenizador do modelo de MLM
        modelName: str
            Nome do modelo
        """
        model.eval()
        fill = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0)
        predicteds, listTrue = [], []
        for index in tqdm(val_loader.index, desc='Testando modelo de MLM {}'.format(modelName), colour='#00ff00'):
            sentence = val_loader['sentenca'][index]
            sentence = " ".join([unidecode(token.lower()) if token != '[MASK]' else token for token in sentence.split()])  
            labels = val_loader['labels'][index]
            labels = eval(labels)
            labels = [unidecode(label.lower()) for label in labels]
            try:
                prediction = [unidecode(token[0]['token_str'].lower()) for token in fill(sentence)]
                predicteds.extend(prediction)
                listTrue.extend(labels)
            except:
                continue
        return self.calculateMetrics(listTrue, predicteds, modelName)


if __name__ == "__main__":
    pass