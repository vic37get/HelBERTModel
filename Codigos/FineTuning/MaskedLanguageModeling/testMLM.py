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
    
    
    def predicaoUnica(self, fill_sentence: list, labels: list) -> str:
        prediction = []
        if len(labels) > 1:
            for mascara in fill_sentence:
                prediction.append(unidecode(mascara[0]['token_str']).lower())
        else:
            prediction.append(unidecode(fill_sentence[0]['token_str']).lower())
        return prediction
    
    
    def predicaoMultipla(self, fill_sentence: list, labels: list) -> list:
        prediction = []
        if len(labels) > 1:
            for indice, mascara in enumerate(fill_sentence):
                for resultado in mascara:
                    if unidecode(resultado['token_str']).lower() == unidecode(labels[indice]).lower():
                        prediction.append(unidecode(resultado['token_str']).lower())
                        break
                else:
                    prediction.append(unidecode(mascara[0]['token_str']).lower())
        else:
            for resultado in fill_sentence:
                if unidecode(resultado['token_str']).lower() == unidecode(labels[0]).lower():
                    prediction.append(unidecode(resultado['token_str']).lower())
                    break
            else:
                prediction.append(unidecode(fill_sentence[0]['token_str']).lower())


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
        fill = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0)
        predicteds, listTrue = [], []
        for index in tqdm(val_loader.index, desc='Testando modelo de MLM {}'.format(modelName), colour='#00ff00'):
            sentence = val_loader['text'][index]
            sentence = " ".join([unidecode(token.lower()) if token != '[MASK]' else token for token in sentence.split()])  
            labels = eval(val_loader['labels'][index])
            labels = [unidecode(label.lower()) for label in labels]
            try:
                mlm_predictions = fill(sentence)
                prediction = self.predicaoUnica(mlm_predictions, labels)
                predicteds.extend(prediction)
                listTrue.extend(labels)
            except RuntimeError:
                print("Sentença muito comprida!")
                continue
        return self.calculateMetrics(listTrue, predicteds, modelName)


if __name__ == "__main__":
    pass