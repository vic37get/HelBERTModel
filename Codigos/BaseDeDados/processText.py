from tqdm.auto import tqdm
import pandas as pd
from transformers import AutoTokenizer
import sys
sys.path.insert(0, '../')
from cleaner import Cleaner, Corretor, Remover


class ProcessText():
    def __init__(self, dataset: str, tokenizador: str, coluna: str, tam_min_sentencas: int, tam_max_sentencas: int, colunas: list,
                 cased: bool=True, accents: bool=False):
        print("Importando o dataset...")
        self.dataset = pd.read_csv(dataset)
        self.dataset = self.dataset[colunas]
        print("Importando o tokenizador...")
        self.tokenizador = AutoTokenizer.from_pretrained(tokenizador)
        self.coluna = coluna
        self.colunas = colunas
        self.cased = cased
        self.accents = accents
        self.tam_min_sentencas = tam_min_sentencas
        self.tam_max_sentencas = tam_max_sentencas
    
    def reduzColunas(self, dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Reduz o DataFrame a um DataFrame concatenado de coluna única, próprio para
        o pré-treinamento. 
        """
        secoes = []
        for coluna in tqdm(dataframe, desc='Reduzindo o DataFrame para coluna única', colour='green'):
            for linha in dataframe[coluna]:
                if isinstance(linha, str):
                    linha = linha.strip()
                    secoes.append(linha)
        dataframe = pd.DataFrame({column: secoes})
        dataframe = self.drops(dataframe, column)
        return dataframe
        
    def drops(self, dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove registros duplicados e nulos.
        """
        dataframe = dataframe.drop_duplicates(subset=[column], keep='first')
        dataframe = dataframe.dropna(subset=[column])
        dataframe = dataframe.reset_index(drop=True)
        return dataframe
    
    def executaLimpeza(self, dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Executa a limpeza do DataFrame usando as classes Cleaner e Corretor.
        """
        sentencesDrop = []
        for indice in tqdm(dataframe.index, desc="Executando a Limpeza", colour='green'):
            texto = dataframe[column][indice]
            if isinstance(texto, str):
                if Remover().removeSentences(texto) != None:
                    sentencesDrop.append(indice)
                    continue
                texto = Cleaner().clear(texto)
                texto = Corretor(self.cased, self.accents).corrige_termos(texto)
                dataframe.at[indice, column] = texto
            else:
                sentencesDrop.append(indice)
        dataframe = dataframe.drop(sentencesDrop)
        return dataframe
        
    
    def saveDataframe(self, saveDir: str, dataframe: pd.DataFrame) -> None:
        """
        Salva o DataFrame em um arquivo csv.
        """
        dataframe.to_csv(saveDir, index=False, encoding='utf8')