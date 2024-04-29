from processText import ProcessText
import sys
sys.path.insert(0, '../../')
from utils.manipulateFiles import openJson


def main() -> None:
    params = openJson('./configCleaner.json')
    limpeza = ProcessText(**params)

    dataframe = limpeza.dataset
    #dataframe = limpeza.reduzColunas(dataframe, limpeza.coluna)
    dataframe = limpeza.executaLimpeza(dataframe, limpeza.coluna)
    dataframe = limpeza.drops(dataframe, limpeza.coluna)
    limpeza.saveDataframe('../../../Datasets/PreTreinamento/dfCleanerCasedFinalRevisado.csv', dataframe)

    # newSentences, dataframe = limpeza.reducesSentences(dataframe, limpeza.coluna)
    # dfNewSentences, dfStatistics = limpeza.breakInSentences(newSentences, dataframe, limpeza.coluna)
    # limpeza.saveDataframe('../../../Datasets/dfNewSentences.csv', dfNewSentences)
    # limpeza.saveDataframe('../../../Datasets/dfStatistics.csv', dfStatistics)
    # limpeza.saveDataframe('../../../Datasets/dfRestante.csv', dataframe)
    # #print(statisticsData(dfStatistics, limpeza.coluna, limpeza.tokenizador))

    # dataframe = limpeza.joinSentencesSize(dataframe, dfNewSentences, limpeza.coluna)
    # dataframe = limpeza.removeSentencesSize(dataframe, limpeza.coluna, limpeza.tam_min_sentencas, limpeza.tam_max_sentencas)
    # print(statisticsData(dataframe, limpeza.coluna, limpeza.tokenizador))
    # limpeza.saveDataframe('../../../Datasets/pncp_contratos-analise.csv', dataframe)


if __name__ == '__main__':
    main()