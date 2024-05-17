import json
import pickle
import os
import glob
import shutil

def openFile(filename: str) -> list:
    """
    Função que abre um arquivo e retorna uma lista com os dados do arquivo.
    Parametros:
    ----------
    filename: str
        Nome do arquivo.
    list
    """
    with open(filename, "r", encoding="utf8") as f:
        dados = [line.strip() for line in f.readlines()]
    return dados


def openJson(filename: str) -> dict:
    """
    Função que abre um arquivo json e retorna um dicionário com os dados do arquivo.
    Parametros:
    ----------
    filename: str
        Nome do arquivo json.
    dict
    """
    
    with open(filename, "r", encoding="utf8") as f:
        dados = json.load(f)
    return dados

def writeJson(filename: str, data: dict) -> None:
    """
    Função que escreve um arquivo json com os dados passados.
    Parametros:
    ----------
    filename: str
        Nome do arquivo json.
    data: dict
        Dados a serem escritos no arquivo.
    None
    """
    
    with open(filename, "w", encoding="utf8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def writePickle(filename: str, data: dict) -> None:
    """
    Função que escreve um arquivo pickle com os dados passados.
    Parâmetros:
    ----------
    filename: str
        Nome do arquivo pickle.
    data: dict
        Dados a serem escritos no arquivo.
    None
    """
    
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def loadPickle(filename: str) -> dict:
    """
    Função que abre um arquivo pickle e retorna um dicionário com os dados do arquivo.
    Parametros:
    ----------
    filename: str
        Nome do arquivo pickle.
    dict
    """
    
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def setModel(model_dir: str, model_name: str, filename: str) -> None:
    """
    Função para setar o modelo e o nome do modelo no arquivo json.
    """
    dados = openJson(filename)
    dados['model'] = model_dir
    dados['model_name'] = model_name
    writeJson(filename, dados)

def setTokenizer(tokenizer_dir: str, filename: str) -> None:
    """
    Função para setar o tokenizer no arquivo json.
    """
    dados = openJson(filename)
    dados['tokenizer_name'] = tokenizer_dir
    writeJson(filename, dados)

def appendJson(filename: str, data: dict) -> None:
    """
    Função para adicionar dados ao arquivo json.
    """
    try:
        with open(filename, "r", encoding="utf8") as f:
            dicionario = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        dicionario = []
        pass
    dicionario.append(data)

    with open(filename, "w", encoding="utf8") as f:
        json.dump(dicionario, f, indent=4)
    print("Dados adicionados ao arquivo json.")

def saveMetricsModel(fileconfigs: str, filemetrics: str, metrics: dict) -> None:
    """
    Função para salvar as métricas do modelo em um arquivo json.
    Parâmetros:
    ----------
    fileconfigs: str
        Nome do arquivo json com as configurações do modelo.
    filemetrics: str
        Nome do arquivo json com as métricas do modelo.
    metrics: dict
        Dicionário com as métricas do modelo.
    Retorno:
        None
    """
    try:
        configs = openJson(fileconfigs)
        configs['metrics'] = metrics
        appendJson(filemetrics, configs)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        raise Exception("Arquivo de configurações não encontrado.")

def writeMetrics(filemetrics: str, metrics: dict) -> None:
    """
    Função para salvar as métricas do modelo de pré-treinamento.
    """
    if not os.path.exists(filemetrics):
        with open(filemetrics, "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=4)
    else:
        dados = openJson(filemetrics)
        if isinstance(dados, list):
            dados.append(metrics)
            writeJson(filemetrics, dados)
        else:
            dados = [dados]
            dados.append(metrics)
            writeJson(filemetrics, dados)

def removeLastModels(dirModels: str) -> None:
    """
    Remove os N-2 ultimos modelos salvos.
    Parâmetros:
    ----------
    dirModels: str
        Diretório onde os modelos estão salvos.
    Retorno:
    -------
    None
    """
    files = [file for file in glob.glob(os.path.join(dirModels, "*")) if (os.path.basename(file) != "metricas" and os.path.basename(file) != 'SaveFiles') and os.path.isdir(file)]
    if len(files) > 2:
        files.sort(key=os.path.getmtime)
        for index in range(len(files)-2):
            shutil.rmtree(files[index])
        print("Os modelos antigos foram removidos.")

def makeRecovery(numeroTensor: str, numeroBase: str, epoca: str) -> None:
    """
    Cria um json com as informações do treinamento.
    """
    recovery = {
        "numeroTensor": numeroTensor,
        "numeroBase": numeroBase,
        "epoca": epoca
    }
    writeJson("recovery.json", recovery)

def recoveryTrainning(epocas: int, num_masks: int, num_tensor: int) -> dict:
    """
    Função que recupera o treinamento de um modelo.
    """
    changed = False
    recovery = openJson("recovery.json")
    numeroTensorRecovery = recovery.get('numeroTensor')
    numeroBaseRecovery = recovery.get('numeroBase')
    epocaRecovery = recovery.get('epoca')
    if epocaRecovery == (epocas -1):
        epocaRecovery = 0
        changed = True
        if numeroBaseRecovery < (num_masks -1):
            numeroBaseRecovery += 1
        else:
            numeroBaseRecovery = 0
            if numeroTensorRecovery < (num_tensor -1):
                numeroTensorRecovery += 1
    if not changed:
        epocaRecovery += 1
    return numeroTensorRecovery, numeroBaseRecovery, epocaRecovery


def makeRecoveryFull(numeroTensor: str, epoca: str, modelName: str) -> None:
    """
    Cria um json com as informações do treinamento.
    """
    recovery = {
        "numeroTensor": numeroTensor,
        "epoca": epoca,
        "modelName": modelName
    }
    writeJson("recovery.json", recovery)


def recoveryTrainingFull(epocas: int, num_tensors: int) -> dict:
    """
    Função que recupera o treinamento de um modelo.
    """
    changed = False
    recovery = openJson("recovery.json")
    numeroTensorRecovery = recovery.get('numeroTensor')
    epocaRecovery = recovery.get('epoca')
    modelRecovery = recovery.get('modelName')
    if numeroTensorRecovery == (num_tensors -1):
        numeroTensorRecovery = 0
        changed = True
        if epocaRecovery < (epocas -1):
            epocaRecovery += 1
    if not changed:
        numeroTensorRecovery += 1
    return numeroTensorRecovery, epocaRecovery, modelRecovery