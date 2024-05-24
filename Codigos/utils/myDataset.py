import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel


class MyDataset(Dataset):
    def __init__(self, dataFrame: pd.DataFrame, labels: list, column: str, tokenizer: BertTokenizer,
                  device: torch.device, modelo: BertModel) -> list:
        self.X = dataFrame[column].tolist()
        self.Y = dataFrame[labels].values.tolist()
        self.tokenizer = tokenizer
        self.device = device
        self.modelo = modelo

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        sample=self.X[index]
        sample=self.get_text_split(sample, self.tokenizer)
        tokens = self.tokenize(sample, self.tokenizer)
        tokens = { k: v.to(self.device) for k, v in tokens.items() }
        with torch.no_grad():
            output=self.modelo(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask'])
        #output=self.modelo(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask'])
        toks_embeds = torch.stack(output.hidden_states)
        try:
            toks_embeds = toks_embeds[12]
        except:
            toks_embeds = toks_embeds[6]
        embed_final=torch.mean(torch.mean(toks_embeds, dim=1), dim=0)
        return embed_final, self.Y[index]
    
    def get_text_split(self, text: str, tokenizer: BertTokenizer, length: int = 200, overlap: int = 0, max_chunks: int = 200) -> list:
        """
        Função que divide o texto em pedaços de tamanho length com overlap de tamanho overlap.
        Parâmetros:
            text: texto a ser dividido
            length: tamanho de cada pedaço
            overlap: tamanho da sobreposição entre os pedaços
            max_chunks: número máximo de pedaços
        Retorno:
            l_total: lista com os pedaços do texto
        """
        l_total = []
        l_parcial = []
        n_words = len(text.split()) 
        #n_words = len(tokenizer.tokenize(text))
        n = n_words//(length-overlap)+1
        if n_words % (length-overlap) == 0:
            n = n-1
        if n ==0:
            n = 1
        n = min(n, max_chunks)
        for w in range(n):
            if w == 0:
                l_parcial = text.split()[:length]
            else:
                l_parcial = text.split()[w*(length-overlap):w*(length-overlap) + length]
            l = " ".join(l_parcial)
            if w==n-1:
                if len(l_parcial) < 0.75*length and n!=1:
                    continue
            l_total.append(l)
        return l_total
    
    def tokenize(self, text: str, tokenizer: BertTokenizer) -> dict:
        """
        Função que tokeniza o texto.
        Parâmetros:
            text: texto a ser tokenizado
            tokenizer: tokenizer
        Retorno:
            tokens: dicionário com os tokens
        """
        text = list(text)
        tokens = tokenizer(
            text, 
            return_attention_mask=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}