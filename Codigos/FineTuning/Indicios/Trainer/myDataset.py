import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer


class MyDataset(Dataset):
    def __init__(self, dataFrame: pd.DataFrame, labels: list, column: str, tokenizer: BertTokenizer,
                  device: torch.device) -> list:
        self.X = dataFrame[column].tolist()
        self.Y = dataFrame[labels].values.tolist()
        self.tokenizer = tokenizer
        self.device = device
        self.tokens = self.tokenizer(self.X, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.tokens['input_ids'][index], self.tokens['attention_mask'][index], self.Y[index]