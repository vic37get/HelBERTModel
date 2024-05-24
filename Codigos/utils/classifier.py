import torch
import torch.nn as nn
from transformers import AutoModel

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, layer, model, tokenizer):
        super(Classifier, self).__init__()
        self.layer = layer
        self.bert = AutoModel.from_pretrained(model, output_hidden_states=True)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.trans = torch.nn.TransformerEncoderLayer(d_model=input_size, nhead=2)
        self.fc = torch.nn.Linear(input_size, 30)
        self.classifier = torch.nn.Linear(30, output_size)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        toks_embeds = torch.stack(output.hidden_states)
        toks_embeds = toks_embeds[self.layer]
        medium_embed = torch.mean(torch.mean(toks_embeds, dim=1), dim=0)      
        x = self.trans(medium_embed.unsqueeze(0))
        x = self.fc(x)
        x = self.classifier(x)
        return x