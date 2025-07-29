import torch
import torch.nn as nn
from transformers import AutoModel

class Classifier(nn.Module):
    def __init__(self, input_size: int, output_size: int, model: str, tokenizer: str, method = "4"):
        super(Classifier, self).__init__()
        self.method = method
        self.bert = AutoModel.from_pretrained(model, output_hidden_states=True)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(input_size, output_size)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        toks_embeds = torch.stack(output.hidden_states)
        if self.method=="1":
            #Sum 12 layers
            toks_embeds = torch.sum(toks_embeds[1:],dim=0)
        elif self.method=="2":
            #First layer only
            toks_embeds = toks_embeds[0]
        elif self.method=="3":
            #Sum last 4 layers
            toks_embeds = torch.sum(toks_embeds[-4:],dim=0)
        elif self.method=="4":
            #Last hidden layer
            toks_embeds = toks_embeds[-1]
        elif self.method=="5":
            #Sum second-to-last hidden layer
            toks_embeds = torch.sum(toks_embeds[1:13],dim=0)
        elif self.method=="6":
            #Concat last four hidden layers
            toks_embeds=toks_embeds.permute(1,0,2,3)
            toks_embeds=torch.cat((toks_embeds[:,-4],toks_embeds[:,-3],toks_embeds[:,-2],toks_embeds[:,-1]),dim=1)
            
        embed_final=torch.mean(torch.mean(toks_embeds,dim=1), dim=0)   
        x = self.dropout(embed_final)
        x = self.classifier(x)
        return x