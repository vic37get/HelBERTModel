import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.trans = torch.nn.TransformerEncoderLayer(d_model=768, nhead=2)
        self.fc = torch.nn.Linear(768, 30)
        self.classifier = torch.nn.Linear(30, output_size)
        
    def forward(self, x):
        x = self.trans(x.unsqueeze(0))
        x = self.fc(x)
        x = self.classifier(x)
        return x