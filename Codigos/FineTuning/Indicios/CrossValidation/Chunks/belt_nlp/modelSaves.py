import os
import shutil
from transformers import BertModel
import torch


class SaveBestMetrics:
    def __init__(self,best_f1=float('-inf')):
        self.best_f1 = best_f1
        self.best_metrics = None

    def __call__(self, current_f1: float, current_metrics: dict, train_loss: float):
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_metrics=current_metrics
            self.best_metrics['train_loss']=train_loss
        return self.best_metrics

class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss: float, model: BertModel,
                  dir_save_models: str, nameModel: str):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            model_save = os.path.join(dir_save_models, 'best_model-{}.pth'.format(nameModel))
            if os.path.exists(model_save):
                os.remove(model_save)
            torch.save(model,model_save)