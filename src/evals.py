# file evals.py
# author Pavel Ševčík

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch

from .dataset import ClassListEncoder

class Eval(ABC):
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        pass

class ArgMaxClassificationEval(Eval):
    def __init__(self, encoder: ClassListEncoder):
        n_classes = len(encoder.classes)
        self.encoder = encoder
        self.confusion_matrix = torch.zeros((n_classes, n_classes))
    
    def get_results(self) -> Dict[str, Any]:
        return {
            "acc": self.avg_acc().item()
        }

    def reset(self):
        self.confusion_matrix.zero_()
    
    def add(self, pred, gt):
        pred_label = torch.argmax(pred, dim=1)
        gt_label = gt
        for i, j in zip(pred_label, gt_label):
            self.confusion_matrix[i, j] += 1
    
    def avg_acc(self):
        cm = self.confusion_matrix
        return cm.trace() / cm.sum()
