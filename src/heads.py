# author: Pavel Ševčík

from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from .nets import net_factory

class Head(ABC):
    @abstractmethod
    def forward(self, batch):
        pass

    @abstractmethod
    def compute_loss(self, batch) -> Dict[str, torch.Tensor]:
        pass

def head_factory(head_config) -> List[Head]:
    """Returns a list of heads according to the configuration"""
    raise NotImplementedError()

class ClassificationHead(torch.nn.Module, Head):
    def __init__(self, name, net_config):
        super().__init__()
        self.name = name
        self.net = net_factory(net_config)
        self.criterion = torch.nn.CrossEntropyLoss()

    def compute_loss(self, batch) -> Dict[str, torch.Tensor]:
        x = self(batch)

        label = getattr(batch, self.name)
        mask = getattr(batch, f"{self.name}_mask", None)
        if mask:
            x = x[mask]
            label = label[mask]
        
        loss = self.criterion(x, label)
        return {self.name: loss}

class NodeClassificationHead(ClassificationHead):
    def forward(self, batch):
        x = self.net(batch.x)
        return x

class CosEdgeClassificationHead(ClassificationHead):
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.net(x)

        src, dst = edge_index
        score = torch.sum(x[src] * x[dst], dim=-1)

        return score
