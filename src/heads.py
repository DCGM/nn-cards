# author: Pavel Ševčík

import logging
from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from .nets import net_factory
from .dataset import DataBuild, NullDataBuild, AddOneHotAttr

class Head(torch.nn.Module, ABC):
    @abstractmethod
    def compute_loss(self, batch) -> Dict[str, torch.Tensor]:
        pass

    def get_data_build(self) -> DataBuild:
        return NullDataBuild()

def head_factory(head_config) -> List[Head]:
    """Returns a list of heads according to the configuration"""
    type = head_config["type"]
    del head_config["type"]
    if type == "node_cls":
        return NodeClassificationHead(**head_config)
    elif type == "cos_edge_cls":
        return CosEdgeClassificationHead(**head_config)
    else:
        msg = f"Unknown head type '{type}'."
        logging.error(msg)
        raise ValueError(msg)

class ClassificationHead( Head):
    def __init__(self, field, net_config, classes):
        super().__init__()
        self.field = field
        self.net = net_factory(net_config)
        self.classes = classes
        self.criterion = torch.nn.CrossEntropyLoss()

    def compute_loss(self, batch) -> Dict[str, torch.Tensor]:
        x = self(batch)

        label = getattr(batch, self.field)
        mask = getattr(batch, f"{self.field}_mask", None)
        if mask:
            x = x[mask]
            label = label[mask]
        
        loss = self.criterion(x, label)
        return {self.field: loss}
    
    def get_data_build(self) -> DataBuild:
        return AddOneHotAttr(self.field, self.field, self.classes)

class NodeClassificationHead(ClassificationHead):
    def forward(self, batch):
        x = self.net(batch)
        return x

class CosEdgeClassificationHead(ClassificationHead):
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.net(batch)

        src, dst = edge_index
        score = torch.sum(x[src] * x[dst], dim=-1)

        return score
