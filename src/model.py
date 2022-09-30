# author: Pavel Ševčík

import logging
import functools
from typing import Dict, List

import torch

from .nets import net_factory
from .heads import Head, head_factory

def model_factory(config):
    backbone_config = config["backbone"]
    head_config = config["heads"]
    backbone = net_factory(backbone_config)
    heads = [head_factory(cfg) for cfg in head_config]
    return MultiHeadModel(backbone, heads)

class MultiHeadModel(torch.nn.Module):
    def __init__(self, backbone, heads: List[Head]):
        super().__init__()
        self.backbone = backbone
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, batch):
        x = self.backbone(batch)
        return {head.name: head(x) for head in self.heads}

    def compute_loss(self, batch) -> Dict[str, torch.Tensor]:
        x = self.backbone(batch)
        batch.x = x
        losses = (head.compute_loss(batch) for head in self.heads)
        return functools.reduce(lambda x, y: {**x, **y}, losses, {})
    
    def do_backward_pass(self, losses: Dict[str, torch.Tensor]):
        # implementation of the simplest balancing, i.e. summation
        total_loss = sum(losses.values())
        total_loss.backward()
    
    def evaluate(self, dataloader_val) -> Dict[str, float]:
        with torch.no_grad():
            evaluation = (head.evaluate((self.backbone(batch) for batch in dataloader_val))
                          for head in self.heads)
            return functools.reduce(lambda x, y: {**x, **y}, evaluation, {})
