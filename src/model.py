# author: Pavel Ševčík

import functools
from typing import List

import torch

from .heads import Head

def model_factory(backbone_config, head_config):
    raise NotImplementedError()

class MultiHeadModel(torch.nn.Module):
    def __init__(self, backbone, heads: List[Head]):
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(self, batch):
        x = self.backbone(batch)
        return {head.name: head(x) for head in self.heads}

    def compute_loss(self, batch) -> dict[str, torch.Tensor]:
        x = self.backbone(batch)
        losses = (head.compute_loss(x) for head in self.heads)
        return functools.reduce(lambda x, y: {**x, **y}, losses, {})
    
    def do_backward_pass(self, losses: dict[str, torch.Tensor]):
        # implementation of the simplest balancing, i.e. summation
        total_loss = sum(losses.values())
        total_loss.backward()
    
    def evaluate(self, dataloader_val):
        raise NotImplementedError()
