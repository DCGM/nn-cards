# file model.py
# author Pavel Ševčík

import functools
from typing import Dict, List

import torch

from .nets import net_factory
from .heads import Head, HeadFactory

def model_factory(config):
    # build backbone
    backbone_config = config["backbone"]
    backbone = net_factory(backbone_config)

    # get backbone output dim
    head_input_dim = backbone.get_output_dim()

    # build heads
    head_config = config["heads"]
    head_factory = HeadFactory(head_input_dim)
    heads = [head_factory(cfg) for cfg in head_config]
    
    return MultiHeadModel(backbone, heads)

class MultiHeadModel(torch.nn.Module):
    def __init__(self, backbone, heads: List[Head]):
        super().__init__()
        self.backbone = backbone
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, batch):
        batch = self.backbone(batch)
        return [(head, head(batch)) for head in self.heads]

    def compute_loss(self, outputs) -> Dict[str, torch.Tensor]:
        losses = (head.compute_loss(output) for head, output in outputs)
        return functools.reduce(lambda x, y: {**x, **y}, losses, {})
    
    def do_backward_pass(self, losses: Dict[str, torch.Tensor]):
        # implementation of the simplest balancing, i.e. summation
        total_loss = sum(losses.values())
        total_loss.backward()
    
    def evaluate(self, dataloader_val) -> Dict[str, float]:
        with torch.no_grad():
            evaluation = (head.evaluate((self.backbone(batch.to(next(self.parameters()).device)) for batch in dataloader_val))
                          for head in self.heads)
            return functools.reduce(lambda x, y: {**x, **y}, evaluation, {})
