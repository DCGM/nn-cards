# file model.py
# author Pavel Ševčík

import functools
from typing import Any, Dict, List

import torch

from .nets import net_factory
from .heads import Head, HeadFactory

def model_factory(config):
    # build backbone
    backbone_config = config["backbone"]
    backbone = net_factory(backbone_config)

    # get backbone output dims
    feature_sizes = backbone.get_output_dims()

    # build heads
    head_config = config["heads"]
    head_factory = HeadFactory(*feature_sizes)
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

    def eval_reset(self):
        for head in self.heads:
            head.eval_reset()

    def eval_add(self, outputs):
        for head, output in outputs:
            head.eval_add(output)

    def eval_get(self) -> Dict[str, Any]:
        evals = [head.eval_get() for head in self.heads]
        return functools.reduce(lambda x, y: {**x, **y}, evals, {})
