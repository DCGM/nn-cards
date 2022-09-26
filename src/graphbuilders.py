# file graphbuilders.py
# author Pavel Ševčík

import logging

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

def graph_builder_factory(config):
    return T.Compose([transform_factory(transform) for transform in config])

def transform_factory(config):
    type = config["type"]
    del config["type"]
    if type == "add_pos":
        return PosCenterRectTransform()
    elif type == "knn":
        return T.KNNGraph(**config)
    elif type == "label_pred_init":
        return AddLabelTargetTransform()
    else:
        msg = f"Unknown trasform '{type}'."
        logging.error(msg)
        raise ValueError(msg)

class PosCenterRectTransform(T.BaseTransform):
    """Adds .pos attribute to the Data object computed from the rectangle coordinates .x ([n_nodes, 4])"""
    def __call__(self, data: Data) -> Data:
        data.pos = torch.stack(((data.x[:,0] + data.x[:,1]) / 2, (data.x[:,2] + data.x[:,3]) / 2), dim=1)
        return data

class AddLabelTargetTransform(T.BaseTransform):
    """Custom transform that sets node classification target"""
    def __call__(self, data: Data) -> Data:
        data.label_pred = data.label
        return data
