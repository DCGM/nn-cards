# file nets.py
# author Michal Hradiš, Pavel Ševčík

import logging
from abc import ABC, abstractmethod
from copy import copy
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class Backbone(ABC):
    @abstractmethod
    def get_output_dims(self) -> Tuple[int, int, int]:
        """When overridden should return dims of node, edge and graph features"""

def net_factory(config):
    net_type = config["type"].lower()
    del config["type"]
    net = None
    if net_type == "mlp":
        return MLP(**config)
    elif net_type == "gcn":
        return GCN(**config)
    elif net_type == "identity":
        return IdentityNet(**config)
    else:
        msg = f"Unknown network type '{net_type}'."
        logging.error(msg)
        raise ValueError(msg)

class SequentialBackbone(Backbone):
    def __init__(self, nets: List[Backbone]):
        if len(nets) < 1:
            msg = f"At least one nets is required to build a sequential backbone"
            logging.error(msg)
            raise ValueError(msg)

        self.nets = nets

    def get_output_dims(self) -> Tuple[int, int, int]:
        return self.nets[-1].get_output_dims()

class IdentityNet(torch.nn.Module, Backbone):
    def __init__(self, input_dim=None, output_dim=None):
        super().__init__()
        if input_dim != output_dim:
            msg = f"Input dim and output dim should be equal but they differ {input_dim}!={output_dim}."
            logging.error(msg)
            raise ValueError(msg)
        
        self.input_dim = input_dim

    def forward(self, batch):
        return copy(batch)
    
    def get_output_dims(self) -> Tuple[int, int, int]:
        return self.input_dim, 0, 0


class MLP(torch.nn.Module, Backbone):

    def __init__(self, target, input_dim, output_dim, edge=False, hidden_dim=128, depth=4):
        super().__init__()
        self.target = target
        self.mapping = {
            "node": "x",
            "edge": "edge_attr",
            "graph": "graph_attr"
        }
        layers = [ torch.nn.Linear(input_dim, hidden_dim)]
        for i in range(depth - 1):
            layers += [torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim)]
        layers += [torch.nn.ReLU(), torch.nn.Linear(hidden_dim, output_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, data):
        value = self._get_input_value(data)
        value = self.net(value)
        d_copy = copy(data)
        self._write_output_value(d_copy, value)
        return d_copy

    def _get_input_value(self, data: Data):
        return getattr(data, self.mapping[self.target])

    def _write_output_value(self, data: Data, value):
        setattr(data, self.mapping[self.target], value)


class GCN(torch.nn.Module, Backbone):
    def __init__(self, input_dim, hidden_dim=128, gcn_layers=2, gcn_repetitions=1, layer_type="GatedGraphConv", activation=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_type = layer_type.lower()
        self.input_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU())
        self.gcn_repetitions = gcn_repetitions

        if activation is None or activation.lower() == "none":
            self.gcn_activation = None
        elif activation.lower() == "relu":
            self.gcn_activation = F.relu
        else:
            logging.error(f"Unknown activation '{activation}'.")
            exit(-1)

        if self.layer_type == "GatedGraphConv".lower():
            self.gcn = torch.nn.ModuleList(
                [torch_geometric.nn.GatedGraphConv(hidden_dim, num_layers=gcn_repetitions) for i in range(gcn_layers)])
            self.gcn_repetitions = 1
        elif self.layer_type == "GraphConv".lower():
            self.gcn = torch.nn.ModuleList(
                [torch_geometric.nn.GraphConv(hidden_dim, hidden_dim) for i in range(gcn_layers)])
        elif self.layer_type == "GATv2Conv".lower():
            self.gcn = torch.nn.ModuleList(
                [torch_geometric.nn.GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=1) for i in range(gcn_layers)])
        else:
            logging.error(f"Unknown graph layer '{layer_type}'.")
            exit(-1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.input_mlp(x)
        for layer in self.gcn:
            for i in range(self.gcn_repetitions):
                x = layer(x, edge_index)
                if self.gcn_activation:
                    x = self.gcn_activation(x)

        d_copy = copy(data)
        d_copy.x = x
        return d_copy

    def get_output_dims(self) -> Tuple[int, int, int]:
        return self.hidden_dim, 0, 0
