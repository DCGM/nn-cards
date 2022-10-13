# file nets.py
# author Michal Hradiš, Pavel Ševčík

import logging
from abc import ABC, abstractmethod
from copy import copy
from turtle import forward
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class Net(ABC):
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
    elif net_type == "edge_nodeconcat":
        return EdgeNodeConcatNet(**config)
    else:
        msg = f"Unknown network type '{net_type}'."
        logging.error(msg)
        raise ValueError(msg)

class SequentialBackbone(torch.nn.Module, Net):
    def __init__(self, backbone_config):
        super().__init__()
        if len(backbone_config) < 1:
            msg = f"At least one net is required to build a sequential backbone"
            logging.error(msg)
            raise ValueError(msg)
        
        nets = [net_factory(backbone_config[0])]
        for cfg in backbone_config[1:]:
            cfg["input_dims"] = nets[-1].get_output_dims()
            nets.append(net_factory(cfg))

        self.nets = torch.nn.ModuleList(nets)

    def forward(self, data):
        for net in self.nets:
            data = net(data)
        return data

    def get_output_dims(self) -> Tuple[int, int, int]:
        return self.nets[-1].get_output_dims()

class IdentityNet(torch.nn.Module, Net):
    def __init__(self, input_dims=None, output_dim=None):
        super().__init__()
        self.input_dims = input_dims

    def forward(self, batch):
        return copy(batch)
    
    def get_output_dims(self) -> Tuple[int, int, int]:
        return self.input_dims

class EdgeNodeConcatNet(torch.nn.Module, Net):
    def __init__(self, input_dims):
        super().__init__()
        self.input_dims = input_dims

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        src = x[edge_index[0]]
        dst = x[edge_index[1]]
        edge_concat = torch.cat([src, dst], dim=-1)
        d_copy = copy(data)
        d_copy.edge_attr = edge_concat
        return d_copy
    
    def get_output_dims(self) -> Tuple[int, int, int]:
        return self.input_dims[0], 2*self.input_dims[0], self.input_dims[2]


class MLP(torch.nn.Module, Net):

    def __init__(self, target, input_dims, output_dim=None, hidden_dim=128, depth=4):
        super().__init__()
        self.target = target
        self.mapping = {
            "node": "x",
            "edge": "edge_attr",
            "graph": "graph_attr"
        }
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.input_dim = self.input_dims[list(self.mapping.keys()).index(self.target)]
        if output_dim is None:
            self.output_dim = self.hidden_dim
        else:
            self.output_dim = output_dim

        layers = [ torch.nn.Linear(self.input_dim, self.hidden_dim)]
        for i in range(depth - 1):
            layers += [torch.nn.ReLU(), torch.nn.Linear(self.hidden_dim, self.hidden_dim)]
        layers += [torch.nn.ReLU(), torch.nn.Linear(self.hidden_dim, self.output_dim)]
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

    def get_output_dims(self) -> Tuple[int, int, int]:
        sizes = list(self.input_dims)
        sizes[list(self.mapping.keys()).index(self.target)] = self.output_dim
        return tuple(sizes)


class GCN(torch.nn.Module, Net):
    def __init__(self, input_dims, hidden_dim=128, gcn_layers=2, gcn_repetitions=1, layer_type="GatedGraphConv", activation=None):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.layer_type = layer_type.lower()
        self.input_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.input_dims[0], hidden_dim),
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
        return self.hidden_dim, self.input_dims[1], self.input_dims[2]
