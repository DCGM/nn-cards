# file nets.py
# author Michal Hradiš

import logging
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric


def net_factory(config):
    net_type = config["type"].lower()
    del config["type"]
    net = None
    if net_type == "mlp":
        return MLP(**config)
    elif net_type == "gcn":
        return GCN(**config)
    elif net_type == "null":
        return NullNet()
    else:
        msg = f"Unknown network type '{net_type}'."
        logging.error(msg)
        raise ValueError(msg)

class NullNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        return batch.x

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, depth=4):
        super().__init__()

        layers = [ torch.nn.Linear(input_dim, hidden_dim)]
        for i in range(depth - 1):
            layers += [torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim)]
        layers += [torch.nn.Linear(hidden_dim, output_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.net(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, gcn_layers=2, gcn_repetitions=1, layer_type="GatedGraphConv", activation=None):
        super().__init__()
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
        else:
            logging.error(f"Unknown graph layer '{layer_type}'.")
            exit(-1)

        self.output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_mlp(x)
        for layer in self.gcn:
            for i in range(self.gcn_repetitions):
                x = layer(x, edge_index)
                if self.gcn_activation:
                    x = self.gcn_activation(x)

        x = self.output(x)

        return x

