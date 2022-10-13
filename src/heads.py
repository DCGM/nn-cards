# file heads.py
# author Pavel Ševčík

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch_geometric.data import Data

from .nets import net_factory
from .dataset import AddVectorAttr, DataBuild, NullDataBuild, AddEncodedAttr, ClassListEncoder, AddOneHotAttrEdgeClassifier
from .evals import ArgMaxClassificationEval

class Head(torch.nn.Module, ABC):
    def __init__(self, target):
        self.target = target
        self.mapping = {
            "node": "x",
            "edge": "edge_attr",
            "graph": "graph_attr"
        }

    @abstractmethod
    def compute_loss(self, output) -> Dict[str, torch.Tensor]:
        pass

    def eval_add(self, output):
        pass

    def eval_reset(self):
        pass

    def eval_get(self) -> Dict[str, Any]:
        return {}

    def get_data_build(self) -> DataBuild:
        return NullDataBuild()
    
    def _get_output_value(self, data: Data):
        return getattr(data, self.mapping[self.target])

class HeadFactory:
    def __init__(self, node_feature_size, edge_feature_size, graph_feature_size):
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.graph_feature_size = graph_feature_size
    
    def __call__(self, head_config) -> List[Head]:
        """Returns a list of heads according to the configuration"""
        type = head_config["type"]
        edge = head_config["edge"]
        del head_config["type"]
        if type == "cls":
            input_size = self.edge_feature_size if edge else self.node_feature_size
            return ClassificationHead(input_size, **head_config)
        elif type == "node_regr":
            return RegressionHead(self.node_feature_size, **head_config)
        else:
            msg = f"Unknown head type '{type}'."
            logging.error(msg)
            raise ValueError(msg)

class ClassificationHead(Head):
    def __init__(self, input_dim, target, field, net_config, classes):
        super().__init__(target)
        self._validate_net_config(net_config)
        net_config["input_dim"] = input_dim
        net_config["output_dim"] = len(classes)
        self.field = field
        self.net = net_factory(net_config)
        self.classes = classes
        self.criterion = torch.nn.CrossEntropyLoss()
        self.encoder = ClassListEncoder(self.classes)
        self.evaluator = ArgMaxClassificationEval(self.encoder)

    def forward(self, data):
        data = self.net(data)
        return data
    
    def compute_loss(self, data) -> Dict[str, torch.Tensor]:
        output_value = self._get_output_value(data)

        label = getattr(data, self.field)
        mask = getattr(data, f"{self.field}_mask", None)
        if mask:
            output_value = output_value[mask]
            label = label[mask]
        
        loss = self.criterion(output_value, label)
        return {self.field: loss}
    
    def get_data_build(self) -> DataBuild:
        return AddEncodedAttr(self.field, self.field, self.encoder)

    def eval_add(self, data):
        output_value = self._get_output_value(data)

        label = getattr(data, self.field)
        mask = getattr(data, f"{self.field}_mask", None)
        if mask:
            output_value = output_value[mask]
            label = label[mask]
            
        self.evaluator.add(output_value, label)

    def eval_reset(self):
        self.evaluator.reset()
    
    def eval_get(self) -> Dict[str, Any]:
        return {
            f"{self.field}_eval_{key}": value
                for key, value in self.evaluator.get_results().items()
        }

    def _validate_net_config(self, net_config):
        forbidden_attributes = ["input_dim", "output_dim"]
        for forbidden_attribute in forbidden_attributes:
            if forbidden_attribute in net_config:
                raise ValueError(f"Invalid 'net_config' for classification head, found '{forbidden_attribute}' attribute which is set automatically.")

class RegressionHead(Head):
    def __init__(self, input_dim, target, field, net_config, edge=False):
        super().__init__(target)
        self._validate_net_config(net_config)
        net_config["input_dim"] = input_dim
        net_config["output_dim"] = 1
        self.field = field
        self.net = net_factory(net_config)
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, data):
        data = self.net(data)
        return data

    def compute_loss(self, data) -> Dict[str, torch.Tensor]:
        output_value = data.x
        label = getattr(data, self.field)
        mask = getattr(data, f"{self.field}_mask", None)
        if mask:
            output_value = output_value[mask]
            label = label[mask]
        
        loss = self.criterion(output_value, label)
        return {self.field: loss}
    
    def get_data_build(self) -> DataBuild:
        return AddVectorAttr(self.field, [self.field])

    def _validate_net_config(self, net_config):
        forbidden_attributes = ["input_dim", "output_dim"]
        for forbidden_attribute in forbidden_attributes:
            if forbidden_attribute in net_config:
                raise ValueError(f"Invalid 'net_config' for regression head, found '{forbidden_attribute}' attribute which is set automatically.")
