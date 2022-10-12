# file heads.py
# author Pavel Ševčík

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch


from .nets import net_factory
from .dataset import AddVectorAttr, DataBuild, NullDataBuild, AddOneHotAttr, OneHotEncoder, AddOneHotAttrEdgeClassifier
from .evals import ArgMaxClassificationEval

class Head(torch.nn.Module, ABC):
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
    
class HeadFactory:
    def __init__(self, node_feature_size, edge_feature_size, graph_feature_size):
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.graph_feature_size = graph_feature_size
    
    def __call__(self, head_config) -> List[Head]:
        """Returns a list of heads according to the configuration"""
        type = head_config["type"]
        del head_config["type"]
        if type == "node_cls":
            return NodeClassificationHead(self.node_feature_size, **head_config)
        elif type == "cos_edge_cls":
            return CosEdgeClassificationHead(self.node_feature_size, **head_config)
        elif type == "node_regr":
            return NodeRegressionHead(self.node_feature_size, **head_config)
        else:
            msg = f"Unknown head type '{type}'."
            logging.error(msg)
            raise ValueError(msg)

class ClassificationHead(Head):
    def __init__(self, input_dim, field, net_config, classes):
        super().__init__()
        self._validate_net_config(net_config)
        net_config["input_dim"] = input_dim
        net_config["output_dim"] = len(classes)
        self.field = field
        self.net = net_factory(net_config)
        self.classes = classes
        self.criterion = torch.nn.CrossEntropyLoss()
        self.encoder = OneHotEncoder(self.classes)
        self.evaluator = ArgMaxClassificationEval(self.encoder)

    def compute_loss(self, output) -> Dict[str, torch.Tensor]:
        output_value = output.x
        label = getattr(output, self.field)
        mask = getattr(output, f"{self.field}_mask", None)
        if mask:
            output_value = output_value[mask]
            label = label[mask]
        
        loss = self.criterion(output_value, label)
        return {self.field: loss}
    
    def get_data_build(self) -> DataBuild:
        return AddOneHotAttr(self.field, self.field, self.encoder)

    def eval_add(self, output):
        x = output.x

        label = getattr(output, self.field)
        mask = getattr(output, f"{self.field}_mask", None)
        if mask:
            x = x[mask]
            label = label[mask]
            
        self.evaluator.add(x, label)

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

class NodeClassificationHead(ClassificationHead):
    def forward(self, batch):
        batch = self.net(batch)
        return batch

class CosEdgeClassificationHead(ClassificationHead):
    def compute_loss(self, output) -> Dict[str, torch.Tensor]:
        output_value = output.score
        label = getattr(output, self.field)
        mask = getattr(output, f"{self.field}_mask", None)
        if mask:
            output_value = output_value[mask]
            label = label[mask]

        loss = self.criterion(output_value, label)
        return {self.field: loss}

    def get_data_build(self) -> DataBuild:
        return AddOneHotAttrEdgeClassifier(self.field, self.field, self.encoder)

    def eval_add(self, output):
        x = output.score

        label = getattr(output, self.field)
        mask = getattr(output, f"{self.field}_mask", None)
        if mask:
            x = x[mask]
            label = label[mask]

        self.evaluator.add(x, label)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        src = x[edge_index[0]]
        dst = x[edge_index[1]]
        edge_concat = torch.cat([src, dst], dim=-1)
        score = self.net(edge_concat)

        batch.score = score
        return batch

class NodeRegressionHead(Head):
    
    def __init__(self, input_dim, field, net_config):
        super().__init__()
        self._validate_net_config(net_config)
        net_config["input_dim"] = input_dim
        net_config["output_dim"] = 1
        self.field = field
        self.net = net_factory(net_config)
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, batch):
        batch = self.net(batch)
        return batch

    def compute_loss(self, output) -> Dict[str, torch.Tensor]:
        output_value = output.x
        label = getattr(output, self.field)
        mask = getattr(output, f"{self.field}_mask", None)
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
