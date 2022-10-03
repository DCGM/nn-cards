# file heads.py
# author Pavel Ševčík

import logging
from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from .nets import net_factory
from .dataset import AddVectorAttr, DataBuild, NullDataBuild, AddOneHotAttr

class Head(torch.nn.Module, ABC):
    @abstractmethod
    def compute_loss(self, output, batch) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def evaluate(self, batches) -> Dict[str, torch.Tensor]:
        pass

    def get_data_build(self) -> DataBuild:
        return NullDataBuild()
    
class HeadFactory:
    def __init__(self, input_dim):
        self.input_dim = input_dim
    
    def __call__(self, head_config) -> List[Head]:
        """Returns a list of heads according to the configuration"""
        type = head_config["type"]
        del head_config["type"]
        if type == "node_cls":
            return NodeClassificationHead(self.input_dim, **head_config)
        elif type == "cos_edge_cls":
            return CosEdgeClassificationHead(self.input_dim, **head_config)
        elif type == "node_regr":
            return NodeRegressionHead(self.input_dim, **head_config)
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
        return AddOneHotAttr(self.field, self.field, self.classes)

    def evaluate(self, batches) -> Dict[str, torch.Tensor]:
        losses = []
        correct = torch.tensor(0)
        counter = torch.tensor(0)

        for batch in batches:
            batch = self(batch)
            x = batch.x

            label = getattr(batch, self.field)
            mask = getattr(batch, f"{self.field}_mask", None)
            if mask:
                x = x[mask]
                label = label[mask]
            
            loss = self.criterion(x, label)
            losses.append(loss)
            pred_label = torch.argmax(x, dim=1)
            corr_label = torch.argmax(label, dim=1)

            correct += (pred_label == corr_label).sum().item()
            counter += pred_label.shape[0]
        
        return {
            f"{self.field}_eval": torch.mean(torch.tensor(losses)),
            f"{self.field}_eval_acc": correct / counter
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
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.net(batch)

        src, dst = edge_index
        score = torch.sum(x[src] * x[dst], dim=-1)

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

    def evaluate(self, batches) -> Dict[str, torch.Tensor]:
        losses = []

        for batch in batches:
            batch = self(batch)
            x = batch.x

            output = getattr(batch, self.field)
            mask = getattr(batch, f"{self.field}_mask", None)
            if mask:
                x = x[mask]
                output = output[mask]
            
            loss = self.criterion(x, output)
            losses.append(loss)
        
        return {
            f"{self.field}_eval": torch.mean(torch.tensor(losses))
        }

    def _validate_net_config(self, net_config):
        forbidden_attributes = ["input_dim", "output_dim"]
        for forbidden_attribute in forbidden_attributes:
            if forbidden_attribute in net_config:
                raise ValueError(f"Invalid 'net_config' for regression head, found '{forbidden_attribute}' attribute which is set automatically.")
