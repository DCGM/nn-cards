# file dataset.py
# author Pavel Ševčík

import functools
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod

import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph

class GraphBuild(ABC):
    @abstractmethod
    def __call__(self, graph) -> Data:
        """When overridden should build a graph and encode it as Data object"""
        pass

class DataBuild(ABC):
    @abstractmethod
    def __call__(self, data: Data, graph) -> Data:
        """When overriden should modify a given Data object"""
        pass

class NullDataBuild(DataBuild):
    def __call__(self, data: Data, graph) -> Data:
        return data

class KnnRectangeCenterBuild(GraphBuild):
    def __init__(self, k, rectangle_coords, leave_pos_attr=False, num_workers=0):
        self.rectange_coords = rectangle_coords
        self.leave_pos_attr = leave_pos_attr
        self.knn_transform = KNNGraph(k=k, num_workers=num_workers)
    
    def __call__(self, graph) -> Data:
        to_tensor = functools.partial(torch.tensor, dtype=torch.float)
        pos = torch.stack(
            [
                (to_tensor(graph["nodes"][start].to_numpy()) + to_tensor(graph["nodes"][end].to_numpy())) * 0.5
                    for start, end in self.rectange_coords
            ], dim=1
        )
        data = Data(pos=pos)
        data = self.knn_transform(data)

        if not self.leave_pos_attr:
            del data.pos
        return data

class SequentialDataBuild(DataBuild):
    def __init__(self, data_builds: List[DataBuild]):
        self.data_builds = data_builds
    
    def __call__(self, data: Data, graph) -> Data:
        for data_build in self.data_builds:
            data = data_build(data, graph)
        return data

class AddVectorAttr(DataBuild):
    def __init__(self, attr_name: str, fields: List[str]):
        self.attr_name = attr_name
        self.fields = fields
    
    def __call__(self, data: Data, graph) -> Data:
        to_tensor = functools.partial(torch.tensor, dtype=torch.float)
        attr_value = torch.stack(
            [to_tensor(graph["nodes"][field].to_numpy()) for field in self.fields], dim=1
        )
        setattr(data, self.attr_name, attr_value)
        return data

class OneHotEncoder:
    def __init__(self, classes: List[str]):
        self.classes = classes
    
    def encode(self, labels: List[str]) -> torch.LongTensor:
        indices = torch.tensor([
            self.classes.index(item) for item in labels
        ], dtype=torch.long)
        one_hot_encoded = torch.nn.functional.one_hot(
            indices,
            num_classes=len(self.classes)
        )
        return one_hot_encoded

    def decode(self, values: torch.LongTensor) -> List[str]:
        labels = torch.argmax(values, dim=1)
        return [self.classes[i] for i in labels]

class AddOneHotAttrEdgeClassifier(DataBuild):
    def __init__(self, attr_name: str, field: str, encoder):
        self.encoder = encoder
        self.attr_name = attr_name
        self.field = field

    def __call__(self, data: Data, graph) -> Data:
        src, dst = data["edge_index"]
        labels = []
        for s, d in zip(src, dst):
            if s + 1 == d:
                labels.append(str(1))
            else:
                labels.append(str(0))
        one_hot_encoded = self.encoder.encode(labels).float()
        setattr(data, self.attr_name, one_hot_encoded)
        return data

class AddOneHotAttr(DataBuild):
    def __init__(self, attr_name: str, field: str, encoder):
        self.encoder = encoder
        self.attr_name = attr_name
        self.field = field

    def __call__(self, data: Data, graph) -> Data:
        field_value = graph["nodes"][self.field]
        one_hot_encoded = self.encoder.encode(field_value).float()
        setattr(data, self.attr_name, one_hot_encoded)
        return data

class GraphDataset(Dataset):
    def __init__(self, csv_path: Path, graph_features, graph_build, data_build):
        if isinstance(csv_path, str):
            csv_path = Path(csv_path)
        
        self.csv_path = csv_path
        self.data_build = data_build
        self.graph_build = graph_build

        self.graphs = self._load_graphs(csv_path, graph_features)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        graph = self.graphs[index]
        data = self.graph_build(graph)
        data = self.data_build(data, graph)
        return data

    def _load_graphs(self, csv_path: Path, graph_features: List[str]):
        csv_data = pd.read_csv(csv_path)
        graphs = []
        for graph_features_values, node_features_values in csv_data.groupby(graph_features):
            graph = {
                feature: feature_value
                for feature, feature_value in zip(graph_features, graph_features_values)
            }
            graph["nodes"] = node_features_values
            graphs.append(graph)
        return graphs
