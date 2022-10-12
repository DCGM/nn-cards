# file dataset.py
# author Pavel Ševčík

import logging
import functools
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod

import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph

from .algorithms import l2_distance_matrix

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

class KnnRectangeEdgeBuild(GraphBuild):
    def __init__(self, k, rectangle_coords, n_subdivisions=1):
        self.k = k
        self.rectangle_coords = rectangle_coords
        self.n_subdivisions = n_subdivisions
        if len(self.rectangle_coords) != 2:
            msg = f"Invalid number of rectangle coords '{len(self.rectangle_coords)}', expected exactly 2."
            logging.error(msg)
            raise ValueError(msg)

    def __call__(self, graph) -> Data:
        to_tensor = functools.partial(torch.tensor, dtype=torch.float)
        rectangles = torch.stack(
            [
                to_tensor(graph["nodes"][attr].to_numpy())
                for coords in self.rectangle_coords for attr in coords
            ], dim=1
        )
        points_per_rect = 2**(self.n_subdivisions+2)
        points = []
        for rect in rectangles:
            points.extend(self._subdivide(self.n_subdivisions, rect))
        points = torch.stack(points)
        
        distance_matrix = l2_distance_matrix(points)
        # deny selfloops within rectangles
        for i in range(0, len(points), points_per_rect):
            distance_matrix[i:i+points_per_rect,i:i+points_per_rect] = float("inf")
        
        # reduce point distances to rect distances
        rect_distances = torch.zeros((len(rectangles), len(rectangles)))
        for i in range(len(rectangles)):
            for j in range(i,len(rectangles)):
                x_start = i * points_per_rect
                x_end = (i+1) * points_per_rect
                y_start = j * points_per_rect
                y_end = (j+1) * points_per_rect
                dist = torch.min(distance_matrix[y_start:y_end, x_start:x_end])
                
                rect_distances[i,j] = dist
                rect_distances[j,i] = dist
        
        indices = torch.argsort(rect_distances, axis=1)
        
        edge_index = []
        for rect_idx in range(len(rectangles)):
            edge_index.extend([[rect_idx, neigh_idx] for neigh_idx in indices[rect_idx,:self.k]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        return Data(edge_index=edge_index)
        
    def _subdivide(self, n_subdivisions, rect: torch.Tensor) -> torch.Tensor:
        x1, x2, y1, y2 = rect
        points = torch.tensor([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
        edges = zip(points, torch.roll(points, shifts=-1, dims=0))
        return torch.cat([self._subdivide_edge(n_subdivisions, a, b) for a, b in edges])
    
    def _subdivide_edge(self, n_subdivisions, a, b):
        if n_subdivisions == 0:
            return torch.stack((a,))
        else:
            mid = (a+b)/2
            first_part = self._subdivide_edge(n_subdivisions-1, a, mid)
            second_part = self._subdivide_edge(n_subdivisions-1, mid, b)
            return torch.cat((first_part, second_part))


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

class ClassListEncoder:
    def __init__(self, classes: List[str]):
        self.classes = classes
    
    def encode(self, labels: List[str]) -> torch.LongTensor:
        indices = torch.tensor([
            self.classes.index(item) for item in labels
        ], dtype=torch.long)
        return indices

    def decode(self, values: torch.LongTensor) -> List[str]:
        labels = torch.argmax(values, dim=1)
        return [self.classes[i] for i in labels]

class AddReadOrderEdgeAttr(DataBuild):
    def __init__(self, attr_name: str, field: str, encoder):
        self.attr_name = attr_name
        self.field = field
        self.encoder = encoder

    def __call__(self, data: Data, graph) -> Data:
        field_value = graph["nodes"][self.field].to_numpy()
        src, dst = data.edge_index
        labels = []
        for s, d in zip(src, dst):
            if int(field_value[s]) + 1 == int(field_value[d]):
                labels.append("1")
            else:
                labels.append("0")
        encoded = self.encoder.encode(labels)
        setattr(data, self.attr_name, encoded)
        return data

class AddEncodedAttr(DataBuild):
    def __init__(self, attr_name: str, field: str, encoder):
        self.encoder = encoder
        self.attr_name = attr_name
        self.field = field

    def __call__(self, data: Data, graph) -> Data:
        field_value = graph["nodes"][self.field]
        encoded = self.encoder.encode(field_value)
        setattr(data, self.attr_name, encoded)
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
        for graph_features_values, node_features_values in csv_data.groupby(by=graph_features):
            if len(graph_features) == 1:
                graph_features_values = (graph_features_values,)
            graph = {
                feature: feature_value
                for feature, feature_value in zip(graph_features, graph_features_values)
            }
            graph["nodes"] = node_features_values
            graphs.append(graph)
        return graphs
