# file dataset.py
# author Pavel Ševčík

from pathlib import Path
from dataclasses import dataclass
from typing import List

import torch
import pandas as pd
from torch_geometric.data import Data
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, csv_path: Path, data_config, attribute_build):
        if isinstance(csv_path, str):
            csv_path = Path(csv_path)
        
        self.csv_path = csv_path
        self.attribute_build = attribute_build

        graph_features = data_config["graph_features"]
        self.graphs = self._load_graphs(csv_path, graph_features)

        self.edge_build_transform = self._get_edge_build_transform(data_config["edge_build"])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        graph = self.graphs[index]
        data = self.edge_build_transform(graph)
        data = self.attribute_build(data, graph)
        return data

    def _load_graphs(self, csv_path: Path, graph_features: List[str]):
        csv_data = pd.read_csv(csv_path)
        graphs = []
        for graph_features_values, node_features_values in csv_data.groupby(graph_features):
            graph = {
                feature: feature_value
                for feature, feature_value in zip(graph_features, graph_features_values)
            }
            graph.nodes = [
                {
                    feature: feature_value
                    for feature, feature_value in node_features_values.iterrows()
                }
            ]
            graphs.append(graph)
            

    def _card_to_data(self, card: Card) -> Data:
        x1 = torch.tensor([line.x1 for line in card.lines])
        x2 = torch.tensor([line.x2 for line in card.lines])
        y1 = torch.tensor([line.y1 for line in card.lines])
        y2 = torch.tensor([line.y2 for line in card.lines])
        x = torch.stack((x1,x2,y1,y2), dim=1)
        label = self.label_encoder([line.label for line in card.lines])

        return Data(name=card.name, width=card.width, height=card.height, x=x, label=label)
        
    def _load_cards(self, csv_path: Path) -> List[Card]:
        convertors = {
            "startX": float,
            "startY": float,
            "endX": float,
            "endY": float,
            "cardHeight": int,
            "cardWidth": int
        }
        data = pd.read_csv(csv_path, converters=convertors)
        cards = []
        for (name, width, height), lines in data.groupby(by=["cardName", "cardWidth", "cardHeight"]):
            lines = [Line(label, x1, x2, y1, y2) for label, x1, x2, y1, y2 in zip(lines["label"], lines["startX"], lines["endX"], lines["startY"], lines["endY"])]
            cards.append(Card(name, width, height, lines))
        
        return cards
    
    def _get_label_encoder(self, cards: List[Card]):
        all_labels = list({line.label for card in cards for line in card.lines})
        n_labels = len(all_labels)

        def encode(labels):
            labels = torch.LongTensor([all_labels.index(label) for label in labels])
            return torch.nn.functional.one_hot(labels, num_classes=n_labels).float()
        return encode
            
        


