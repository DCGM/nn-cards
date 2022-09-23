# author: Pavel Ševčík

from pathlib import Path
from dataclasses import dataclass
from typing import List

import torch
import pandas as pd
from torch_geometric.data import Data
from torch.utils.data import Dataset

@dataclass
class Line:
    label: str
    x1: float
    x2: float
    y1: float
    y2: float

@dataclass
class Card:
    name: str
    width: int
    height: int
    lines: List[Line]

class GraphDataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        if isinstance(csv_path, str):
            csv_path = Path(csv_path)
        
        self.csv_path = csv_path
        self.cards = self._load_cards(csv_path)
        self.label_encoder = self._get_label_encoder(self.cards)
        self.transform = transform

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, index):
        card = self.cards[index]
        data = self.transform(self._card_to_data(card))
        return data

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
            lines = [Line(line["label"], line["startX"], line["endX"], line["startY"], line["endY"]) for line in lines]
            cards.append(Card(name, width, height, lines))
        
        return cards
    
    def _get_label_encoder(self, cards: List[Card]):
        all_labels = list({line.label for card in cards for line in card.lines})
        n_labels = len(all_labels)

        def encode(labels):
            labels = [all_labels.index(label) for label in labels]
            return torch.nn.functional.one_hot(labels, num_classes=n_labels)
        return encode
            
        
        


