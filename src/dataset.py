# author: Pavel Ševčík

from pathlib import Path

from torch.utils.data import Dataset

from .graphbuilders import graph_builder_factory

class GraphDataset(Dataset):
    def __init__(self, csv_path: Path, graph_build_config=None):
        if isinstance(csv_path, str):
            csv_path = Path(csv_path)
        
        self.csv_path = csv_path
        self.graph_builder = graph_builder_factory(graph_build_config)