# author: Pavel Ševčík

from collections import defaultdict
from typing import Dict


class Stats:
    def __init__(self):
        self.data = defaultdict(lambda: [])
    
    def add(self, values: Dict[str, float]):
        for key, value in values.items():
            self.data[key].append(value)
    
    def clear(self):
        for value in self.data.values():
            value.clear()
