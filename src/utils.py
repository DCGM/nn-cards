# file utils.py
# author Pavel Ševčík

import json
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

def json_str_or_path(s):
    try:
        value = json.loads(s)
        return value
    except ValueError:
        pass
    with open(s) as f:
        return json.load(f)
