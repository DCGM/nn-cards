# file utils.py
# author Pavel Ševčík

import json
from collections import defaultdict
from typing import Dict
import numpy as np
import ast


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

def parse_LN_inputs(line_coords):
    inputs = []

    for line_coord in line_coords:
        line_coord = line_coord.replace("'", "")
        line_coord = ast.literal_eval(line_coord)

        s = np.sum(line_coord, axis=1)
        upper_left = line_coord[np.argmin(s)]
        lower_right = line_coord[np.argmax(s)]
        coords = np.concatenate((upper_left, lower_right))
        inputs.append(coords)

    return np.array(inputs)
