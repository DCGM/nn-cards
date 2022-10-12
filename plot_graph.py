# file plot_graph.py
# author Pavel Ševčík, Kristína Hostačná

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
from torch_geometric.data import Data

from src.config import Config
from src.utils import json_str_or_path

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data-path", type=Path, help="Path to the csv data file", required=True)
    parser.add_argument("--model-config", type=json_str_or_path,
                            help="Json string or path to a json file containing model config.", required=True)
    parser.add_argument("--data-config", type=json_str_or_path,
                        help="Json string or path to a json file containing data config.", required=True)
    parser.add_argument("-i", "--indices", type=int, nargs="+", help="Indicies of graphs to be plotted.", required=True)
    args = parser.parse_args()
    return args

def plot_data(data: Data):
    img = np.zeros((4000,1200, 3), dtype=np.uint8)
    color = (255,0,0)
    color_edge = (0,0,255)

    for rect_coords in data.rect_coords:
        x1, x2, y1, y2 = rect_coords.round().int().numpy()
        img = cv2.rectangle(img, (x1,y1), (x2,y2), color, 1)
    
    for e in data.edge_index.T:
        pos = []
        for i in e:
            x1, x2, y1, y2 = data.rect_coords[i].round().int().numpy()
            pos.append(((x1+x2)//2, (y1+y2)//2))
        img = cv2.line(img, pos[0], pos[1], color_edge, 1)

    return img

def main():
    args = parse_arguments()

    config = Config()
    dataset_factory, model = config.configure(args.data_config, args.model_config)
    dataset = dataset_factory(args.data_path)

    n_subdivisions = list(range(10))
    def get_dataset(n_subdivisions):
        args_copy = deepcopy(args)
        args_copy.data_config["graph_build"]["n_subdivisions"] = n_subdivisions
        
        dataset_factory, model = config.configure(args_copy.data_config, args_copy.model_config)
        dataset = dataset_factory(args_copy.data_path)
        return dataset
        
    for i in args.indices:
        item = dataset[i]
        img = plot_data(item)
        cv2.imshow("Graph", img)
        cv2.waitKey(0)
    

if __name__ == "__main__":
    main()