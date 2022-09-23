# author: Pavel Ševčík

import logging
from abc import ABC, abstractmethod

import numpy as np

from .algorithms import k_nearest_neighbours

class GraphBuilder(ABC):
    @abstractmethod
    def build_graph(self, node_coords: np.ndarray) -> np.ndarray:
        """When overriden in a subclass the method should return an edge matrix of the graph"""

def graph_builder_factory(graph_build_config) -> GraphBuilder:
    """Returns a graph builder that is reponsible for creating a graph from a set of nodes"""
    builder_type = graph_build_config["type"].lower()
    del graph_build_config["type"]
    if builder_type == "knearest":
        return KNearestGraphBuilder(**graph_build_config)
    else:
        msg = f"Unknown graph buider type '{builder_type}'"
        logging.error(msg)
        raise ValueError(msg)


class KNearestGraphBuilder(GraphBuilder):
    def __init__(self, k_nearest: int):
        self.k_nearest = k_nearest
    
    def build_graph(self, node_coords: np.ndarray) -> np.ndarray:
        """Build a graph from a set of node 2D-coords connecting k-nearest neighbours
        
        Args:
            node_coords: a numpy array of shape (n_nodes, 2)
            
        Returns:
            An edge index of the corresponding graph"""

        indices, _ = k_nearest_neighbours(node_coords, self.k_nearest)
        return np.array([[i, j] for i in range(node_coords.shape[0]) for j in indices[i]])
        
