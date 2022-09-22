# author: Pavel Ševčík

from abc import ABC, abstractmethod

import numpy as np

from .algorithms import k_nearest_neighbours

def graph_builder_factory(graph_build_config):
    raise NotImplementedError()

class GraphBuilder(ABC):
    @abstractmethod
    def build_graph(self, node_coords: np.ndarray) -> np.ndarray:
        """When overriden in a subclass the method should return an edge matrix of the graph"""


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
        
