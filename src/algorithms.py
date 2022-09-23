# author: Pavel Ševčík

import numpy as np

def k_nearest_neighbours(points, k) -> tuple[np.ndarray, np.ndarray]:
    r"""Finds k nearest neighbours for all points
    
    Args:
        points: a numpy array of shape (n_points, 2)
    
    Returns:
        A tuple of the matrix listing indices the k nearest neighbours for each point (n_points, k)
        and the matrix listing corresponding distances (n_points, k)"""
    # (a1-b1)**2 + (a2-b2)**2 = a1**2 + a2**2 - 2*(a1*b1 + a2*b2) + b1**2 + b2**2
    squared = np.sum(points**2, axis=1)
    distance_matrix = squared - 2*points@points.T + squared.reshape(-1, 1)
    indices = np.argsort(distance_matrix, axis=1)
    distances = np.sort(distance_matrix, axis=1)
    return indices[:, 1:k+1], distances[:, 1:k+1]

