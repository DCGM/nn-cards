# file algorithms.py
# author Pavel Ševčík

import torch

def k_nearest_neighbours(points, k) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Finds k nearest neighbours for all points
    
    Args:
        points: a numpy array of shape (n_points, 2)
    
    Returns:
        A tuple of the matrix listing indices the k nearest neighbours for each point (n_points, k)
        and the matrix listing corresponding distances (n_points, k)"""
    return k_nearest_neighbours_outside_group(points, k, 1)

def k_nearest_neighbours_outside_group(points, k, group_size) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Finds k nearest neighbours outside their group for all points
    
    Args:
        points: a numpy array of shape (n_points, 2)
        group_size: a span of indices belonging to the same group 
    
    Returns:
        A tuple of the matrix listing indices the k nearest neighbours for each point (n_points, k)
        and the matrix listing corresponding distances (n_points, k)"""
    distance_matrix = l2_distance_matrix(points)
    for i in range(0, len(points), group_size):
        distance_matrix[i:i+group_size,i:i+group_size] = float("inf")
    
    indices = torch.argsort(distance_matrix, axis=1)
    distances = torch.sort(distance_matrix, axis=1)
    return indices[:, k], distances[:, k]


def l2_distance_matrix(points) -> torch.Tensor:
    # (a1-b1)**2 + (a2-b2)**2 = a1**2 + a2**2 - 2*(a1*b1 + a2*b2) + b1**2 + b2**2
    squared = torch.sum(points**2, axis=1)
    distance_matrix = squared - 2*points@points.T + squared.unsqueeze(-1)
    return distance_matrix