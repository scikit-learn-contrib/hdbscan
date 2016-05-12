#cython: boundscheck=False, nonecheck=False, initializedcheck=False
# Tree handling (condensing, finding stable clusters) for hdbscan
# Authors: Leland McInnes
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree, BallTree
import gc

def mutual_reachability(distance_matrix, min_points=5, alpha=1.0):
    """Compute the weighted adjacency matrix of the mutual reachability
    graph of a distance matrix.
    
    Parameters
    ----------
    distance_matrix : array [n_samples, n_samples]
        Array of distances between samples.
        
    min_points : int optional
        The number of points in a neighbourhood for a point to be considered
        a core point. (defaults to 5)

    Returns
    -------
    mututal_reachability: array [n_samples, n_samples]
        Weighted adjacency matrix of the mutual reachability graph.
        
    References
    ----------
    R. Campello, D. Moulavi, and J. Sander, "Density-Based Clustering Based on
    Hierarchical Density Estimates"
    In: Advances in Knowledge Discovery and Data Mining, Springer, pp 160-172.
    2013
    """
    size = distance_matrix.shape[0]
    min_points = min(size - 1, min_points)
    try:
        core_distances = np.partition(distance_matrix, 
                                      min_points, 
                                      axis=0)[min_points]
    except AttributeError:
        core_distances = np.sort(distance_matrix,
                                 axis=0)[min_points]

    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha
                                  
    stage1 = np.where(core_distances > distance_matrix, 
                      core_distances, distance_matrix)
    result = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    return result

def kdtree_mutual_reachability(X, distance_matrix, metric, p=2, min_points=5, alpha=1.0, **kwargs):
    dim = distance_matrix.shape[0]
    min_points = min(dim - 1, min_points)

    if metric == 'minkowski':
        tree = KDTree(X, metric=metric, p=p)
    else:
        tree = KDTree(X, metric=metric, **kwargs)

    core_distances = tree.query(X, k=min_points)[0][:,-1]

    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha

    stage1 = np.where(core_distances > distance_matrix,
                      core_distances, distance_matrix)
    result = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    return result

def balltree_mutual_reachability(X, distance_matrix, metric, p=2, min_points=5, alpha=1.0, **kwargs):
    dim = distance_matrix.shape[0]
    min_points = min(dim - 1, min_points)

    tree = BallTree(X, metric=metric, **kwargs)

    core_distances = tree.query(X, k=min_points)[0][:,-1]

    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha

    stage1 = np.where(core_distances > distance_matrix,
                      core_distances, distance_matrix)
    result = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    return result

cdef np.ndarray[np.double_t, ndim=1] mutual_reachability_from_pdist(
        np.ndarray[np.double_t, ndim=1] core_distances, np.ndarray[np.double_t, ndim=1] dists, np.intp_t dim):

    cdef np.intp_t i
    cdef np.intp_t j
    cdef np.intp_t result_pos

    result_pos = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            if core_distances[i] > core_distances[j]:
                if core_distances[i] > dists[result_pos]:
                    dists[result_pos] = core_distances[i]

            else:
                if core_distances[j] > dists[result_pos]:
                    dists[result_pos] = core_distances[j]

            result_pos += 1

    return dists


def kdtree_pdist_mutual_reachability(X,  metric, p=2, min_points=5, alpha=1.0, **kwargs):

    dim = X.shape[0]
    min_points = min(dim - 1, min_points)

    if metric == 'minkowski':
        tree = KDTree(X, metric=metric, p=p)
    else:
        tree = KDTree(X, metric=metric, **kwargs)

    core_distances = tree.query(X, k=min_points)[0][:,-1]

    del tree
    gc.collect()

    dists = pdist(X, metric=metric, p=p, **kwargs)

    if alpha != 1.0:
        dists /= alpha

    dists = mutual_reachability_from_pdist(core_distances, dists, dim)

    return dists

def balltree_pdist_mutual_reachability(X, metric, p=2, min_points=5, alpha=1.0, **kwargs):

    dim = X.shape[0]
    min_points = min(dim - 1, min_points)

    tree = BallTree(X, metric=metric, **kwargs)

    core_distances = tree.query(X, k=min_points)[0][:,-1]

    del tree
    gc.collect()

    dists = pdist(X, metric=metric, p=p, **kwargs)

    if alpha != 1.0:
        dists /= alpha

    dists = mutual_reachability_from_pdist(core_distances, dists, dim)

    return dists
