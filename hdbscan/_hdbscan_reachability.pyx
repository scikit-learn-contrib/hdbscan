#cython: boundscheck=False, nonecheck=False, initializedcheck=False
# Tree handling (condensing, finding stable clusters) for hdbscan
# Authors: Leland McInnes
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree

def mutual_reachability(distance_matrix, min_points=5):
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
    dim = distance_matrix.shape[0]
    min_points = min(dim - 1, min_points)
    try:
        core_distances = np.partition(distance_matrix, 
                                      min_points, 
                                      axis=0)[min_points]
    except AttributeError:
        core_distances = np.sort(distance_matrix,
                                 axis=0)[min_points]        
                                  
    stage1 = np.where(core_distances > distance_matrix, 
                      core_distances, distance_matrix)
    result = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    return result

def kdtree_mutual_reachability(X, distance_matrix, metric, p=2, min_points=5):
    dim = distance_matrix.shape[0]
    min_points = min(dim - 1, min_points)

    if metric == 'minkowski':
        tree = KDTree(X, metric=metric, p=p)
    else:
        tree = KDTree(X, metric=metric)

    core_distances = tree.query(X, k=min_points)[0][:,-1]

    stage1 = np.where(core_distances > distance_matrix, 
                      core_distances, distance_matrix)
    result = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    return result


cpdef np.ndarray[np.double_t, ndim=1] kdtree_pdist_mutual_reachability(np.ndarray X, object metric, long long p=2, long long min_points=5):
    
    cdef long long dim
    cdef object tree
    cdef np.ndarray[np.double_t, ndim=1] core_distances
    cdef np.ndarray[np.double_t, ndim=1] dists
    cdef np.ndarray[np.double_t, ndim=1] result
    cdef long long i
    cdef long long result_pos

    dim = X.shape[0]
    min_points = min(dim - 1, min_points)
    
    if metric == 'minkowski':
        tree = KDTree(X, metric=metric, p=p)
    else:
        tree = KDTree(X, metric=metric)

    core_distances = tree.query(X, k=min_points)[0][:,-1]

    dists = pdist(X, metric=metric, p=p)

    result = np.empty(dists.shape[0], dtype=np.double)

    result_pos = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            if core_distances[i] > core_distances[j]:
                if core_distances[i] > dists[result_pos]:
                    result[result_pos] = core_distances[i]
                else:
                    result[result_pos] = dists[result_pos]
            else:
                if core_distances[j] > dists[result_pos]:
                    result[result_pos] = core_distances[j]
                else:
                    result[result_pos] = dists[result_pos]
            result_pos += 1

    return result
