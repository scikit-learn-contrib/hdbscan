# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# mutual reachability distance computations
# Authors: Leland McInnes
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from numpy.math cimport INFINITY, isfinite
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree, BallTree
import gc


def mutual_reachability(distance_matrix, min_points=5, alpha=1.0):
    """Compute the weighted adjacency matrix of the mutual reachability
    graph of a distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray, shape (n_samples, n_samples)
        Array of distances between samples.

    min_points : int, optional (default=5)
        The number of points in a neighbourhood for a point to be considered
        a core point.

    Returns
    -------
    mututal_reachability: ndarray, shape (n_samples, n_samples)
        Weighted adjacency matrix of the mutual reachability graph.

    References
    ----------
    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates.
       In Pacific-Asia Conference on Knowledge Discovery and Data Mining
       (pp. 160-172). Springer Berlin Heidelberg.
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


cpdef sparse_mutual_reachability(object distance_matrix, np.intp_t min_points=5,
                                 float alpha=1.0, float max_dist=0.0):
    """ Compute mutual reachability for distance matrix. For best performance
    pass distance_matrix in CSR form which will modify it in place and return
    it without making a copy """

    # tocsr() is a fast noop if distance_matrix is already CSR
    D = distance_matrix.tocsr()
    # Convert to smallest supported data type if necessary
    if D.dtype not in (np.float32, np.float64):
        if D.dtype <= np.dtype(np.float32):
            D = D.astype(np.float32)
        else:
            D = D.astype(np.float64)

    # Call typed function which modifies D in place
    t = (D.data.dtype, D.indices.dtype, D.indptr.dtype)
    if t == (np.float32, np.int32, np.int32):
        sparse_mr_fast[np.float32_t, np.int32_t](D.data, D.indices, D.indptr,
                                                 min_points, alpha, max_dist)
    elif t == (np.float32, np.int64, np.int64):
        sparse_mr_fast[np.float32_t, np.int64_t](D.data, D.indices, D.indptr,
                                                 min_points, alpha, max_dist)
    elif t == (np.float64, np.int32, np.int32):
        sparse_mr_fast[np.float64_t, np.int32_t](D.data, D.indices, D.indptr,
                                                 min_points, alpha, max_dist)
    elif t == (np.float64, np.int64, np.int64):
        sparse_mr_fast[np.float64_t, np.int64_t](D.data, D.indices, D.indptr,
                                                 min_points, alpha, max_dist)
    else:
        raise Exception("Unsupported CSR format {}".format(t))

    return D


ctypedef fused mr_indx_t:
    np.int32_t
    np.int64_t

ctypedef fused mr_data_t:
    np.float32_t
    np.float64_t

cdef sparse_mr_fast(np.ndarray[dtype=mr_data_t, ndim=1] data,
                    np.ndarray[dtype=mr_indx_t, ndim=1] indices,
                    np.ndarray[dtype=mr_indx_t, ndim=1] indptr,
                    mr_indx_t min_points,
                    mr_data_t alpha,
                    mr_data_t max_dist):
    cdef mr_indx_t row, col, colptr
    cdef mr_data_t mr_dist
    cdef np.ndarray[dtype=mr_data_t, ndim=1] row_data
    cdef np.ndarray[dtype=mr_data_t, ndim=1] core_distance

    core_distance = np.empty(data.shape[0], dtype=data.dtype)

    for row in range(indptr.shape[0]-1):
        row_data = data[indptr[row]:indptr[row+1]].copy()
        if min_points < row_data.shape[0]:
            # sort is faster for small arrays because of lower startup cost but
            # partition has worst case O(n) runtime for larger ones.
            # https://stackoverflow.com/questions/43588711/numpys-partition-slower-than-sort-for-small-arrays
            if row_data.shape[0] > 200:
                row_data.partition(min_points)
            else:
                row_data.sort()
            core_distance[row] = row_data[min_points]
        else:
            core_distance[row] = INFINITY

    if alpha != <mr_data_t>1.0:
        data /= alpha

    for row in range(indptr.shape[0]-1):
        for colptr in range(indptr[row],indptr[row+1]):
            col = indices[colptr]
            mr_dist = max(core_distance[row], core_distance[col], data[colptr])
            if isfinite(mr_dist):
                data[colptr] = mr_dist
            elif max_dist > 0:
                data[colptr] = max_dist


def kdtree_mutual_reachability(X, distance_matrix, metric, p=2, min_points=5,
                               alpha=1.0, **kwargs):
    dim = distance_matrix.shape[0]
    min_points = min(dim - 1, min_points)

    if metric == 'minkowski':
        tree = KDTree(X, metric=metric, p=p)
    else:
        tree = KDTree(X, metric=metric, **kwargs)

    core_distances = tree.query(X, k=min_points)[0][:, -1]

    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha

    stage1 = np.where(core_distances > distance_matrix,
                      core_distances, distance_matrix)
    result = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    return result


def balltree_mutual_reachability(X, distance_matrix, metric, p=2, min_points=5,
                                 alpha=1.0, **kwargs):
    dim = distance_matrix.shape[0]
    min_points = min(dim - 1, min_points)

    tree = BallTree(X, metric=metric, **kwargs)

    core_distances = tree.query(X, k=min_points)[0][:, -1]

    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha

    stage1 = np.where(core_distances > distance_matrix,
                      core_distances, distance_matrix)
    result = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    return result


cdef np.ndarray[np.double_t, ndim=1] mutual_reachability_from_pdist(
        np.ndarray[np.double_t, ndim=1] core_distances,
        np.ndarray[np.double_t, ndim=1] dists, np.intp_t dim):

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


def kdtree_pdist_mutual_reachability(X,  metric, p=2, min_points=5, alpha=1.0,
                                     **kwargs):

    dim = X.shape[0]
    min_points = min(dim - 1, min_points)

    if metric == 'minkowski':
        tree = KDTree(X, metric=metric, p=p)
    else:
        tree = KDTree(X, metric=metric, **kwargs)

    core_distances = tree.query(X, k=min_points)[0][:, -1]

    del tree
    gc.collect()

    dists = pdist(X, metric=metric, p=p, **kwargs)

    if alpha != 1.0:
        dists /= alpha

    dists = mutual_reachability_from_pdist(core_distances, dists, dim)

    return dists


def balltree_pdist_mutual_reachability(X, metric, p=2, min_points=5, alpha=1.0,
                                       **kwargs):

    dim = X.shape[0]
    min_points = min(dim - 1, min_points)

    tree = BallTree(X, metric=metric, **kwargs)

    core_distances = tree.query(X, k=min_points)[0][:, -1]

    del tree
    gc.collect()

    dists = pdist(X, metric=metric, p=p, **kwargs)

    if alpha != 1.0:
        dists /= alpha

    dists = mutual_reachability_from_pdist(core_distances, dists, dim)

    return dists
