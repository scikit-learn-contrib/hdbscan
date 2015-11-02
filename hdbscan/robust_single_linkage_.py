# -*- coding: utf-8 -*-
"""
Robust Single Linkage: Density based single linkage clustering.
"""
# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from scipy.sparse import issparse

try:
    from sklearn.utils import check_array
except ImportError:
    from sklearn.utils import check_arrays

    check_array = check_arrays

from ._hdbscan_linkage import single_linkage, mst_linkage_core_cdist, label
from ._hdbscan_boruvka import KDTreeBoruvkaAlgorithm, BallTreeBoruvkaAlgorithm
from dist_metrics import DistanceMetric
from ._hdbscan_reachability import mutual_reachability
from .plots import SingleLinkageTree
from sklearn.neighbors import KDTree, BallTree

from warnings import warn

FAST_METRICS = KDTree.valid_metrics + BallTree.valid_metrics

def _rsl_generic(X, cut, k=5, alpha=1.4142135623730951, gamma=5, metric='minkowski', p=2):

    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')

        distance_matrix = pairwise_distances(X, metric=metric, p=p)
    else:
        distance_matrix = pairwise_distances(X, metric=metric)

    mutual_reachability_ = mutual_reachability(distance_matrix, k)

    min_spanning_tree = mst_linkage_core(mutual_reachability_)
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    single_linkage_tree = label(min_spanning_tree)
    single_linkage_tree = SingleLinkageTree(single_linkage_tree)

    labels = single_linkage_tree.get_clusters(cut, gamma)

    return labels, single_linkage_tree

def _rsl_prims_kdtree(X, cut, k=5, alpha=1.4142135623730951, gamma=5, metric='minkowski', p=2):

    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')
    elif p is None:
        p = 2 # Unused, but needs to be integer; assume euclidean

    dim = X.shape[0]
    k = min(dim - 1, k)

    tree = KDTree(X, metric=metric)

    dist_metric = DistanceMetric.get_metric(metric)

    core_distances = tree.query(X, k=k)[0][:,-1]
    min_spanning_tree = mst_linkage_core_cdist(X, core_distances, dist_metric)

    single_linkage_tree = label(min_spanning_tree)
    single_linkage_tree = SingleLinkageTree(single_linkage_tree)

    labels = single_linkage_tree.get_clusters(cut, gamma)

    return labels, single_linkage_tree

def _rsl_prims_balltree(X, cut, k=5, alpha=1.4142135623730951, gamma=5, metric='minkowski', p=2):

    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')
    elif p is None:
        p = 2 # Unused, but needs to be integer; assume euclidean

    dim = X.shape[0]
    k = min(dim - 1, k)

    tree = BallTree(X, metric=metric)

    dist_metric = DistanceMetric.get_metric(metric)

    core_distances = tree.query(X, k=k)[0][:,-1]
    min_spanning_tree = mst_linkage_core_cdist(X, core_distances, dist_metric)

    single_linkage_tree = label(min_spanning_tree)
    single_linkage_tree = SingleLinkageTree(single_linkage_tree)

    labels = single_linkage_tree.get_clusters(cut, gamma)

    return labels, single_linkage_tree

def _rsl_boruvka_kdtree(X, min_cluster_size=5, min_samples=None, alpha=1.0,
                            metric='minkowski', p=2, leaf_size=10,
                            algorithm='best', gen_min_span_tree=False):

    dim = X.shape[0]
    min_samples = min(dim - 1, min_samples)

    tree = KDTree(X, metric=metric, leaf_size=leaf_size)
    alg = KDTreeBoruvkaAlgorithm(tree, min_samples, metric=metric)
    min_spanning_tree = alg.spanning_tree()

    single_linkage_tree = label(min_spanning_tree)
    single_linkage_tree = SingleLinkageTree(single_linkage_tree)

    labels = single_linkage_tree.get_clusters(cut, gamma)

    return labels, single_linkage_tree

def _rsl_boruvka_balltree(X, min_cluster_size=5, min_samples=None, alpha=1.0,
                              metric='minkowski', p=2, leaf_size=30,
                              algorithm='best', gen_min_span_tree=False):

    dim = X.shape[0]
    min_samples = min(dim - 1, min_samples)

    tree = BallTree(X, metric=metric, leaf_size=leaf_size)
    alg = BallTreeBoruvkaAlgorithm(tree, min_samples, metric=metric)
    min_spanning_tree = alg.spanning_tree()

    single_linkage_tree = label(min_spanning_tree)
    single_linkage_tree = SingleLinkageTree(single_linkage_tree)

    labels = single_linkage_tree.get_clusters(cut, gamma)

    return labels, single_linkage_tree

def robust_single_linkage(X, cut, k=5, alpha=1.4142135623730951, gamma=5, metric='minkowski', p=2, algorithm=None):
    """Perform robust single linkage clustering from a vector array
    or distance matrix.

    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    cut : float
        The reachability distance value to cut the cluster heirarchy at
        to derive a flat cluster labelling.

    k : int, optional
        Reachability distances will be computed with regard to the `k`
        nearest neighbors. (default 5)

    alpha : float, optional
        Distance scaling for reachability distance computation. Reachability
        distance is computed as $max \{ core_k(a), core_k(b), 1/\alpha d(a,b) \}$.
        (default sqrt(2))

    gamma : int, optional
        Ignore any clusters in the flat clustering with size less than gamma,
        and declare points in such clusters as noise points. (default 5)

    metric : string, or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.

    algorithm : string, optional
        Exactly which algorithm to use; hdbscan has variants specialised
        for different characteristics of the data. By default this is set
        to ``best`` which chooses the "best" algorithm given the nature of
        the data. You can force other options if you believe you know
        better. Options are:
            * ``generic``
            * ``best``
            * ``prims_kdtree``
            * ``prims_balltree``
            * ``boruvka_kdtree``
            * ``boruvka_balltree``

    Returns
    -------
    labels : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

    single_linkage_tree : array [n_samples - 1, 4]
        The single linkage tree produced during clustering in scipy
        hierarchical clustering format
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).

    References
    ----------
    K. Chaudhuri and S. Dasgupta.
    "Rates of convergence for the cluster tree."
    In Advances in Neural Information Processing Systems, 2010.

    """

    if type(k) is not int or k < 1:
        raise ValueError('k must be an integer greater than zero!')

    if type(alpha) is not float or alpha < 1.0:
        raise ValueError('alpha must be a float greater than or equal to 1.0!')

    if type(gamma) is not int or gamma < 1:
        raise ValueError('gamma must be an integer greater than zero!')

    X = check_array(X, accept_sparse='csr')

    if algorithm is not None:
        if algorithm == 'generic':
            return _rsl_generic(X, cut, k, alpha, gamma, metric, p)
        elif algorithm == 'prims_kdtree':
            return _rsl_prims_kdtree(X, cut, k, alpha, gamma, metric, p)
        elif algorithm == 'prims_balltree':
            return _rsl_prims_balltree(X, cut, k, alpha, gamma, metric, p)
        elif algorithm == 'boruvka_kdtree':
            return _rsl_boruvka_kdtree(X, cut, k, alpha, gamma, metric, p)
        elif algorithm == 'boruvka_balltree':
            return _rsl_boruvka_balltree(X, cut, k, alpha, gamma, metric, p)
        else:
            raise TypeError('Unknown algorithm type %s specified' % algorithm)

    if issparse(X) or metric not in FAST_METRICS:  # We can't do much with sparse matrices ...
        return _rsl_generic(X, cut, k, alpha, gamma, metric, p)
    elif metric in KDTree.valid_metrics:
        # Need heuristic to decide when to go to boruvka; still debugging for now
        if True:
            return _rsl_prims_kdtree(X, cut, k, alpha, gamma, metric, p)
        else:
            return _rsl_boruvka_kdtree(X, cut, k, alpha, gamma, metric, p)
    else: # Metric is a valid BallTree metric
        # Need heuristic to decide when to go to boruvka; still debugging for now
        if True:
            return _rsl_prims_kdtree(X, cut, k, alpha, gamma, metric, p)
        else:
            return _rsl_boruvka_balltree(X, cut, k, alpha, gamma, metric, p)



class RobustSingleLinkage (BaseEstimator, ClusterMixin):
    """Perform robust single linkage clustering from a vector array
    or distance matrix.

    Roust single linkage is a modified version of single linkage that
    attempts to be more robust to noise. Specifically the goal is to
    more accurately approximate the level set tree of the unknown
    probability density function from which the sample data has
    been drawn.

    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    cut : float
        The reachability distance value to cut the cluster heirarchy at
        to derive a flat cluster labelling.

    k : int, optional
        Reachability distances will be computed with regard to the `k`
        nearest neighbors. (default 5)

    alpha : float, optional
        Distance scaling for reachability distance computation. Reachability
        distance is computed as $max \{ core_k(a), core_k(b), 1/\alpha d(a,b) \}$.
        (default sqrt(2))

    gamma : int, optional
        Ignore any clusters in the flat clustering with size less than gamma,
        and declare points in such clusters as noise points. (default 5)

    metric : string, or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.

    algorithm : string, optional
        Exactly which algorithm to use; hdbscan has variants specialised
        for different characteristics of the data. By default this is set
        to ``best`` which chooses the "best" algorithm given the nature of
        the data. You can force other options if you believe you know
        better. Options are:
            * ``small``
            * ``small_kdtree``
            * ``large_kdtree``
            * ``large_kdtree_fastcluster``

    Attributes
    -------
    labels_ : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

    cluster_hierarchy_ : SingleLinkageTree object
        The single linkage tree produced during clustering. This object provides
        several methods for:
            * Plotting
            * Generating a flat clustering
            * Exporting to NetworkX
            * Exporting to Pandas

    References
    ----------
    K. Chaudhuri and S. Dasgupta.
    "Rates of convergence for the cluster tree."
    In Advances in Neural Information Processing Systems, 2010.

    """

    def __init__(self, k=5, alpha=1.4142135623730951, gamma=5, metric='euclidean', p=None):

        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.metric = metric
        self.p = p

        self._cluster_hierarchy_ = None

    def fit(self, X, y=None):
        """Perform robust single linkage clustering from features or distance matrix.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
        """
        X = check_array(X, accept_sparse='csr')
        self.labels_, self._cluster_hierarchy = robust_single_linkage(X, **self.get_params())
        return self

    def fit_predict(self, X, y=None):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            cluster labels
        """

        self.fit(X)
        return self.labels_

    @property
    def cluster_hierarchy_(self):
        if self._cluster_hierarchy_ is not None:
            return SingleLinkageTree(self._cluster_hierarchy_)
        else:
            warn('No single linkage tree was generated; try running fit first.')
            return None
