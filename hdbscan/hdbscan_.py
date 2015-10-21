# -*- coding: utf-8 -*-
"""
HDBSCAN: Hierarchical Density-Based Spatial Clustering 
         of Applications with Noise
"""
# Author: Leland McInnes <leland.mcinnes@gmail.com>
#         Steve Astels <sastels@gmail.com>
#         John Healy <jchealy@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from scipy.sparse import issparse
from sklearn.neighbors import KDTree
from warnings import warn

try:
    from sklearn.utils import check_array
except ImportError:
    from sklearn.utils import check_arrays

    check_array = check_arrays

from ._hdbscan_linkage import (single_linkage,
                               mst_linkage_core,
                               mst_linkage_core_pdist,
                               mst_linkage_core_cdist,
                               label)
from ._hdbscan_tree import (get_points,
                            condense_tree,
                            compute_stability,
                            get_clusters)
from ._hdbscan_reachability import kdtree_pdist_mutual_reachability, kdtree_mutual_reachability, mutual_reachability
from .plots import CondensedTree, SingleLinkageTree, MinimumSpanningTree

try:
    from fastcluster import single
    HAVE_FASTCLUSTER = True
except ImportError:
    HAVE_FASTCLUSTER = False

def _hdbscan_small(X, min_cluster_size=5, min_samples=None, alpha=1.0,
                   metric='minkowski', p=2, gen_min_span_tree=False):
    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')

        distance_matrix = pairwise_distances(X, metric=metric, p=p)
    else:
        distance_matrix = pairwise_distances(X, metric=metric)

    mutual_reachability_ = mutual_reachability(distance_matrix,
                                               min_samples, alpha)

    min_spanning_tree = mst_linkage_core(mutual_reachability_)

    if gen_min_span_tree:
        result_min_span_tree = min_spanning_tree.copy()
        for index, row in enumerate(result_min_span_tree[1:], 1):
            candidates = np.where(np.isclose(mutual_reachability_[row[1]], row[2]))[0]
            candidates = np.intersect1d(candidates, min_spanning_tree[:index, :2].astype(int))
            candidates = candidates[candidates != row[1]]
            assert(len(candidates) > 0)
            row[0] = candidates[0]
    else:
        result_min_span_tree = None

    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    single_linkage_tree = label(min_spanning_tree)
    condensed_tree = condense_tree(single_linkage_tree,
                                   min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    cluster_list = get_clusters(condensed_tree, stability_dict)

    labels = -1 * np.ones(X.shape[0], dtype=int)
    probabilities = np.zeros(X.shape[0], dtype=float)
    for index, (cluster, prob) in enumerate(cluster_list):
        labels[cluster] = index
        probabilities[cluster] = prob
    return labels, probabilities, condensed_tree, single_linkage_tree, result_min_span_tree


def _hdbscan_small_kdtree(X, min_cluster_size=5, min_samples=None, alpha=1.0,
                          metric='minkowski', p=2, gen_min_span_tree=False):
    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')

        distance_matrix = pairwise_distances(X, metric=metric, p=p)
    else:
        distance_matrix = pairwise_distances(X, metric=metric)

    mutual_reachability_ = kdtree_mutual_reachability(X,
                                                      distance_matrix,
                                                      metric,
                                                      p=p,
                                                      min_points=min_samples,
                                                      alpha=alpha)

    min_spanning_tree = mst_linkage_core(mutual_reachability_)

    if gen_min_span_tree:
        result_min_span_tree = min_spanning_tree.copy()
        for index, row in enumerate(result_min_span_tree[1:], 1):
            candidates = np.where(np.isclose(mutual_reachability_[row[1]], row[2]))[0]
            candidates = np.intersect1d(candidates, min_spanning_tree[:index, :2].astype(int))
            candidates = candidates[candidates != row[1]]
            assert(len(candidates) > 0)
            row[0] = candidates[0]
    else:
        result_min_span_tree = None

    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    single_linkage_tree = label(min_spanning_tree)
    condensed_tree = condense_tree(single_linkage_tree,
                                   min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    cluster_list = get_clusters(condensed_tree, stability_dict)

    labels = -1 * np.ones(X.shape[0], dtype=int)
    probabilities = np.zeros(X.shape[0], dtype=float)
    for index, (cluster, prob) in enumerate(cluster_list):
        labels[cluster] = index
        probabilities[cluster] = prob
    return labels, probabilities, condensed_tree, single_linkage_tree, result_min_span_tree


def _hdbscan_large_kdtree(X, min_cluster_size=5, min_samples=None, alpha=1.0,
                          metric='minkowski', p=2, gen_min_span_tree=False):
    if p is None:
        p = 2

    mutual_reachability_ = kdtree_pdist_mutual_reachability(X, metric, p, min_samples, alpha)

    min_spanning_tree = mst_linkage_core_pdist(mutual_reachability_)

    if gen_min_span_tree:
        result_min_span_tree = min_spanning_tree.copy()
        for index, row in enumerate(result_min_span_tree[1:], 1):
            candidates = np.where(np.isclose(mutual_reachability_[row[1]], row[2]))[0]
            candidates = np.intersect1d(candidates, min_spanning_tree[:index, :2].astype(int))
            candidates = candidates[candidates != row[1]]
            assert(len(candidates) > 0)
            row[0] = candidates[0]
    else:
        result_min_span_tree = None

    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    single_linkage_tree = label(min_spanning_tree)
    condensed_tree = condense_tree(single_linkage_tree,
                                   min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    cluster_list = get_clusters(condensed_tree, stability_dict)

    labels = -1 * np.ones(X.shape[0], dtype=int)
    probabilities = np.zeros(X.shape[0], dtype=float)
    for index, (cluster, prob) in enumerate(cluster_list):
        labels[cluster] = index
        probabilities[cluster] = prob
    return labels, probabilities, condensed_tree, single_linkage_tree, result_min_span_tree


def _hdbscan_large_kdtree_fastcluster(X, min_cluster_size=5, min_samples=None, alpha=1.0,
                                      metric='minkowski', p=2, gen_min_span_tree=False):
    if p is None:
        p = 2

    mutual_reachability_ = kdtree_pdist_mutual_reachability(X, metric,
                                                            p, min_samples, alpha)

    single_linkage_tree = single(mutual_reachability_)
    condensed_tree = condense_tree(single_linkage_tree,
                                   min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    cluster_list = get_clusters(condensed_tree, stability_dict)

    labels = -1 * np.ones(X.shape[0], dtype=int)
    probabilities = np.zeros(X.shape[0], dtype=float)
    for index, (cluster, prob) in enumerate(cluster_list):
        labels[cluster] = index
        probabilities[cluster] = prob
    return labels, probabilities, condensed_tree, single_linkage_tree, None

def _hdbscan_large_kdtree_cdist(X, min_cluster_size=5, min_samples=None, alpha=1.0,
                                metric='minkowski', p=2, gen_min_span_tree=False):

    if p is None:
        p = 2

    dim = X.shape[0]
    min_samples = min(dim - 1, min_samples)

    if metric == 'minkowski':
        tree = KDTree(X, metric=metric, p=p)
    else:
        tree = KDTree(X, metric=metric)

    core_distances = tree.query(X, k=min_samples)[0][:,-1]

    min_spanning_tree = mst_linkage_core_cdist(X, core_distances, metric, p)
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    single_linkage_tree = label(min_spanning_tree)
    condensed_tree = condense_tree(single_linkage_tree,
                                   min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    cluster_list = get_clusters(condensed_tree, stability_dict)

    labels = -1 * np.ones(X.shape[0], dtype=int)
    probabilities = np.zeros(X.shape[0], dtype=float)
    for index, (cluster, prob) in enumerate(cluster_list):
        labels[cluster] = index
        probabilities[cluster] = prob
    return labels, probabilities, condensed_tree, single_linkage_tree, None


def hdbscan(X, min_cluster_size=5, min_samples=None, alpha=1.0,
            metric='minkowski', p=2,
            algorithm='best', gen_min_span_tree=False):
    """Perform HDBSCAN clustering from a vector array or distance matrix.
    
    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.
        
    min_cluster_size : int optional
        The minimum number of samples in a group for that group to be
        considered a cluster; groupings smaller than this size will be left
        as noise.

    min_samples : int, optional
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        defaults to the min_cluster_size.

    metric : string, or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.
        (default minkowski)

    p : int, optional
        p value to use if using the minkowski metric. (default 2)

    alpha : float, optional
        A distance scaling parameter as used in robust single linkage.
        See (K. Chaudhuri and S. Dasgupta  "Rates of convergence
        for the cluster tree."). (default 1.0)

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
            * ``large_kdtree_low_memory``

    gen_min_span_tree : bool, optional
        Whether to generate the minimum spanning tree for later analysis.
        (default False)

    Returns
    -------
    labels : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

    probabilities : array [n_samples]
        Cluster membership stringths for each point. Noisy samples are assigned 0.

    condensed_tree : record array
        The condensed cluster hierarchy used to generate clusters.

    single_linkage_tree : array [n_samples - 1, 4]
        The single linkage tree produced during clustering in scipy
        hierarchical clustering format
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).

    min_spanning_tree : array [n_samples - 1, 3]
        The minimum spanning as an edgelist. If gen_min_span_tree was False
        this will be None.
        
    References
    ----------
    R. Campello, D. Moulavi, and J. Sander, "Density-Based Clustering Based on
    Hierarchical Density Estimates"
    In: Advances in Knowledge Discovery and Data Mining, Springer, pp 160-172.
    2013
    """
    if min_samples is None:
        min_samples = min_cluster_size

    if type(min_samples) is not int or type(min_cluster_size) is not int:
        raise ValueError('Min samples and min cluster size must be integers!')

    if min_samples <= 0 or min_cluster_size <= 0:
        raise ValueError('Min samples and Min cluster size must be positive integers')

    X = check_array(X, accept_sparse='csr')

    if algorithm != 'best':
        if algorithm == 'small':
            return _hdbscan_small(X, min_cluster_size, min_samples,
                                  alpha, metric, p, gen_min_span_tree)
        elif algorithm == 'small_kdtree':
            return _hdbscan_small_kdtree(X, min_cluster_size,
                                         min_samples, alpha, metric,
                                         p, gen_min_span_tree)
        elif algorithm == 'large_kdtree':
            return _hdbscan_large_kdtree(X, min_cluster_size,
                                         min_samples, alpha, metric,
                                         p, gen_min_span_tree)
        elif algorithm == 'large_kdtree_fastcluster':
            return _hdbscan_large_kdtree_fastcluster(X, min_cluster_size,
                                                     min_samples, alpha, metric,
                                                     p, gen_min_span_tree)
        elif algorithm == 'large_kdtree_low_memory':
            return _hdbscan_large_kdtree_cdist(X, min_cluster_size,
                                               min_samples, alpha, metric,
                                               p, gen_min_span_tree)
        else:
            raise TypeError('Unknown algorithm type %s specified' % algorithm)

    if issparse(X) or metric not in KDTree.valid_metrics:  # We can't do much with sparse matrices ...
        return _hdbscan_small(X, min_cluster_size, min_samples, alpha,
                              metric, p, gen_min_span_tree)
    elif X.shape[0] < 4000:
        return _hdbscan_small_kdtree(X, min_cluster_size,
                                     min_samples, alpha, metric,
                                     p, gen_min_span_tree)
    elif X.shape[0] < 30000:
        if HAVE_FASTCLUSTER:
            return _hdbscan_large_kdtree_fastcluster(X, min_cluster_size,
                                                     min_samples, alpha, metric,
                                                     p, gen_min_span_tree)
        else:
            return _hdbscan_large_kdtree(X, min_cluster_size,
                                         min_samples, alpha, metric,
                                         p, gen_min_span_tree)
    else:
        return _hdbscan_large_kdtree_cdist(X, min_cluster_size,
                                           min_samples, alpha, metric,
                                           p, gen_min_span_tree)



class HDBSCAN(BaseEstimator, ClusterMixin):
    """Perform HDBSCAN clustering from vector array or distance matrix.
    
    HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
    with Noise. Performs DBSCAN over varying epsilon values and integrates 
    the result to find a clustering that gives the best stability over epsilon.
    This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN),
    and be more robust to parameter selection.
    
    Parameters
    ----------
    min_cluster_size : int, optional
        The minimum size of clusters; single linkage splits that contain
        fewer points than this will be considered points "falling out" of a
        cluster rather than a cluster splitting into two new clusters.
        
    min_samples : int, optional
        The number of samples in a neighbourhood for a point to be
        considered a core point. (defaults to min_cluster_size)
        
    
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.
        (default euclidean)

    p : int, optional
        p value to use if using the minkowski metric. (default None)

    alpha : float, optional
        A distance scaling parameter as used in robust single linkage.
        See (K. Chaudhuri and S. Dasgupta  "Rates of convergence
        for the cluster tree."). (default 1.0)

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
            * ``large_kdtree_low_memory``

    gen_min_span_tree: bool, optional
        Whether to generate the minimum spanning tree with regard
        to mutual reachability distance for later analysis.
        (default False)
        
    Attributes
    ----------
    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    probabilities_ : array, shape = [n_samples]
        The strength with which each sample is a member of its assigned
        cluster. Noise points have probability zero; points in clusters
        have values assigned proportional to the degree that they
        persist as part of the cluster.

    condensed_tree_ : CondensedTree object
        The condensed tree produced by HDBSCAN. The object has methods
        for converting to pandas, networkx, and plotting.

    single_linkage_tree_ : SingleLinkageTree object
        The single linkage tree produced by HDBSCAN. The object has methods
        for converting to pandas, networkx, and plotting.

    minimum_spanning_tree_ : MinimumSpanningTree object
        The minimum spanning tree of the mutual reachability graph generated
        by HDBSCAN. Note that this is not generated by default and will only
        be available if `gen_min_span_tree` was set to True on object creation.
        Even then in some optimized cases a tre may not be generated.
        
    References
    ----------
    R. Campello, D. Moulavi, and J. Sander, "Density-Based Clustering Based on
    Hierarchical Density Estimates"
    In: Advances in Knowledge Discovery and Data Mining, Springer, pp 160-172.
    2013
    """

    def __init__(self, min_cluster_size=5, min_samples=None,
                 metric='euclidean', alpha=1.0, p=None,
                 algorithm='best', gen_min_span_tree=False):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha

        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.gen_min_span_tree = gen_min_span_tree

        self._condensed_tree = None
        self._single_linkage_tree = None
        self._min_spanning_tree = None
        self._raw_data = None

    def fit(self, X, y=None):
        """Perform HDBSCAN clustering from features or distance matrix.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
        """
        X = check_array(X, accept_sparse='csr')
        if self.metric != 'precomputed':
            self._raw_data = X
        (self.labels_,
         self.probabilities_,
         self._condensed_tree,
         self._single_linkage_tree,
         self._min_spanning_tree) = hdbscan(X, **self.get_params())
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
    def condensed_tree_(self):
        if self._condensed_tree is not None:
            return CondensedTree(self._condensed_tree)
        else:
            warn('No condensed tree was generated; try running fit first.')
            return None

    @property
    def single_linkage_tree_(self):
        if self._single_linkage_tree is not None:
            return SingleLinkageTree(self._single_linkage_tree)
        else:
            warn('No single linkage tree was generated; try running fit first.')
            return None

    @property
    def minimum_spanning_tree_(self):
        if self._min_spanning_tree is not None:
            if self._raw_data is not None:
                return MinimumSpanningTree(self._min_spanning_tree, self._raw_data)
            else:
                warn(
                    'No raw data is available; this may be due to using a precomputed metric matrix.'
                    'No minimum spanning tree will be provided without raw data.')
                return None
        else:
            warn('No minimum spanning tree was generated. \n'
                 'This may be due to optimized algorithm variations that skip\n'
                 'explicit generation of the spanning tree.')
            return None
