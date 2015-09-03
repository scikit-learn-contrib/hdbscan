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
import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree
try:
    from sklearn.utils import check_array
except ImportError:
    from sklearn.utils import check_arrays
    check_array = check_arrays

from ._hdbscan_linkage import single_linkage, mst_linkage_core, label
from ._hdbscan_tree import (get_points,
                            condense_tree, 
                            compute_stability, 
                            get_clusters)
from .plots import CondensedTree, SingleLinkageTree, MinimumSpanningTree
  
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
    return result.astype(np.double)

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
    return result.astype(np.double)


def hdbscan(X, min_cluster_size=5, min_samples=None, metric='minkowski', p=2):
    """Perform HDBSCAN clustering from a vector array or distance matrix.
    
    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.
        
    min_cluster_size : int optional
        The minimum number of samples in a groupo for that group to be 
        considered a cluster; groupings smaller than this size will be left
        as noise.

    min_samples : int, optional
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        defaults to the min_cluster_size.

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.

    Returns
    -------
    labels : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

    condensed_tree : record array
        The condensed cluster hierarchy used to generate clusters.
        
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
    
    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')
        if X.shape[0] > 32000:
            # sklearn pairwise_distance segfaults for large comparisons
            distance_matrix = squareform(pdist(X, metric=metric, p=p))
        else:
            distance_matrix = pairwise_distances(X, metric=metric, p=p)
    elif metric != 'precomputed' and X.shape[0] > 32000:
        # sklearn pairwise_distance segfaults for large comparisons
        distance_matrix = squareform(pdist(X, metric=metric, p=p))
    else:
        distance_matrix = pairwise_distances(X, metric=metric)
        
    if metric in KDTree.valid_metrics and X.shape[1] < 8:
        mutual_reachability = kdtree_mutual_reachability(X, 
                                                         distance_matrix,
                                                         metric,
                                                         p=p,
                                                         min_points=min_samples)
    else:
        mutual_reachability = mutual_reachability(distance_matrix,
                                                  min_samples)

    min_spanning_tree = mst_linkage_core(mutual_reachability)
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
    
    single_linkage_tree = label(min_spanning_tree)
    condensed_tree, new_points = condense_tree(single_linkage_tree, 
                                               get_points(single_linkage_tree),
                                               min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    cluster_list = get_clusters(condensed_tree, stability_dict, new_points)
    
    labels = -1 * np.ones(distance_matrix.shape[0], dtype=int)
    for index, cluster in enumerate(cluster_list):
        labels[cluster] = index
    return labels, condensed_tree, single_linkage_tree, min_spanning_tree

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
        The minumum size of clusters; single linkage splits that contain
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
        
    Attributes
    ----------
    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.
        
    References
    ----------
    R. Campello, D. Moulavi, and J. Sander, "Density-Based Clustering Based on
    Hierarchical Density Estimates"
    In: Advances in Knowledge Discovery and Data Mining, Springer, pp 160-172.
    2013
    """
    
    def __init__(self, min_cluster_size=5, min_samples=None, 
                 metric='euclidean', p=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
            
        self.metric = metric
        self.p = p

        self._condensed_tree = None
        self._single_linkage_tree = None
        self._min_spanning_tree = None
        
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
        (self.labels_, 
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
    def condensed_tree(self):
        if self._condensed_tree is not None:
            return CondensedTree(self._condensed_tree)
        else:
            return None

    @property
    def single_linkage_tree(self):
        if self._condensed_tree is not None:
            return SingleLinkageTree(self._single_linkage_tree)
        else:
            return None

    @property
    def minimum_spanning_tree(self):
        if self._min_spanning_tree is not None:
            return MinimumSpanningTree(self._min_spanning_tree)
        else:
            return None
