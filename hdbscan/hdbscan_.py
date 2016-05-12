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
from sklearn.neighbors import KDTree, BallTree
from sklearn.externals.joblib import Memory
from sklearn.externals import six
from warnings import warn

#Try and work around older sklearn api
try:
    from sklearn.utils import check_array
except ImportError:
    from sklearn.utils import check_arrays

    check_array = check_arrays

from ._hdbscan_linkage import (single_linkage,
                               mst_linkage_core,
                               mst_linkage_core_vector,
                               label)
from ._hdbscan_tree import (condense_tree,
                            compute_stability,
                            get_clusters,
                            outlier_scores)
from ._hdbscan_reachability import mutual_reachability

from ._hdbscan_boruvka import KDTreeBoruvkaAlgorithm, BallTreeBoruvkaAlgorithm
from .dist_metrics import DistanceMetric

from .plots import CondensedTree, SingleLinkageTree, MinimumSpanningTree

FAST_METRICS = KDTree.valid_metrics + BallTree.valid_metrics


def _tree_to_labels(X, single_linkage_tree, min_cluster_size=10):
    """ Converts a pretrained tree and cluster size into a 
    set of labels and probabilities.
    """
    condensed_tree = condense_tree(single_linkage_tree,
                                   min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    labels, probabilities, stabilities = get_clusters(condensed_tree, stability_dict)

    return labels, probabilities, stabilities, condensed_tree, single_linkage_tree


def _hdbscan_generic(X, min_samples=5, alpha=1.0,
                     metric='minkowski', p=2, leaf_size=None, gen_min_span_tree=False, **kwargs):
    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')

        distance_matrix = pairwise_distances(X, metric=metric, p=p)
    else:
        distance_matrix = pairwise_distances(X, metric=metric, **kwargs)

    mutual_reachability_ = mutual_reachability(distance_matrix,
                                               min_samples, alpha)

    min_spanning_tree = mst_linkage_core(mutual_reachability_)

    #mst_linkage_core does not generate a full minimal spanning tree
    #If a tree is required then we must build the edges from the information
    #returned by mst_linkage_core (i.e. just the order of points to be merged)
    if gen_min_span_tree:
        result_min_span_tree = min_spanning_tree.copy()
        for index, row in enumerate(result_min_span_tree[1:], 1):
            candidates = np.where(np.isclose(mutual_reachability_[row[1]], row[2]))[0]
            candidates = np.intersect1d(candidates, min_spanning_tree[:index, :2].astype(int))
            candidates = candidates[candidates != row[1]]
            assert (len(candidates) > 0)
            row[0] = candidates[0]
    else:
        result_min_span_tree = None

    #Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    #Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    return single_linkage_tree, result_min_span_tree


def _hdbscan_prims_kdtree(X, min_samples=5, alpha=1.0,
                          metric='minkowski', p=2, leaf_size=40, gen_min_span_tree=False, **kwargs):
    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')
    elif p is None:
        p = 2  # Unused, but needs to be integer; assume euclidean

    # The Cython routines used require contiguous arrays
    if not X.flags['C_CONTIGUOUS']:
        X = np.array(X, dtype=np.double, order='C')

    size = X.shape[0]
    min_samples = min(size - 1, min_samples)

    tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)

    #TO DO: Deal with p for minkowski appropriately
    dist_metric = DistanceMetric.get_metric(metric, **kwargs)

    #Get distance to kth nearest neighbour
    core_distances = tree.query(X, k=min_samples,
                                dualtree=True,
                                breadth_first=True)[0][:, -1].copy(order='C')
    #Mutual reachability distance is implicite in mst_linkage_core_vector
    min_spanning_tree = mst_linkage_core_vector(X, core_distances, dist_metric, alpha)

    #Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    #Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    return single_linkage_tree, None


def _hdbscan_prims_balltree(X, min_samples=5, alpha=1.0,
                            metric='minkowski', p=2, leaf_size=40, gen_min_span_tree=False, **kwargs):
    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')
    elif p is None:
        p = 2  # Unused, but needs to be integer; assume euclidean

    # The Cython routines used require contiguous arrays
    if not X.flags['C_CONTIGUOUS']:
        X = np.array(X, dtype=np.double, order='C')

    size = X.shape[0]
    min_samples = min(size - 1, min_samples)

    tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)

    dist_metric = DistanceMetric.get_metric(metric, **kwargs)

    #Get distance to kth nearest neighbour
    core_distances = tree.query(X, k=min_samples,
                                dualtree=True,
                                breadth_first=True)[0][:, -1].copy(order='C')

    #Mutual reachability distance is implicite in mst_linkage_core_vector
    min_spanning_tree = mst_linkage_core_vector(X, core_distances, dist_metric, alpha)
    #Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
    #Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    return single_linkage_tree, None


def _hdbscan_boruvka_kdtree(X, min_samples=5, alpha=1.0,
                            metric='minkowski', p=2, leaf_size=40,
                            approx_min_span_tree=True,
                            gen_min_span_tree=False,
                            core_dist_n_jobs=4, **kwargs):
    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')

    if leaf_size < 3:
        leaf_size = 3

    if core_dist_n_jobs < 1:
        raise ValueError('Parallel core distance computation requires 1 or more jobs!')

    size = X.shape[0]
    min_samples = min(size - 1, min_samples)

    tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
    alg = KDTreeBoruvkaAlgorithm(tree, min_samples, metric=metric, leaf_size=leaf_size // 3,
                                 approx_min_span_tree=approx_min_span_tree, n_jobs=core_dist_n_jobs, **kwargs)
    min_spanning_tree = alg.spanning_tree()
    #Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
    #Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None


def _hdbscan_boruvka_balltree(X, min_samples=5, alpha=1.0,
                              metric='minkowski', p=2, leaf_size=40,
                              approx_min_span_tree=True,
                              gen_min_span_tree=False,
                              core_dist_n_jobs=4, **kwargs):
    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')

    if leaf_size < 3:
        leaf_size = 3

    if core_dist_n_jobs < 1:
        raise ValueError('Parallel core distance computation requires 1 or more jobs!')

    size = X.shape[0]
    min_samples = min(size - 1, min_samples)

    tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
    alg = BallTreeBoruvkaAlgorithm(tree, min_samples, metric=metric, leaf_size=leaf_size // 3,
                                   approx_min_span_tree=approx_min_span_tree, n_jobs=core_dist_n_jobs, **kwargs)
    min_spanning_tree = alg.spanning_tree()
    #Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
    #Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None


def hdbscan(X, min_cluster_size=5, min_samples=None, alpha=1.0,
            metric='minkowski', p=2, leaf_size=40,
            algorithm='best', memory=Memory(cachedir=None, verbose=0),
            approx_min_span_tree=True, gen_min_span_tree=False,
            core_dist_n_jobs=4, **kwargs):

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

    alpha : float, optional
        A distance scaling parameter as used in robust single linkage.
        See (K. Chaudhuri and S. Dasgupta  "Rates of convergence
        for the cluster tree."). (default 1.0)

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

    leaf_size : int, optional
        Leaf size for trees responsible for fast nearest
        neighbour queries. (default 40)

    algorithm : string, optional
        Exactly which algorithm to use; hdbscan has variants specialised 
        for different characteristics of the data. By default this is set
        to ``best`` which chooses the "best" algorithm given the nature of 
        the data. You can force other options if you believe you know 
        better. Options are:
            * ``best``
            * ``generic``
            * ``prims_kdtree``
            * ``prims_balltree``
            * ``boruvka_kdtree``
            * ``boruvka_balltree``

    memory : Instance of joblib.Memory or string (optional)
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    approx_min_span_tree : Bool, optional
        Whether to accept an only approximate minimum spanning tree.
        For some algorithms this can provide a significant speedup, but
        the resulting clustering may be of marginally lower quality.
        If you are willing to sacrifice speed for correctness you may want
        to explore this; in general this should be left at the default True.
        (default True)

    gen_min_span_tree : bool, optional
        Whether to generate the minimum spanning tree for later analysis.
        (default False)

    core_dist_n_jobs : int, optional
        Number of parallel jobs to run in core distance computations (if
        supported by the specific algorithm).
        (default 4)

    **kwargs : optional
        Arguments passed to the distance metric

    Returns
    -------
    labels : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

    probabilities : array [n_samples]
        Cluster membership strengths for each point. Noisy samples are assigned
        0.

    cluster_persistence : array, shape = [n_clusters]
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores can be guage the relative coherence of the clusters output
        by the algorithm.

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

    if alpha <= 0.0 or type(alpha) is int:
        raise ValueError('Alpha must be a positive value greater than 0!')

    if leaf_size < 1:
        raise ValueError('Leaf size must be greater than 0!')

    #Checks input and converts to an nd-array where possible
    X = check_array(X, accept_sparse='csr')
    #Python 2 and 3 compliant string_type checking
    if isinstance(memory, six.string_types):
        memory = Memory(cachedir=memory, verbose=0)

    if algorithm != 'best':
        if algorithm == 'generic':
            (single_linkage_tree, 
             result_min_span_tree) = \
                memory.cache(_hdbscan_generic)(X, min_samples, alpha, metric,
                                               p, leaf_size, gen_min_span_tree, **kwargs)
        elif algorithm == 'prims_kdtree':
            if metric not in KDTree.valid_metrics:
                raise ValueError("Cannot use Prim's with KDTree for this metric!")
            (single_linkage_tree, 
             result_min_span_tree) = \
                memory.cache(_hdbscan_prims_kdtree)(X, min_samples, alpha,
                                                    metric, p, leaf_size,
                                                    gen_min_span_tree, **kwargs)
        elif algorithm == 'prims_balltree':
            if metric not in BallTree.valid_metrics:
                raise ValueError("Cannot use Prim's with BallTree for this metric!")
            (single_linkage_tree, 
             result_min_span_tree) = \
                memory.cache(_hdbscan_prims_balltree)(X, min_samples, alpha,
                                                      metric, p, leaf_size,
                                                      gen_min_span_tree, **kwargs)
        elif algorithm == 'boruvka_kdtree':
            if metric not in BallTree.valid_metrics:
                raise ValueError("Cannot use Boruvka with KDTree for this metric!")
            (single_linkage_tree,
             result_min_span_tree) = \
                memory.cache(_hdbscan_boruvka_kdtree)(X, min_samples, alpha,
                                                      metric, p, leaf_size,
                                                      approx_min_span_tree,
                                                      gen_min_span_tree,
                                                      core_dist_n_jobs, **kwargs)
        elif algorithm == 'boruvka_balltree':
            if metric not in BallTree.valid_metrics:
                raise ValueError("Cannot use Boruvka with BallTree for this metric!")
            (single_linkage_tree,
             result_min_span_tree) = \
                memory.cache(_hdbscan_boruvka_balltree)(X, min_samples, alpha,
                                                        metric, p, leaf_size,
                                                        approx_min_span_tree,
                                                        gen_min_span_tree,
                                                        core_dist_n_jobs, **kwargs)
        else:
            raise TypeError('Unknown algorithm type %s specified' % algorithm)
    else:

        if issparse(X) or metric not in FAST_METRICS:  # We can't do much with sparse matrices ...
            (single_linkage_tree,
             result_min_span_tree) = \
                memory.cache(_hdbscan_generic)(X, min_samples,
                                               alpha, metric, p, leaf_size,
                                               gen_min_span_tree, **kwargs)
        elif metric in KDTree.valid_metrics:
            #TO DO: Need heuristic to decide when to go to boruvka; still debugging for now
            if X.shape[1] > 60:
                (single_linkage_tree,
                 result_min_span_tree) = \
                    memory.cache(_hdbscan_prims_kdtree)(X, min_samples, alpha,
                                                        metric, p, leaf_size,
                                                        gen_min_span_tree, **kwargs)
            else:
                (single_linkage_tree,
                 result_min_span_tree) = \
                    memory.cache(_hdbscan_boruvka_kdtree)(X, min_samples, alpha,
                                                          metric, p, leaf_size,
                                                          approx_min_span_tree,
                                                          gen_min_span_tree,
                                                          core_dist_n_jobs, **kwargs)
        else:  # Metric is a valid BallTree metric
            #TO DO: Need heuristic to decide when to go to boruvka; still debugging for now
            if X.shape[1] > 60:
                (single_linkage_tree,
                 result_min_span_tree) = \
                    memory.cache(_hdbscan_prims_kdtree)(X, min_samples, alpha,
                                                        metric, p, leaf_size,
                                                        gen_min_span_tree, **kwargs)
            else:
                (single_linkage_tree,
                 result_min_span_tree) = \
                    memory.cache(_hdbscan_boruvka_balltree)(X, min_samples, alpha,
                                                            metric, p, leaf_size,
                                                            approx_min_span_tree,
                                                            gen_min_span_tree,
                                                            core_dist_n_jobs, **kwargs)

    return _tree_to_labels(X, single_linkage_tree, min_cluster_size) + (result_min_span_tree,)

#Inherits from sklearn
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
            * ``best``
            * ``generic``
            * ``prims_kdtree``
            * ``prims_balltree``
            * ``boruvka_kdtree``
            * ``boruvka_balltree``

    leaf_size: int, optional
        If using a space tree algorithm (kdtree, or balltree) the number
        of points ina leaf node of the tree. This does not alter the
        resulting clustering, but may have an effect on the runtime
        of the algorithm. (default 40)

    memory : Instance of joblib.Memory or string (optional)
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    approx_min_span_tree : Bool, optional
        Whether to accept an only approximate minimum spanning tree.
        For some algorithms this can provide a significant speedup, but
        the resulting clustering may be of marginally lower quality.
        If you are willing to sacrifice speed for correctness you may want
        to explore this; in general this should be left at the default True.
        (default True)

    gen_min_span_tree: bool, optional
        Whether to generate the minimum spanning tree with regard
        to mutual reachability distance for later analysis.
        (default False)

    **kwargs : optional
        Arguments passed to the distance metric

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

    cluster_persistence_ : array, shape = [n_clusters]
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores can be guage the relative coherence of the clusters output
        by the algorithm.

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

    outlier_scores_ : array, shape = [n_samples]
        Outlier scores for clustered points; the larger the score the more
        outlier-like the point. Useful as an outlier detection technique.
        Based on the GLOSH algorithm by Campello, Moulavi, Zimek and Sander.

    References
    ----------
    R. Campello, D. Moulavi, and J. Sander, "Density-Based Clustering Based on
    Hierarchical Density Estimates"
    In: Advances in Knowledge Discovery and Data Mining, Springer, pp 160-172.
    2013

    R. Campello, D. Moulavi, A. Zimek, J. Sander, "Hierarchical Density Estimates
    for Data Clustering, Visualization, and Outlier Detection"
    In: ACM Transaction on Knowledge Discovery, Vol 10, No. 1, Article 5.
    2015
    """

    def __init__(self, min_cluster_size=5, min_samples=None,
                 metric='euclidean', alpha=1.0, p=None,
                 algorithm='best', leaf_size=40,
                 memory=Memory(cachedir=None, verbose=0),
                 approx_min_span_tree=True,
                 gen_min_span_tree=False, **kwargs):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha

        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory
        self.approx_min_span_tree = approx_min_span_tree
        self.gen_min_span_tree = gen_min_span_tree

        self._metric_kwargs = kwargs

        self._condensed_tree = None
        self._single_linkage_tree = None
        self._min_spanning_tree = None
        self._raw_data = None
        self._outlier_scores = None

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

        kwargs = self.get_params()
        kwargs.update(self._metric_kwargs)

        (self.labels_,
         self.probabilities_,
         self.cluster_persistence_,
         self._condensed_tree,
         self._single_linkage_tree,
         self._min_spanning_tree) = hdbscan(X, **kwargs)
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
    def outlier_scores_(self):
        if self._outlier_scores is not None:
            return self._outlier_scores
        else:
            self._outlier_scores = outlier_scores(self._condensed_tree)
            return self._outlier_scores

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
