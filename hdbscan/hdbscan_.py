# -*- coding: utf-8 -*-
"""
HDBSCAN: Hierarchical Density-Based Spatial Clustering
         of Applications with Noise
"""

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from scipy.sparse import issparse
from sklearn.neighbors import KDTree, BallTree
from joblib import Memory
from warnings import warn
from sklearn.utils import check_array
from joblib.parallel import cpu_count

from scipy.sparse import csgraph

from ._hdbscan_linkage import (
    mst_linkage_core,
    mst_linkage_core_vector,
    label,
)
from ._hdbscan_tree import (
    condense_tree,
    compute_stability,
    get_clusters,
    outlier_scores,
    simplify_hierarchy,
)
from ._hdbscan_reachability import mutual_reachability, sparse_mutual_reachability

from ._hdbscan_boruvka import KDTreeBoruvkaAlgorithm, BallTreeBoruvkaAlgorithm
from .dist_metrics import DistanceMetric

from .plots import CondensedTree, SingleLinkageTree, MinimumSpanningTree
from .prediction import PredictionData
from .branch_data import BranchDetectionData

KDTREE_VALID_METRICS = ["euclidean", "l2", "minkowski", "p", "manhattan", "cityblock", "l1", "chebyshev", "infinity"]
BALLTREE_VALID_METRICS = KDTREE_VALID_METRICS + [
    "braycurtis",
    "canberra",
    "dice",
    "hamming",
    "haversine",
    "jaccard",
    "mahalanobis",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
]
FAST_METRICS = KDTREE_VALID_METRICS + BALLTREE_VALID_METRICS + ["cosine", "arccos"]

# Author: Leland McInnes <leland.mcinnes@gmail.com>
#         Steve Astels <sastels@gmail.com>
#         John Healy <jchealy@gmail.com>
#
# License: BSD 3 clause
from numpy import isclose


def _tree_to_labels(
    X,
    single_linkage_tree,
    min_cluster_size=10,
    cluster_selection_method="eom",
    allow_single_cluster=False,
    match_reference_implementation=False,
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
    max_cluster_size=0,
    cluster_selection_epsilon_max=float('inf'),
):
    """Converts a pretrained tree and cluster size into a
    set of labels and probabilities.
    """
    condensed_tree = condense_tree(single_linkage_tree, min_cluster_size)
    if cluster_selection_persistence > 0.0:
        condensed_tree = simplify_hierarchy(condensed_tree, cluster_selection_persistence)
    stability_dict = compute_stability(condensed_tree)
    labels, probabilities, stabilities = get_clusters(
        condensed_tree,
        stability_dict,
        cluster_selection_method,
        allow_single_cluster,
        match_reference_implementation,
        cluster_selection_epsilon,
        max_cluster_size,
        cluster_selection_epsilon_max,
    )

    return (labels, probabilities, stabilities, condensed_tree, single_linkage_tree)


def _hdbscan_generic(
    X,
    min_samples=5,
    alpha=1.0,
    metric="minkowski",
    p=2,
    leaf_size=None,
    gen_min_span_tree=False,
    **kwargs
):
    if metric == "minkowski":
        distance_matrix = pairwise_distances(X, metric=metric, p=p)
    elif metric == "arccos":
        distance_matrix = pairwise_distances(X, metric="cosine", **kwargs)
    elif metric == "precomputed":
        # Treating this case explicitly, instead of letting
        #   sklearn.metrics.pairwise_distances handle it,
        #   enables the usage of numpy.inf in the distance
        #   matrix to indicate missing distance information.
        # TODO: Check if copying is necessary
        distance_matrix = X.copy()
    elif metric == "graph":
        assert issparse(X), f"Graphs must be passed as sparse arrays, was a {type(X)}."

        distance_matrix = X.copy()
    else:
        distance_matrix = pairwise_distances(X, metric=metric, **kwargs)

    if issparse(distance_matrix):
        # raise TypeError('Sparse distance matrices not yet supported')
        return _hdbscan_sparse_distance_matrix(
            distance_matrix,
            min_samples,
            alpha,
            metric,
            p,
            leaf_size,
            gen_min_span_tree,
            **kwargs
        )

    mutual_reachability_ = mutual_reachability(distance_matrix, min_samples, alpha)

    min_spanning_tree = mst_linkage_core(mutual_reachability_)

    # Warn if the MST couldn't be constructed around the missing distances
    if np.isinf(min_spanning_tree.T[2]).any():
        warn(
            "The minimum spanning tree contains edge weights with value "
            "infinity. Potentially, you are missing too many distances "
            "in the initial distance matrix for the given neighborhood "
            "size.",
            UserWarning,
        )

    # mst_linkage_core does not generate a full minimal spanning tree
    # If a tree is required then we must build the edges from the information
    # returned by mst_linkage_core (i.e. just the order of points to be merged)
    if gen_min_span_tree:
        result_min_span_tree = min_spanning_tree.copy()
        for index, row in enumerate(result_min_span_tree[1:], 1):
            candidates = np.where(isclose(mutual_reachability_[int(row[1])], row[2]))[0]
            candidates = np.intersect1d(
                candidates, min_spanning_tree[:index, :2].astype(int)
            )
            candidates = candidates[candidates != row[1]]
            assert len(candidates) > 0
            row[0] = candidates[0]
    else:
        result_min_span_tree = None

    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    return single_linkage_tree, result_min_span_tree


def _hdbscan_sparse_distance_matrix(
    X,
    min_samples=5,
    alpha=1.0,
    metric="minkowski",
    p=2,
    leaf_size=40,
    gen_min_span_tree=False,
    **kwargs
):
    assert issparse(X)

    # if the metric is not graph, compute mutual_reachability of distance matrix
    if metric == "graph":
        mutual_reachability_ = X.tocsr()

    else:
        # Check for connected component on X
        if csgraph.connected_components(X, directed=False, return_labels=False) > 1:
            raise ValueError(
                "Sparse distance matrix has multiple connected "
                "components!\nThat is, there exist groups of points "
                "that are completely disjoint -- there are no distance "
                "relations connecting them\n"
                "Run hdbscan on each component."
            )

        lil_matrix = X.tolil()

        # Compute sparse mutual reachability graph
        # if max_dist > 0, max distance to use when the reachability is infinite
        max_dist = kwargs.get("max_dist", 0.0)
        mutual_reachability_ = sparse_mutual_reachability(
            lil_matrix, min_points=min_samples, max_dist=max_dist, alpha=alpha
        )

    # Check connected component on mutual reachability
    # If more than one component, it means that even if the distance matrix X
    # has one component, there exists with less than `min_samples` neighbors
    if (
            csgraph.connected_components(
                mutual_reachability_, directed=False, return_labels=False
            )
            > 1
    ):
        raise ValueError(
            (
                "There exists points with less than %s neighbors. "
                "Ensure your distance matrix (or graph for metric= `graph`) has non zeros values for "
                "at least `min_sample`=%s neighbors for each points (i.e. K-nn graph), "
                "or specify a `max_dist` to use when distances are missing."
            )
            % (min_samples, min_samples)
        )

    # Compute the minimum spanning tree for the sparse graph
    sparse_min_spanning_tree = csgraph.minimum_spanning_tree(mutual_reachability_)

    # Convert the graph to scipy cluster array format
    nonzeros = sparse_min_spanning_tree.nonzero()
    nonzero_vals = sparse_min_spanning_tree[nonzeros]
    min_spanning_tree = np.vstack(nonzeros + (nonzero_vals,)).T

    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :][0]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None


def _hdbscan_prims_kdtree(
    X,
    min_samples=5,
    alpha=1.0,
    metric="minkowski",
    p=2,
    leaf_size=40,
    gen_min_span_tree=False,
    **kwargs
):
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    # The Cython routines used require contiguous arrays
    if not X.flags["C_CONTIGUOUS"]:
        X = np.array(X, dtype=np.double, order="C")

    tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)

    # TO DO: Deal with p for minkowski appropriately
    dist_metric = DistanceMetric.get_metric(metric, **kwargs)

    # Get distance to kth nearest neighbour
    core_distances = tree.query(
        X, k=min_samples + 1, dualtree=True, breadth_first=True
    )[0][:, -1].copy(order="C")

    # Mutual reachability distance is implicit in mst_linkage_core_vector
    min_spanning_tree = mst_linkage_core_vector(X, core_distances, dist_metric, alpha)

    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None


def _hdbscan_prims_balltree(
    X,
    min_samples=5,
    alpha=1.0,
    metric="minkowski",
    p=2,
    leaf_size=40,
    gen_min_span_tree=False,
    **kwargs
):
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    # The Cython routines used require contiguous arrays
    if not X.flags["C_CONTIGUOUS"]:
        X = np.array(X, dtype=np.double, order="C")

    tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)

    dist_metric = DistanceMetric.get_metric(metric, **kwargs)

    # Get distance to kth nearest neighbour
    core_distances = tree.query(
        X, k=min_samples + 1, dualtree=True, breadth_first=True
    )[0][:, -1].copy(order="C")

    # Mutual reachability distance is implicit in mst_linkage_core_vector
    min_spanning_tree = mst_linkage_core_vector(X, core_distances, dist_metric, alpha)
    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None


def _hdbscan_boruvka_kdtree(
    X,
    min_samples=5,
    alpha=1.0,
    metric="minkowski",
    p=2,
    leaf_size=40,
    approx_min_span_tree=True,
    gen_min_span_tree=False,
    core_dist_n_jobs=4,
    **kwargs
):
    if leaf_size < 3:
        leaf_size = 3

    if core_dist_n_jobs < 1:
        core_dist_n_jobs = max(cpu_count() + 1 + core_dist_n_jobs, 1)

    if X.dtype != np.float64:
        X = X.astype(np.float64)

    tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
    alg = KDTreeBoruvkaAlgorithm(
        tree,
        min_samples,
        metric=metric,
        leaf_size=leaf_size // 3,
        approx_min_span_tree=approx_min_span_tree,
        n_jobs=core_dist_n_jobs,
        **kwargs
    )
    min_spanning_tree = alg.spanning_tree()
    # Sort edges of the min_spanning_tree by weight
    row_order = np.argsort(min_spanning_tree.T[2])
    min_spanning_tree = min_spanning_tree[row_order, :]
    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None


def _hdbscan_boruvka_balltree(
    X,
    min_samples=5,
    alpha=1.0,
    metric="minkowski",
    p=2,
    leaf_size=40,
    approx_min_span_tree=True,
    gen_min_span_tree=False,
    core_dist_n_jobs=4,
    **kwargs
):
    if leaf_size < 3:
        leaf_size = 3

    if core_dist_n_jobs < 1:
        core_dist_n_jobs = max(cpu_count() + 1 + core_dist_n_jobs, 1)

    if X.dtype != np.float64:
        X = X.astype(np.float64)

    tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
    alg = BallTreeBoruvkaAlgorithm(
        tree,
        min_samples,
        metric=metric,
        leaf_size=leaf_size // 3,
        approx_min_span_tree=approx_min_span_tree,
        n_jobs=core_dist_n_jobs,
        **kwargs
    )
    min_spanning_tree = alg.spanning_tree()
    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None


def check_precomputed_distance_matrix(X):
    """Perform check_array(X) after removing infinite values (numpy.inf) from the given distance matrix."""
    tmp = X.copy()
    tmp[np.isinf(tmp)] = 1
    check_array(tmp)


def remap_condensed_tree(tree, internal_to_raw, outliers):
    """
    Takes an internal condensed_tree structure and adds back in a set of points
    that were initially detected as non-finite and returns that new tree.
    These points will all be split off from the maximal node at lambda zero and
    considered noise points.

    Parameters
    ----------
    tree: condensed_tree
    internal_to_raw: dict
        a mapping from internal integer index to the raw integer index
    finite_index: ndarray
        Boolean array of which entries in the raw data were finite
    """
    finite_count = len(internal_to_raw)

    outlier_count = len(outliers)
    for i, (parent, child, lambda_val, child_size) in enumerate(tree):
        if child < finite_count:
            child = internal_to_raw[child]
        else:
            child = child + outlier_count
        tree[i] = (parent + outlier_count, child, lambda_val, child_size)

    outlier_list = []
    root = tree[0][0]  # Should I check to be sure this is the minimal lambda?
    for outlier in outliers:
        outlier_list.append((root, outlier, 0, 1))

    outlier_tree = np.array(
        outlier_list,
        dtype=[
            ("parent", np.intp),
            ("child", np.intp),
            ("lambda_val", float),
            ("child_size", np.intp),
        ],
    )
    tree = np.append(outlier_tree, tree)
    return tree


def remap_single_linkage_tree(tree, internal_to_raw, outliers):
    """
    Takes an internal single_linkage_tree structure and adds back in a set of points
    that were initially detected as non-finite and returns that new tree.
    These points will all be merged into the final node at np.inf distance and
    considered noise points.

    Parameters
    ----------
    tree: single_linkage_tree
    internal_to_raw: dict
        a mapping from internal integer index to the raw integer index
    finite_index: ndarray
        Boolean array of which entries in the raw data were finite
    """
    finite_count = len(internal_to_raw)

    outlier_count = len(outliers)
    for i, (left, right, distance, size) in enumerate(tree):
        if left < finite_count:
            tree[i, 0] = internal_to_raw[left]
        else:
            tree[i, 0] = left + outlier_count
        if right < finite_count:
            tree[i, 1] = internal_to_raw[right]
        else:
            tree[i, 1] = right + outlier_count

    outlier_tree = np.zeros((len(outliers), 4))
    last_cluster_id = tree[tree.shape[0] - 1][0:2].max()
    last_cluster_size = tree[tree.shape[0] - 1][3]
    for i, outlier in enumerate(outliers):
        outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)
        last_cluster_id += 1
        last_cluster_size += 1
    tree = np.vstack([tree, outlier_tree])
    return tree


def is_finite(matrix):
    """Returns true only if all the values of a ndarray or sparse matrix are finite"""
    if issparse(matrix):
        return np.all(np.isfinite(matrix.tocoo().data))
    else:
        return np.all(np.isfinite(matrix))


def get_finite_row_indices(matrix):
    """Returns the indices of the purely finite rows of a sparse matrix or dense ndarray"""
    if issparse(matrix):
        row_indices = np.array(
            [i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))]
        )
    else:
        row_indices = np.where(np.isfinite(matrix).sum(axis=1) == matrix.shape[1])[0]
    return row_indices


def hdbscan(
    X,
    min_cluster_size=5,
    min_samples=None,
    alpha=1.0,
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
    max_cluster_size=0,
    metric="minkowski",
    p=2,
    leaf_size=40,
    algorithm="best",
    memory=Memory(None, verbose=0),
    approx_min_span_tree=True,
    gen_min_span_tree=False,
    core_dist_n_jobs=4,
    cluster_selection_method="eom",
    allow_single_cluster=False,
    match_reference_implementation=False,
    cluster_selection_epsilon_max=float('inf'),
    **kwargs
):
    """Perform HDBSCAN clustering from a vector array or distance matrix.

    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    min_cluster_size : int, optional (default=5)
        The minimum number of samples in a group for that group to be
        considered a cluster; groupings smaller than this size will be left
        as noise.

    min_samples : int, optional (default=None)
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        defaults to the min_cluster_size.

    cluster_selection_epsilon: float, optional (default=0.0)
        A distance threshold. Clusters below this value will be merged.
        See [3]_ for more information. Note that this should not be used
        if we want to predict the cluster labels for new points in future
        (e.g. using approximate_predict), as the approximate_predict function
        is not aware of this argument. This is the minimum epsilon allowed.

    cluster_selection_persistence: float, optional (default=0.0)
        A persistence threshold. Clusters with a persistence lower than this
        value will be merged. Note that this should not be used if we want to
        predict the cluster labels for new points in future (e.g. using
        approximate_predict), as the approximate_predict function is not aware
        of this argument.

    alpha : float, optional (default=1.0)
        A distance scaling parameter as used in robust single linkage.
        See [2]_ for more information.

    max_cluster_size : int, optional (default=0)
        A limit to the size of clusters returned by the eom algorithm.
        Has no effect when using leaf clustering (where clusters are
        usually small regardless) and can also be overridden in rare
        cases by a high value for cluster_selection_epsilon. Note that
        this should not be used if we want to predict the cluster labels
        for new points in future (e.g. using approximate_predict), as
        the approximate_predict function is not aware of this argument.

    metric : string or callable, optional (default='minkowski')
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.

    p : int, optional (default=2)
        p value to use if using the minkowski metric.

    leaf_size : int, optional (default=40)
        Leaf size for trees responsible for fast nearest
        neighbor queries.

    algorithm : string, optional (default='best')
        Exactly which algorithm to use; hdbscan has variants specialized
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

    memory : instance of joblib.Memory or string, optional
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    approx_min_span_tree : bool, optional (default=True)
        Whether to accept an only approximate minimum spanning tree.
        For some algorithms this can provide a significant speedup, but
        the resulting clustering may be of marginally lower quality.
        If you are willing to sacrifice speed for correctness you may want
        to explore this; in general this should be left at the default True.

    gen_min_span_tree : bool, optional (default=False)
        Whether to generate the minimum spanning tree for later analysis.

    core_dist_n_jobs : int, optional (default=4)
        Number of parallel jobs to run in core distance computations (if
        supported by the specific algorithm). For ``core_dist_n_jobs``
        below -1, (n_cpus + 1 + core_dist_n_jobs) are used.

    cluster_selection_method : string, optional (default='eom')
        The method used to select clusters from the condensed tree. The
        standard approach for HDBSCAN* is to use an Excess of Mass algorithm
        to find the most persistent clusters. Alternatively you can instead
        select the clusters at the leaves of the tree -- this provides the
        most fine grained and homogeneous clusters. Options are:
            * ``eom``
            * ``leaf``

    allow_single_cluster : bool, optional (default=False)
        By default HDBSCAN* will not produce a single cluster, setting this
        to t=True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.
        (default False)

    match_reference_implementation : bool, optional (default=False)
        There exist some interpretational differences between this
        HDBSCAN* implementation and the original authors reference
        implementation in Java. This can result in very minor differences
        in clustering results. Setting this flag to True will, at a some
        performance cost, ensure that the clustering results match the
        reference implementation.

    cluster_selection_epsilon_max: float, optional (default=inf)
        A distance threshold. Clusters above this value will be split.
        Has no effect when using leaf clustering (where clusters are
        usually small regardless) and can also be overridden in rare
        cases by a high value for cluster_selection_epsilon. Note that
        this should not be used if we want to predict the cluster labels
        for new points in future (e.g. using approximate_predict), as
        the approximate_predict function is not aware of this argument.
        This is the maximum epsilon allowed.

    **kwargs : optional
        Arguments passed to the distance metric

    Returns
    -------
    labels : ndarray, shape (n_samples, )
        Cluster labels for each point.  Noisy samples are given the label -1.

    probabilities : ndarray, shape (n_samples, )
        Cluster membership strengths for each point. Noisy samples are assigned
        0.

    cluster_persistence : array, shape  (n_clusters, )
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores can be gauge the relative coherence of the clusters output
        by the algorithm.

    condensed_tree : record array
        The condensed cluster hierarchy used to generate clusters.

    single_linkage_tree : ndarray, shape (n_samples - 1, 4)
        The single linkage tree produced during clustering in scipy
        hierarchical clustering format
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).

    min_spanning_tree : ndarray, shape (n_samples - 1, 3)
        The minimum spanning as an edgelist. If gen_min_span_tree was False
        this will be None.

    References
    ----------

    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates.
       In Pacific-Asia Conference on Knowledge Discovery and Data Mining
       (pp. 160-172). Springer Berlin Heidelberg.

    .. [2] Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the
       cluster tree. In Advances in Neural Information Processing Systems
       (pp. 343-351).

    .. [3] Malzer, C., & Baum, M. (2019). A Hybrid Approach To Hierarchical 
	   Density-based Cluster Selection. arxiv preprint 1911.02282.
    """
    if min_samples is None:
        min_samples = min_cluster_size

    if not np.issubdtype(type(min_samples), np.integer) or \
       not np.issubdtype(type(min_cluster_size), np.integer):
        raise ValueError("Min samples and min cluster size must be integers!")

    if min_samples <= 0 or min_cluster_size <= 0:
        raise ValueError(
            "Min samples and Min cluster size must be positive" " integers"
        )

    if min_cluster_size == 1:
        raise ValueError("Min cluster size must be greater than one")

    if np.issubdtype(type(cluster_selection_epsilon), np.integer):
        cluster_selection_epsilon = float(cluster_selection_epsilon)

    if type(cluster_selection_epsilon) is not float or cluster_selection_epsilon < 0.0:
        raise ValueError("Epsilon must be a float value greater than or equal to 0!")

    if type(cluster_selection_persistence) is not float or cluster_selection_persistence < 0.0:
        raise ValueError("Persistence must be a float value greater than or equal to 0!")

    if not isinstance(alpha, float) or alpha <= 0.0:
        raise ValueError("Alpha must be a positive float value greater than" " 0!")

    if leaf_size < 1:
        raise ValueError("Leaf size must be greater than 0!")

    if metric == "minkowski":
        if p is None:
            raise TypeError("Minkowski metric given but no p value supplied!")
        if p < 0:
            raise ValueError(
                "Minkowski metric with negative p value is not" " defined!"
            )

    if cluster_selection_epsilon_max < cluster_selection_epsilon:
        raise ValueError("Cluster selection epsilon max must be greater than epsilon!")

    if match_reference_implementation:
        min_samples = min_samples - 1
        min_cluster_size = min_cluster_size + 1
        approx_min_span_tree = False

    if cluster_selection_method not in ("eom", "leaf"):
        raise ValueError(
            "Invalid Cluster Selection Method: %s\n" 'Should be one of: "eom", "leaf"\n'
        )

    # Checks input and converts to an nd-array where possible
    if metric != "precomputed" or issparse(X):
        X = check_array(X, accept_sparse="csr", force_all_finite=False)
    else:
        # Only non-sparse, precomputed distance matrices are handled here
        #   and thereby allowed to contain numpy.inf for missing distances
        check_precomputed_distance_matrix(X)

    # Python 2 and 3 compliant string_type checking
    if isinstance(memory, str):
        memory = Memory(memory, verbose=0)

    size = X.shape[0]
    min_samples = min(size - 1, min_samples)
    if min_samples == 0:
        min_samples = 1

    if algorithm != "best":
        if metric != "precomputed" and issparse(X) and algorithm != "generic":
            raise ValueError("Sparse data matrices only support algorithm 'generic'.")

        if algorithm == "generic":
            (single_linkage_tree, result_min_span_tree) = memory.cache(
                _hdbscan_generic
            )(X, min_samples, alpha, metric, p, leaf_size, gen_min_span_tree, **kwargs)
        elif algorithm == "prims_kdtree":
            if metric not in KDTREE_VALID_METRICS:
                raise ValueError("Cannot use Prim's with KDTree for this" " metric!")
            (single_linkage_tree, result_min_span_tree) = memory.cache(
                _hdbscan_prims_kdtree
            )(X, min_samples, alpha, metric, p, leaf_size, gen_min_span_tree, **kwargs)
        elif algorithm == "prims_balltree":
            if metric not in BALLTREE_VALID_METRICS:
                raise ValueError("Cannot use Prim's with BallTree for this" " metric!")
            (single_linkage_tree, result_min_span_tree) = memory.cache(
                _hdbscan_prims_balltree
            )(X, min_samples, alpha, metric, p, leaf_size, gen_min_span_tree, **kwargs)
        elif algorithm == "boruvka_kdtree":
            if metric not in BALLTREE_VALID_METRICS:
                raise ValueError("Cannot use Boruvka with KDTree for this" " metric!")
            (single_linkage_tree, result_min_span_tree) = memory.cache(
                _hdbscan_boruvka_kdtree
            )(
                X,
                min_samples,
                alpha,
                metric,
                p,
                leaf_size,
                approx_min_span_tree,
                gen_min_span_tree,
                core_dist_n_jobs,
                **kwargs
            )
        elif algorithm == "boruvka_balltree":
            if metric not in BALLTREE_VALID_METRICS:
                raise ValueError("Cannot use Boruvka with BallTree for this" " metric!")
            if (X.shape[0] // leaf_size) > 16000:
                warn(
                    "A large dataset size and small leaf_size may induce excessive "
                    "memory usage. If you are running out of memory consider "
                    "increasing the ``leaf_size`` parameter."
                )
            (single_linkage_tree, result_min_span_tree) = memory.cache(
                _hdbscan_boruvka_balltree
            )(
                X,
                min_samples,
                alpha,
                metric,
                p,
                leaf_size,
                approx_min_span_tree,
                gen_min_span_tree,
                core_dist_n_jobs,
                **kwargs
            )
        else:
            raise TypeError("Unknown algorithm type %s specified" % algorithm)
    else:

        if issparse(X) or metric not in FAST_METRICS:
            # We can't do much with sparse matrices ...
            (single_linkage_tree, result_min_span_tree) = memory.cache(
                _hdbscan_generic
            )(X, min_samples, alpha, metric, p, leaf_size, gen_min_span_tree, **kwargs)
        elif metric in KDTREE_VALID_METRICS:
            # TO DO: Need heuristic to decide when to go to boruvka;
            # still debugging for now
            if X.shape[1] > 60:
                (single_linkage_tree, result_min_span_tree) = memory.cache(
                    _hdbscan_prims_kdtree
                )(
                    X,
                    min_samples,
                    alpha,
                    metric,
                    p,
                    leaf_size,
                    gen_min_span_tree,
                    **kwargs
                )
            else:
                (single_linkage_tree, result_min_span_tree) = memory.cache(
                    _hdbscan_boruvka_kdtree
                )(
                    X,
                    min_samples,
                    alpha,
                    metric,
                    p,
                    leaf_size,
                    approx_min_span_tree,
                    gen_min_span_tree,
                    core_dist_n_jobs,
                    **kwargs
                )
        else:  # Metric is a valid BallTree metric
            # TO DO: Need heuristic to decide when to go to boruvka;
            # still debugging for now
            if X.shape[1] > 60:
                (single_linkage_tree, result_min_span_tree) = memory.cache(
                    _hdbscan_prims_balltree
                )(
                    X,
                    min_samples,
                    alpha,
                    metric,
                    p,
                    leaf_size,
                    gen_min_span_tree,
                    **kwargs
                )
            else:
                (single_linkage_tree, result_min_span_tree) = memory.cache(
                    _hdbscan_boruvka_balltree
                )(
                    X,
                    min_samples,
                    alpha,
                    metric,
                    p,
                    leaf_size,
                    approx_min_span_tree,
                    gen_min_span_tree,
                    core_dist_n_jobs,
                    **kwargs
                )

    return (
        _tree_to_labels(
            X,
            single_linkage_tree,
            min_cluster_size,
            cluster_selection_method,
            allow_single_cluster,
            match_reference_implementation,
            cluster_selection_epsilon,
            cluster_selection_persistence,
            max_cluster_size,
            cluster_selection_epsilon_max,
        )
        + (result_min_span_tree,)
    )


# Inherits from sklearn
class HDBSCAN(BaseEstimator, ClusterMixin):
    """Perform HDBSCAN clustering from vector array or distance matrix.

    HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
    with Noise. Performs DBSCAN over varying epsilon values and integrates
    the result to find a clustering that gives the best stability over epsilon.
    This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN),
    and be more robust to parameter selection.

    Parameters
    ----------
    min_cluster_size : int, optional (default=5)
        The minimum size of clusters; single linkage splits that contain
        fewer points than this will be considered points "falling out" of a
        cluster rather than a cluster splitting into two new clusters.

    min_samples : int, optional (default=None)
        The number of samples in a neighborhood for a point to be considered as a core point.
	This includes the point itself. When None, defaults to min_cluster_size.

    metric : string, or callable, optional (default='euclidean')
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.

    p : int, optional (default=None)
        p value to use if using the minkowski metric.

    alpha : float, optional (default=1.0)
        A distance scaling parameter as used in robust single linkage.
        See [3]_ for more information.

    cluster_selection_epsilon: float, optional (default=0.0)
        A distance threshold. Clusters below this value will be merged.
        This is the minimum epsilon allowed.
        See [5]_ for more information.

    cluster_selection_persistence: float, optional (default=0.0)
        A persistence threshold. Clusters with a persistence lower than this
        value will be merged.

    algorithm : string, optional (default='best')
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

    leaf_size: int, optional (default=40)
        If using a space tree algorithm (kdtree, or balltree) the number
        of points ina leaf node of the tree. This does not alter the
        resulting clustering, but may have an effect on the runtime
        of the algorithm.

    memory : Instance of joblib.Memory or string (optional)
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    approx_min_span_tree : bool, optional (default=True)
        Whether to accept an only approximate minimum spanning tree.
        For some algorithms this can provide a significant speedup, but
        the resulting clustering may be of marginally lower quality.
        If you are willing to sacrifice speed for correctness you may want
        to explore this; in general this should be left at the default True.

    gen_min_span_tree: bool, optional (default=False)
        Whether to generate the minimum spanning tree with regard
        to mutual reachability distance for later analysis.

    core_dist_n_jobs : int, optional (default=4)
        Number of parallel jobs to run in core distance computations (if
        supported by the specific algorithm). For ``core_dist_n_jobs``
        below -1, (n_cpus + 1 + core_dist_n_jobs) are used.

    cluster_selection_method : string, optional (default='eom')
        The method used to select clusters from the condensed tree. The
        standard approach for HDBSCAN* is to use an Excess of Mass algorithm
        to find the most persistent clusters. Alternatively you can instead
        select the clusters at the leaves of the tree -- this provides the
        most fine grained and homogeneous clusters. Options are:
            * ``eom``
            * ``leaf``

    allow_single_cluster : bool, optional (default=False)
        By default HDBSCAN* will not produce a single cluster, setting this
        to True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.

    prediction_data : boolean, optional
        Whether to generate extra cached data for predicting labels or
        membership vectors for new unseen points later. If you wish to
        persist the clustering object for later re-use you probably want
        to set this to True.
        (default False)

    branch_detection_data : boolean, optional
        Whether to generated extra cached data for detecting branch-
        hierarchies within clusters. If you wish to use functions from
        ``hdbscan.branches`` set this to True. (default False)

    match_reference_implementation : bool, optional (default=False)
        There exist some interpretational differences between this
        HDBSCAN* implementation and the original authors reference
        implementation in Java. This can result in very minor differences
        in clustering results. Setting this flag to True will, at a some
        performance cost, ensure that the clustering results match the
        reference implementation.

    cluster_selection_epsilon_max: float, optional (default=inf)
        A distance threshold. Clusters above this value will be split.
        Has no effect when using leaf clustering (where clusters are
        usually small regardless) and can also be overridden in rare
        cases by a high value for cluster_selection_epsilon. Note that
        this should not be used if we want to predict the cluster labels
        for new points in future (e.g. using approximate_predict), as
        the approximate_predict function is not aware of this argument.
        This is the maximum epsilon allowed.

    **kwargs : optional
        Arguments passed to the distance metric

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples, )
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    probabilities_ : ndarray, shape (n_samples, )
        The strength with which each sample is a member of its assigned
        cluster. Noise points have probability zero; points in clusters
        have values assigned proportional to the degree that they
        persist as part of the cluster.

    cluster_persistence_ : ndarray, shape (n_clusters, )
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores can be gauge the relative coherence of the clusters output
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

    outlier_scores_ : ndarray, shape (n_samples, )
        Outlier scores for clustered points; the larger the score the more
        outlier-like the point. Useful as an outlier detection technique.
        Based on the GLOSH algorithm by Campello, Moulavi, Zimek and Sander.

    prediction_data_ : PredictionData object
        Cached data used for predicting the cluster labels of new or
        unseen points. Necessary only if you are using functions from
        ``hdbscan.prediction`` (see
        :func:`~hdbscan.prediction.approximate_predict`,
        :func:`~hdbscan.prediction.membership_vector`,
        and :func:`~hdbscan.prediction.all_points_membership_vectors`).

    branch_detection_data_ : BranchDetectionData object
        Cached data used for detecting branch-hierarchies within clusters.
        Necessary only if you are using function from ``hdbscan.branches``.

    exemplars_ : list
        A list of exemplar points for clusters. Since HDBSCAN supports
        arbitrary shapes for clusters we cannot provide a single cluster
        exemplar per cluster. Instead a list is returned with each element
        of the list being a numpy array of exemplar points for a cluster --
        these points are the "most representative" points of the cluster.

    relative_validity_ : float
        A fast approximation of the Density Based Cluster Validity (DBCV)
        score [4]. The only difference, and the speed, comes from the fact
        that this relative_validity_ is computed using the mutual-
        reachability minimum spanning tree, i.e. minimum_spanning_tree_,
        instead of the all-points minimum spanning tree used in the
        reference. This score might not be an objective measure of the
        goodness of clustering. It may only be used to compare results
        across different choices of hyper-parameters, therefore is only a
        relative score.

    References
    ----------

    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates.
       In Pacific-Asia Conference on Knowledge Discovery and Data Mining
       (pp. 160-172). Springer Berlin Heidelberg.

    .. [2] Campello, R. J., Moulavi, D., Zimek, A., & Sander, J. (2015).
       Hierarchical density estimates for data clustering, visualization,
       and outlier detection. ACM Transactions on Knowledge Discovery
       from Data (TKDD), 10(1), 5.

    .. [3] Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the
       cluster tree. In Advances in Neural Information Processing Systems
       (pp. 343-351).

    .. [4] Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and
       Sander, J., 2014. Density-Based Clustering Validation. In SDM
       (pp. 839-847).

    .. [5] Malzer, C., & Baum, M. (2019). A Hybrid Approach To Hierarchical
           Density-based Cluster Selection. arxiv preprint 1911.02282.

    """

    def __init__(
        self,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        cluster_selection_persistence=0.0,
        max_cluster_size=0,
        metric="euclidean",
        alpha=1.0,
        p=None,
        algorithm="best",
        leaf_size=40,
        memory=Memory(None, verbose=0),
        approx_min_span_tree=True,
        gen_min_span_tree=False,
        core_dist_n_jobs=4,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        prediction_data=False,
        branch_detection_data=False,
        match_reference_implementation=False,
        cluster_selection_epsilon_max=float('inf'),
        **kwargs
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha
        self.max_cluster_size = max_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_persistence = cluster_selection_persistence
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory
        self.approx_min_span_tree = approx_min_span_tree
        self.gen_min_span_tree = gen_min_span_tree
        self.core_dist_n_jobs = core_dist_n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.match_reference_implementation = match_reference_implementation
        self.prediction_data = prediction_data
        self.branch_detection_data = branch_detection_data
        self.cluster_selection_epsilon_max = cluster_selection_epsilon_max

        self._metric_kwargs = kwargs

        self._condensed_tree = None
        self._single_linkage_tree = None
        self._min_spanning_tree = None
        self._raw_data = None
        self._outlier_scores = None
        self._prediction_data = None
        self._finite_index = None
        self._branch_detection_data = None
        self._relative_validity = None

    def fit(self, X, y=None):
        """Perform HDBSCAN clustering from features or distance matrix.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        self : object
            Returns self
        """
        if self.metric != "precomputed":
            # Non-precomputed matrices may contain non-finite values.
            # Rows with these values
            X = check_array(X, accept_sparse="csr", force_all_finite=False)
            self._raw_data = X

            self._all_finite = is_finite(X)
            if ~self._all_finite:
                # Pass only the purely finite indices into hdbscan
                # We will later assign all non-finite points to the background -1 cluster
                finite_index = get_finite_row_indices(X)
                clean_data = X[finite_index]
                internal_to_raw = {
                    x: y for x, y in zip(range(len(finite_index)), finite_index)
                }
                outliers = list(set(range(X.shape[0])) - set(finite_index))
            else:
                clean_data = X
        elif issparse(X):
            # Handle sparse precomputed distance matrices separately
            X = check_array(X, accept_sparse="csr")
            clean_data = X
        else:
            # Only non-sparse, precomputed distance matrices are allowed
            #   to have numpy.inf values indicating missing distances
            check_precomputed_distance_matrix(X)
            clean_data = X

        kwargs = self.get_params()
        # prediction data only applies to the persistent model, so remove
        # it from the keyword args we pass on the the function
        kwargs.pop("prediction_data", None)
        kwargs.pop("branch_detection_data", None)
        kwargs.update(self._metric_kwargs)
        kwargs['gen_min_span_tree'] |= self.branch_detection_data

        (
            self.labels_,
            self.probabilities_,
            self.cluster_persistence_,
            self._condensed_tree,
            self._single_linkage_tree,
            self._min_spanning_tree,
        ) = hdbscan(clean_data, **kwargs)

        if self.metric != "precomputed" and not self._all_finite:
            # remap indices to align with original data in the case of non-finite entries.
            self._condensed_tree = remap_condensed_tree(
                self._condensed_tree, internal_to_raw, outliers
            )
            self._single_linkage_tree = remap_single_linkage_tree(
                self._single_linkage_tree, internal_to_raw, outliers
            )
            new_labels = np.full(X.shape[0], -1)
            new_labels[finite_index] = self.labels_
            self.labels_ = new_labels

            new_probabilities = np.zeros(X.shape[0])
            new_probabilities[finite_index] = self.probabilities_
            self.probabilities_ = new_probabilities

        if self.prediction_data:
            self.generate_prediction_data()
        if self.branch_detection_data:
            self.generate_branch_detection_data()

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
        y : ndarray, shape (n_samples, )
            cluster labels
        """
        self.fit(X)
        return self.labels_

    def generate_prediction_data(self):
        """
        Create data that caches intermediate results used for predicting
        the label of new/unseen points. This data is only useful if
        you are intending to use functions from ``hdbscan.prediction``.
        """

        if self.metric in FAST_METRICS:
            min_samples = self.min_samples or self.min_cluster_size
            if self.metric in KDTREE_VALID_METRICS:
                tree_type = "kdtree"
            elif self.metric in BALLTREE_VALID_METRICS:
                tree_type = "balltree"
            else:
                warn("Metric {} not supported for prediction data!".format(self.metric))
                return

            self._prediction_data = PredictionData(
                self._raw_data,
                self.condensed_tree_,
                min_samples,
                tree_type=tree_type,
                metric=self.metric,
                **self._metric_kwargs
            )
        else:
            warn(
                "Cannot generate prediction data for non-vector "
                "space inputs -- access to the source data rather "
                "than mere distances is required!"
            )

    def generate_branch_detection_data(self):
        """
        Create data that caches intermediate results used for detecting
        branches within clusters. This data is only useful if you are
        intending to use functions from ``hdbscan.branches``.
        """
        if self.metric in FAST_METRICS:
            min_samples = self.min_samples or self.min_cluster_size
            if self.metric in KDTREE_VALID_METRICS:
                tree_type = "kdtree"
            elif self.metric in BALLTREE_VALID_METRICS:
                tree_type = "balltree"
            else:
                warn("Metric {} not supported for branch detection!".format(self.metric))
                return

            self._branch_detection_data = BranchDetectionData(
                self._raw_data,
                self.labels_,
                self._condensed_tree,
                min_samples,
                tree_type=tree_type,
                metric=self.metric,
                **self._metric_kwargs
            )
        else:
            warn(
                "Branch detection for non-vector space inputs is not (yet)"
                " implemented."
            )

    def weighted_cluster_centroid(self, cluster_id):
        """Provide an approximate representative point for a given cluster.
        Note that this technique assumes a euclidean metric for speed of
        computation. For more general metrics use the ``weighted_cluster_medoid``
        method which is slower, but can work with the metric the model trained
        with.

        Parameters
        ----------
        cluster_id: int
            The id of the cluster to compute a centroid for.

        Returns
        -------
        centroid: array of shape (n_features,)
            A representative centroid for cluster ``cluster_id``.
        """
        if not hasattr(self, "labels_"):
            raise AttributeError("Model has not been fit to data")

        if cluster_id == -1:
            raise ValueError(
                "Cannot calculate weighted centroid for -1 cluster "
                "since it is a noise cluster"
            )

        mask = self.labels_ == cluster_id
        cluster_data = self._raw_data[mask]
        cluster_membership_strengths = self.probabilities_[mask]

        return np.average(cluster_data, weights=cluster_membership_strengths, axis=0)

    def weighted_cluster_medoid(self, cluster_id):
        """Provide an approximate representative point for a given cluster.
        Note that this technique can be very slow and memory intensive for
        large clusters. For faster results use the ``weighted_cluster_centroid``
        method which is faster, but assumes a euclidean metric.

        Parameters
        ----------
        cluster_id: int
            The id of the cluster to compute a medoid for.

        Returns
        -------
        centroid: array of shape (n_features,)
            A representative medoid for cluster ``cluster_id``.
        """
        if not hasattr(self, "labels_"):
            raise AttributeError("Model has not been fit to data")

        if cluster_id == -1:
            raise ValueError(
                "Cannot calculate weighted centroid for -1 cluster "
                "since it is a noise cluster"
            )

        mask = self.labels_ == cluster_id
        cluster_data = self._raw_data[mask]
        cluster_membership_strengths = self.probabilities_[mask]

        dist_mat = pairwise_distances(
            cluster_data, metric=self.metric, **self._metric_kwargs
        )

        dist_mat = dist_mat * cluster_membership_strengths
        medoid_index = np.argmin(dist_mat.sum(axis=1))
        return cluster_data[medoid_index]

    def dbscan_clustering(self, cut_distance, min_cluster_size=5):
        """Return clustering that would be equivalent to running DBSCAN* for a particular cut_distance (or epsilon)
        DBSCAN* can be thought of as DBSCAN without the border points.  As such these results may differ slightly
        from sklearns implementation of dbscan in the non-core points.

        This can also be thought of as a flat clustering derived from constant height cut through the single
        linkage tree.

        This represents the result of selecting a cut value for robust single linkage
        clustering. The `min_cluster_size` allows the flat clustering to declare noise
        points (and cluster smaller than `min_cluster_size`).

        Parameters
        ----------

        cut_distance : float
            The mutual reachability distance cut value to use to generate a flat clustering.

        min_cluster_size : int, optional
            Clusters smaller than this value with be called 'noise' and remain unclustered
            in the resulting flat clustering.

        Returns
        -------

        labels : array [n_samples]
            An array of cluster labels, one per datapoint. Unclustered points are assigned
            the label -1.
        """
        return self.single_linkage_tree_.get_clusters(
            cut_distance=cut_distance,
            min_cluster_size=min_cluster_size,
        )

    @property
    def prediction_data_(self):
        if self._prediction_data is None:
            raise AttributeError("No prediction data was generated")
        else:
            return self._prediction_data
        
    @prediction_data_.setter
    def prediction_data_(self, value):
        self._prediction_data = value

    @property
    def branch_detection_data_(self):
        if self._branch_detection_data is None:
            raise AttributeError("No branch detection data was generated")
        else:
            return self._branch_detection_data
        
    @branch_detection_data_.setter
    def branch_detection_data_(self, value):
        self._branch_detection_data = value

    @property
    def outlier_scores_(self):
        if self._outlier_scores is not None:
            return self._outlier_scores
        else:
            if self._condensed_tree is not None:
                self._outlier_scores = outlier_scores(self._condensed_tree)
                return self._outlier_scores
            else:
                raise AttributeError(
                    "No condensed tree was generated; try running fit first."
                )
            
    @outlier_scores_.setter
    def outlier_scores_(self, value):
        self._outlier_scores = value

    @property
    def condensed_tree_(self):
        if self._condensed_tree is not None:
            return CondensedTree(
                self._condensed_tree,
                self.labels_
            )
        else:
            raise AttributeError(
                "No condensed tree was generated; try running fit first."
            )
        
    @condensed_tree_.setter
    def condensed_tree_(self, value):
        self._condensed_tree = value

    @property
    def single_linkage_tree_(self):
        if self._single_linkage_tree is not None:
            return SingleLinkageTree(self._single_linkage_tree)
        else:
            raise AttributeError(
                "No single linkage tree was generated; try running fit" " first."
            )
        
    @single_linkage_tree_.setter
    def single_linkage_tree_(self, value):
        self._single_linkage_tree = value

    @property
    def minimum_spanning_tree_(self):
        if self._min_spanning_tree is not None:
            if self._raw_data is not None:
                return MinimumSpanningTree(self._min_spanning_tree, self._raw_data)
            else:
                warn(
                    "No raw data is available; this may be due to using"
                    " a precomputed metric matrix. No minimum spanning"
                    " tree will be provided without raw data."
                )
                return None
        else:
            raise AttributeError(
                "No minimum spanning tree was generated."
                "This may be due to optimized algorithm variations that skip"
                " explicit generation of the spanning tree."
            )
    
    @minimum_spanning_tree_.setter
    def minimum_spanning_tree_(self, value):
        self._min_spanning_tree = value

    @property
    def exemplars_(self):
        if self._prediction_data is not None:
            return self._prediction_data.exemplars
        elif self.metric in FAST_METRICS:
            self.generate_prediction_data()
            return self._prediction_data.exemplars
        else:
            raise AttributeError(
                "Currently exemplars require the use of vector input data"
                "with a suitable metric. This will likely change in the "
                "future, but for now no exemplars can be provided"
            )

    @exemplars_.setter
    def exemplars_(self, value):
        self._exemplars = value

    @property
    def relative_validity_(self):
        if self._relative_validity is not None:
            return self._relative_validity

        if not self.gen_min_span_tree:
            raise AttributeError(
                "Minimum spanning tree not present. "
                + "Either HDBSCAN object was created with "
                + "gen_min_span_tree=False or the tree was "
                + "not generated in spite of it owing to "
                + "internal optimization criteria."
            )
            return

        labels = self.labels_
        sizes = np.bincount(labels + 1)
        noise_size = sizes[0]
        cluster_size = sizes[1:]
        total = noise_size + np.sum(cluster_size)
        num_clusters = len(cluster_size)
        DSC = np.zeros(num_clusters)
        min_outlier_sep = np.inf  # only required if num_clusters = 1
        correction_const = 2  # only required if num_clusters = 1

        # Unltimately, for each Ci, we only require the
        # minimum of DSPC(Ci, Cj) over all Cj != Ci.
        # So let's call this value DSPC_wrt(Ci), i.e.
        # density separation 'with respect to' Ci.
        DSPC_wrt = np.ones(num_clusters) * np.inf
        max_distance = 0

        mst_df = self.minimum_spanning_tree_.to_pandas()

        for edge in mst_df.iterrows():
            label1 = labels[int(edge[1]["from"])]
            label2 = labels[int(edge[1]["to"])]
            length = edge[1]["distance"]

            max_distance = max(max_distance, length)

            if label1 == -1 and label2 == -1:
                continue
            elif label1 == -1 or label2 == -1:
                # If exactly one of the points is noise
                min_outlier_sep = min(min_outlier_sep, length)
                continue

            if label1 == label2:
                # Set the density sparseness of the cluster
                # to the sparsest value seen so far.
                DSC[label1] = max(length, DSC[label1])
            else:
                # Check whether density separations with
                # respect to each of these clusters can
                # be reduced.
                DSPC_wrt[label1] = min(length, DSPC_wrt[label1])
                DSPC_wrt[label2] = min(length, DSPC_wrt[label2])

        # In case min_outlier_sep is still np.inf, we assign a new value to it.
        # This only makes sense if num_clusters = 1 since it has turned out
        # that the MR-MST has no edges between a noise point and a core point.
        min_outlier_sep = max_distance if min_outlier_sep == np.inf else min_outlier_sep

        # DSPC_wrt[Ci] might be infinite if the connected component for Ci is
        # an "island" in the MR-MST. Whereas for other clusters Cj and Ck, the
        # MR-MST might contain an edge with one point in Cj and the other one
        # in Ck. Here, we replace the infinite density separation of Ci by
        # another large enough value.
        #
        # TODO: Think of a better yet efficient way to handle this.
        correction = correction_const * (
            max_distance if num_clusters > 1 else min_outlier_sep
        )
        DSPC_wrt[np.where(DSPC_wrt == np.inf)] = correction

        V_index = [
            (DSPC_wrt[i] - DSC[i]) / max(DSPC_wrt[i], DSC[i])
            for i in range(num_clusters)
        ]
        score = np.sum(
            [(cluster_size[i] * V_index[i]) / total for i in range(num_clusters)]
        )
        self._relative_validity = score
        return self._relative_validity
    
    @relative_validity_.setter
    def relative_validity_(self, value):
        self._relative_validity = value
