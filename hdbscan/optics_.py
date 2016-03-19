# -*- coding: utf-8 -*-
"""
OPTICS: Ordering Points To Identify Clusters
"""
# Author: Leland McInnes <leland.mcinnes@gmail.com>
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
                               mst_linkage_core_cdist,
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
