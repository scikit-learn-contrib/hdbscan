"""
Tests for Robust Single Linkage clustering algorithm
"""
#import pickle
from nose.tools import assert_less
import numpy as np
from scipy.spatial import distance
from scipy import sparse
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_in
from sklearn.utils.testing import assert_not_in
from sklearn.utils.testing import assert_no_warnings
from sklearn.utils.testing import if_matplotlib
from hdbscan import RobustSingleLinkage
from hdbscan import robust_single_linkage
from sklearn.cluster.tests.common import generate_clustered_data

from sklearn import datasets

n_clusters = 3
X = generate_clustered_data(n_clusters=n_clusters, n_samples_per_cluster=50)

def test_rsl_distance_matrix():
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)

    labels, tree = robust_single_linkage(D, 0.25, metric='precomputed')
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels) # ignore noise
    #assert_equal(n_clusters_1, n_clusters)

    labels = RobustSingleLinkage(metric="precomputed").fit(D).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    #assert_equal(n_clusters_2, n_clusters)

def test_rsl_feature_vector():
    labels, tree = robust_single_linkage(X, 0.2)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    #assert_equal(n_clusters_1, n_clusters)

    labels = RobustSingleLinkage().fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    #assert_equal(n_clusters_2, n_clusters)

def test_rsl_callable_metric():
    # metric is the function reference, not the string key.
    metric = distance.euclidean

    labels, tree = robust_single_linkage(X, 0.2, metric=metric)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    #assert_equal(n_clusters_1, n_clusters)

    labels = RobustSingleLinkage(metric=metric).fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    #assert_equal(n_clusters_2, n_clusters)

def test_rsl_input_lists():
    X = [[1., 2.], [3., 4.]]
    RobustSingleLinkage().fit(X)  # must not raise exception
