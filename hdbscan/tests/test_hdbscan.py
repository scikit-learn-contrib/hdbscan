"""
Tests for HDBSCAN clustering algorithm
Shamelessly based on (i.e. ripped off from) the DBSCAN test code
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
from hdbscan import HDBSCAN
from hdbscan import hdbscan
from sklearn.cluster.tests.common import generate_clustered_data
from scipy.stats import mode

from sklearn import datasets

n_clusters = 3
X = generate_clustered_data(n_clusters=n_clusters, n_samples_per_cluster=50)

def relabel(labels):
    result = np.zeros(labels.shape[0])
    labels_to_go = set(labels)
    i = 0
    new_l = 0
    while len(labels_to_go) > 0:
        l = labels[i]
        if l in labels_to_go:
            result[labels == l] = new_l
            new_l += 1
            labels_to_go.remove(l)
        i += 1
    return result

def generate_noisy_data():
    blobs, _ = datasets.make_blobs(n_samples=200,
                                    centers=[(-0.75,2.25), (1.0, 2.0)],
                                    cluster_std=0.25)
    moons, _ = datasets.make_moons(n_samples=200, noise=0.05)
    noise = np.random.uniform(-1.0, 3.0, (50, 2))
    return np.vstack([blobs, moons, noise])

def homogeneity(labels1, labels2):
    num_missed = 0.0
    for label in set(labels1):
        matches = labels2[labels1 == label]
        match_mode = mode(matches).mode[0]
        num_missed += np.sum(matches != match_mode)

    for label in set(labels2):
        matches = labels1[labels2 == label]
        match_mode = mode(matches).mode[0]
        num_missed += np.sum(matches != match_mode)

    return num_missed / 2.0

def test_hdbscan_distance_matrix():
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)
    
    labels, p, ctree, ltree, mtree = hdbscan(D, metric='precomputed')
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels) # ignore noise
    assert_equal(n_clusters_1, n_clusters)

    labels = HDBSCAN(metric="precomputed").fit(D).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)
    
def test_hdbscan_feature_vector():    
    labels, p, ctree, ltree, mtree = hdbscan(X)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, n_clusters)

    labels = HDBSCAN().fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)

def test_hdbscan_no_clusters():
    labels, p, ctree, ltree, mtree = hdbscan(X, min_cluster_size=len(X)+1)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, 0)
    
    labels = HDBSCAN(min_cluster_size=len(X)+1).fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, 0)
    
def test_hdbscan_callable_metric():
    # metric is the function reference, not the string key.
    metric = distance.euclidean

    labels, p, ctree, ltree, mtree = hdbscan(X, metric=metric)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, n_clusters)

    labels = HDBSCAN(metric=metric).fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)

def test_hdbscan_input_lists():
    X = [[1., 2.], [3., 4.]]
    HDBSCAN().fit(X)  # must not raise exception

def test_hdbscan_boruvka_kdtree_matches():

    data = generate_noisy_data()

    labels_prims, p, ctree, ltree, mtree = hdbscan(data, algorithm='generic')
    labels_boruvka, p, ctree, ltree, mtree = hdbscan(data, algorithm='boruvka_kdtree')

    num_mismatches = homogeneity(labels_prims,  labels_boruvka)

    assert_less(num_mismatches / float(data.shape[0]), 0.015)

    labels_prims = HDBSCAN(algorithm='generic').fit_predict(data)
    labels_boruvka = HDBSCAN(algorithm='boruvka_kdtree').fit_predict(data)

    num_mismatches = homogeneity(labels_prims,  labels_boruvka)

    assert_less(num_mismatches / float(data.shape[0]), 0.015)

def test_hdbscan_boruvka_balltree_matches():

    data = generate_noisy_data()

    labels_prims, p, ctree, ltree, mtree = hdbscan(data, algorithm='generic')
    labels_boruvka, p, ctree, ltree, mtree = hdbscan(data, algorithm='boruvka_balltree')

    num_mismatches = homogeneity(labels_prims,  labels_boruvka)

    assert_less(num_mismatches / float(data.shape[0]), 0.015)

    labels_prims = HDBSCAN(algorithm='generic').fit_predict(data)
    labels_boruvka = HDBSCAN(algorithm='boruvka_balltree').fit_predict(data)

    num_mismatches = homogeneity(labels_prims,  labels_boruvka)

    assert_less(num_mismatches / float(data.shape[0]), 0.015)


def test_hdbscan_badargs():
    assert_raises(ValueError,
                  hdbscan,
                  X='fail')
    assert_raises(ValueError,
                  hdbscan,
                  X=None)
    assert_raises(ValueError,
                  hdbscan,
                  X, min_cluster_size='fail')
    assert_raises(ValueError,
                  hdbscan,
                  X, min_samples='fail')
    assert_raises(ValueError,
                  hdbscan,
                  X, min_samples=-1)
    assert_raises(ValueError,
                  hdbscan,
                  X, metric='imperial')
    assert_raises(ValueError,
                  hdbscan,
                  X, metric=None)
    assert_raises(ValueError,
                  hdbscan,
                  X, metric='minkowski', p=-1)
    
    
### Probably not applicable now #########################
#def test_dbscan_sparse():
#def test_dbscan_balltree():
#def test_pickle():
#def test_dbscan_core_samples_toy():
#def test_boundaries():

