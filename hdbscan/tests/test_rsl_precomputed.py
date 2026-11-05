import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.spatial.distance import pdist, squareform
from hdbscan import RobustSingleLinkage, robust_single_linkage


@pytest.mark.parametrize('readonly', [False, True])
@pytest.mark.parametrize('estimator', [False, True])
def test_rsl_preserves_precomputed_distances(readonly, estimator):
    distances = squareform(pdist(np.array([[0.], [1.], [4.], [5.], [10.]])))
    original = distances.copy()
    if readonly:
        distances.setflags(write=False)
    kwargs = dict(k=2, alpha=1., metric='precomputed', algorithm='generic')
    if estimator:
        RobustSingleLinkage(cut=2., **kwargs).fit(distances)
    else:
        robust_single_linkage(distances, cut=2., **kwargs)
    assert_array_equal(distances, original)


@pytest.mark.parametrize('distances', [np.ones((3, 2)), -np.eye(3)])
def test_rsl_precomputed_validation(distances):
    with pytest.raises(ValueError):
        robust_single_linkage(distances, cut=2., metric='precomputed')
