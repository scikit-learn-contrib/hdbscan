import numpy as np
import pytest
from numpy.testing import assert_array_equal
from hdbscan import robust_single_linkage


@pytest.mark.parametrize('algorithm', [
    'generic', 'prims_kdtree', 'prims_balltree',
    'boruvka_kdtree', 'boruvka_balltree', 'best',
])
@pytest.mark.parametrize('cut', [0.75, 1.1, 4.0])
def test_rsl_edge_order(algorithm, cut):
    # The tree traversals discover edges in a different order from their weights.
    data = np.array([[0.], [4.], [5.], [1.], [20.], [20.5]])
    labels, tree = robust_single_linkage(
        data, cut=cut, k=1, alpha=1., gamma=1, algorithm=algorithm,
    )
    assert np.all(np.diff(tree[:, 2]) >= 0)

    # For k=1 and alpha=1 the reachability graph is the distance graph.
    # These are its connected components at each cut, independent of label IDs.
    expected = {
        0.75: [0, 1, 2, 3, 4, 4],
        1.1: [0, 1, 1, 0, 2, 2],
        4.0: [0, 0, 0, 0, 1, 1],
    }[cut]
    expected = np.asarray(expected)
    assert_array_equal(labels[:, None] == labels, expected[:, None] == expected)
