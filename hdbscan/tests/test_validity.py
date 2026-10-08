import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.distance import pdist, squareform

from hdbscan.validity import all_points_core_distance, validity_index


@pytest.mark.parametrize("d", [2, 64, 200])
@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_all_points_core_distance_scaled_formula(d, scale):
    distances = np.array([[0., 2., 3.], [2., 0., 4.], [3., 4., 0.]])
    expected = np.array([
        ((2. ** -d + 3. ** -d) / 2) ** (-1. / d),
        ((2. ** -d + 4. ** -d) / 2) ** (-1. / d),
        ((3. ** -d + 4. ** -d) / 2) ** (-1. / d),
    ])

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        actual = all_points_core_distance(distances * scale, d=d)

    assert_allclose(actual, expected * scale, rtol=1e-12, atol=0)


@pytest.mark.parametrize("d", [2, 768])
@pytest.mark.parametrize("distance", [1e-300, 1.0, 1e100])
def test_all_points_core_distance_equal_distances(d, distance):
    distances = np.full((3, 3), distance)
    np.fill_diagonal(distances, 0)

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        actual = all_points_core_distance(distances, d=d)

    assert_allclose(actual, np.full(3, distance), rtol=1e-12, atol=0)


def test_all_points_core_distance_duplicate_points():
    distances = np.array([[0., 0., 2.], [0., 0., 2.], [2., 2., 0.]])
    actual = all_points_core_distance(distances, d=2)

    assert_allclose(actual, [2 * np.sqrt(2), 2 * np.sqrt(2), 2])
    assert_allclose(all_points_core_distance(np.zeros((3, 3))), np.zeros(3))


def test_all_points_core_distance_float32():
    distances = np.full((3, 3), 0.5, dtype=np.float32)
    np.fill_diagonal(distances, 0)

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        actual = all_points_core_distance(distances, d=200)

    assert actual.dtype == distances.dtype
    assert_allclose(actual, np.full(3, 0.5), rtol=1e-6, atol=0)


def test_all_points_core_distance_infinite_distances():
    distances = np.array([[0., 2., np.inf], [2., 0., np.inf], [np.inf, np.inf, 0.]])
    with np.errstate(divide="ignore"):
        actual = all_points_core_distance(distances, d=2)

    assert_allclose(actual, [2 * np.sqrt(2), 2 * np.sqrt(2), np.inf])
    assert_allclose(
        all_points_core_distance(np.array([[0., np.inf], [np.inf, 0.]])), np.zeros(2)
    )


@pytest.mark.parametrize("metric", ["euclidean", "precomputed"])
@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_validity_index_scale_invariance(metric, scale):
    d = 768
    positions = np.array([0., .5, 1., 1.5, 2., 15., 15.5, 16., 16.5, 17.])
    labels = np.repeat([0, 1], 5)
    if metric == "precomputed":
        data = squareform(pdist(positions[:, None]))
    else:
        data = np.repeat(positions[:, None], d, axis=1) / np.sqrt(d)

    expected, expected_clusters = validity_index(
        data, labels, metric=metric, d=d, per_cluster_scores=True
    )
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        actual, actual_clusters = validity_index(
            data * scale, labels, metric=metric, d=d, per_cluster_scores=True
        )

    assert np.isfinite(actual)
    assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert_allclose(actual_clusters, expected_clusters, rtol=1e-10, atol=1e-12)
