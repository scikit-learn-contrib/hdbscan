# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: nonecheck=False
# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes, Steve Astels
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX

from hdbscan.dist_metrics cimport DistanceMetric


cpdef np.ndarray[np.double_t, ndim=2] mst_linkage_core(
                               np.ndarray[np.double_t,
                                          ndim=2] distance_matrix):

    cdef np.intp_t dim = distance_matrix.shape[0]
    cdef np.ndarray[np.double_t, ndim=2] result_arr = np.zeros((dim - 1, 3))
    cdef np.ndarray[np.int8_t, ndim=1] in_tree_arr = np.zeros(dim, dtype=np.int8)
    cdef np.ndarray[np.double_t, ndim=1] current_distances_arr = np.full(dim, DBL_MAX)
    cdef np.ndarray[np.intp_t, ndim=1] current_sources_arr = np.zeros(dim, dtype=np.intp)

    cdef np.double_t *current_distances = <np.double_t *> current_distances_arr.data
    cdef np.intp_t *current_sources = <np.intp_t *> current_sources_arr.data
    cdef np.int8_t *in_tree = <np.int8_t *> in_tree_arr.data
    cdef np.double_t[:, ::1] result = (<np.double_t[:dim - 1, :3:1]> (
        <np.double_t *> result_arr.data))
    cdef np.double_t[:, ::1] dist_view = distance_matrix

    cdef np.intp_t current_node = 0
    cdef np.intp_t source_node
    cdef np.intp_t new_node
    cdef np.intp_t i, j
    cdef np.double_t new_distance, d

    for i in range(1, dim):
        in_tree[current_node] = 1
        new_distance = DBL_MAX
        new_node = 0
        source_node = 0

        for j in range(dim):
            if in_tree[j]:
                continue

            d = dist_view[current_node, j]
            if d < current_distances[j]:
                current_distances[j] = d
                current_sources[j] = current_node

            if current_distances[j] < new_distance:
                new_distance = current_distances[j]
                source_node = current_sources[j]
                new_node = j

        result[i - 1, 0] = <double> source_node
        result[i - 1, 1] = <double> new_node
        result[i - 1, 2] = new_distance
        current_node = new_node

    return result_arr


cpdef np.ndarray[np.double_t, ndim=2] mst_linkage_core_vector(
        np.ndarray[np.double_t, ndim=2, mode='c'] raw_data,
        np.ndarray[np.double_t, ndim=1, mode='c'] core_distances,
        DistanceMetric dist_metric,
        np.double_t alpha=1.0):

    # Add a comment
    cdef np.ndarray[np.double_t, ndim=1] current_distances_arr
    cdef np.ndarray[np.double_t, ndim=1] current_sources_arr
    cdef np.ndarray[np.int8_t, ndim=1] in_tree_arr
    cdef np.ndarray[np.double_t, ndim=2] result_arr

    cdef np.double_t * current_distances
    cdef np.double_t * current_sources
    cdef np.double_t * current_core_distances
    cdef np.double_t * raw_data_ptr
    cdef np.int8_t * in_tree
    cdef np.double_t[:, ::1] raw_data_view
    cdef np.double_t[:, ::1] result

    cdef np.ndarray label_filter

    cdef np.intp_t current_node
    cdef np.intp_t source_node
    cdef np.intp_t right_node, right_source
    cdef np.intp_t left_node, left_source
    cdef np.intp_t new_node
    cdef np.intp_t i
    cdef np.intp_t j
    cdef np.intp_t dim
    cdef np.intp_t num_features

    cdef double current_node_core_distance
    cdef double right_value
    cdef double left_value
    cdef double core_value
    cdef double new_distance

    dim = raw_data.shape[0]
    num_features = raw_data.shape[1]

    raw_data_view = (<np.double_t[:raw_data.shape[0], :raw_data.shape[1]:1]> (
        <np.double_t *> raw_data.data))
    raw_data_ptr = (<np.double_t *> &raw_data_view[0, 0])

    result_arr = np.zeros((dim - 1, 3))
    in_tree_arr = np.zeros(dim, dtype=np.int8)
    current_node = 0
    current_distances_arr = np.inf * np.ones(dim)
    current_sources_arr = np.ones(dim)

    result = (<np.double_t[:dim - 1, :3:1]> (<np.double_t *> result_arr.data))
    in_tree = (<np.int8_t *> in_tree_arr.data)
    current_distances = (<np.double_t *> current_distances_arr.data)
    current_sources = (<np.double_t *> current_sources_arr.data)
    current_core_distances = (<np.double_t *> core_distances.data)

    for i in range(1, dim):

        in_tree[current_node] = 1

        current_node_core_distance = current_core_distances[current_node]

        new_distance = DBL_MAX
        source_node = 0
        new_node = 0

        for j in range(dim):
            if in_tree[j]:
                continue

            right_value = current_distances[j]
            right_source = <np.intp_t> current_sources[j]

            left_value = dist_metric.dist(&raw_data_ptr[num_features *
                                                        current_node],
                                          &raw_data_ptr[num_features * j],
                                          num_features)
            left_source = current_node

            if alpha != 1.0:
                left_value /= alpha

            core_value = core_distances[j]
            if (current_node_core_distance > right_value or
                    core_value > right_value or
                    left_value > right_value):
                if right_value < new_distance:
                    new_distance = right_value
                    source_node = right_source
                    new_node = j
                continue

            if core_value > current_node_core_distance:
                if core_value > left_value:
                    left_value = core_value
            else:
                if current_node_core_distance > left_value:
                    left_value = current_node_core_distance

            if left_value < right_value:
                current_distances[j] = left_value
                current_sources[j] = left_source
                if left_value < new_distance:
                    new_distance = left_value
                    source_node = left_source
                    new_node = j
            else:
                if right_value < new_distance:
                    new_distance = right_value
                    source_node = right_source
                    new_node = j

        result[i - 1, 0] = <double> source_node
        result[i - 1, 1] = <double> new_node
        result[i - 1, 2] = new_distance
        current_node = new_node

    return result_arr


cdef class UnionFind (object):

    cdef np.ndarray parent_arr
    cdef np.ndarray size_arr
    cdef np.intp_t next_label
    cdef np.intp_t *parent
    cdef np.intp_t *size

    def __init__(self, N):
        self.parent_arr = -1 * np.ones(2 * N - 1, dtype=np.intp, order='C')
        self.next_label = N
        self.size_arr = np.hstack((np.ones(N, dtype=np.intp),
                                   np.zeros(N-1, dtype=np.intp)))
        self.parent = (<np.intp_t *> self.parent_arr.data)
        self.size = (<np.intp_t *> self.size_arr.data)

    cdef void union(self, np.intp_t m, np.intp_t n):
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.next_label += 1
        return

    cdef np.intp_t fast_find(self, np.intp_t n):
        cdef np.intp_t p, tmp
        p = n
        while self.parent[n] != -1:
            n = self.parent[n]
        # label up to the root if this is not the root already
        if p != n:
            while self.parent[p] != n:
                tmp = self.parent[p]
                self.parent[p] = n
                p = tmp
        return n


cpdef np.ndarray[np.double_t, ndim=2] label(np.ndarray[np.double_t, ndim=2] L):

    cdef np.ndarray[np.double_t, ndim=2] result_arr
    cdef np.double_t[:, ::1] result

    cdef np.intp_t N, a, aa, b, bb, index
    cdef np.double_t delta

    result_arr = np.zeros((L.shape[0], L.shape[1] + 1))
    result = (<np.double_t[:L.shape[0], :4:1]> (
        <np.double_t *> result_arr.data))
    N = L.shape[0] + 1
    U = UnionFind(N)

    for index in range(L.shape[0]):

        a = <np.intp_t> L[index, 0]
        b = <np.intp_t> L[index, 1]
        delta = L[index, 2]

        aa, bb = U.fast_find(a), U.fast_find(b)

        result[index][0] = aa
        result[index][1] = bb
        result[index][2] = delta
        result[index][3] = U.size[aa] + U.size[bb]

        U.union(aa, bb)

    return result_arr


cpdef np.ndarray[np.double_t, ndim=2] single_linkage(distance_matrix):

    cdef np.ndarray[np.double_t, ndim=2] hierarchy
    cdef np.ndarray[np.double_t, ndim=2] for_labelling

    hierarchy = mst_linkage_core(distance_matrix)
    for_labelling = hierarchy[np.argsort(hierarchy.T[2]), :]

    return label(for_labelling)
