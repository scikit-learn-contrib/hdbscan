#cython: boundscheck=False, nonecheck=False, wraparound=False
# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes
# License: 3-clause BSD

cimport cython

import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX

# from scipy.spatial.distance import cdist, pdist, squareform
import sklearn.neighbors.dist_metrics as dist_metrics

cdef struct NodeData_t:
    np.int64_t idx_start
    np.int64_t idx_end
    np.int64_t is_leaf
    np.double_t radius

cdef inline double min_dist_dual(np.double_t radius1,
                                 np.double_t radius2,
                                 long long node1,
                                 long long node2,
                                 np.double_t[:, ::1] centroid_dist):
    cdef np.double_t dist_pt = centroid_dist[node1, node2]
    return max(0, (dist_pt - radius1 - radius2))

cdef class BoruvkaUnionFind (object):

    cdef np.ndarray _data_arr
    cdef np.int64_t[:,::1] _data

    def __init__(self, size):
        self._data_arr = np.zeros((size, 2), dtype=np.int64)
        self._data_arr.T[0] = np.arange(size)
        self._data = (<np.int64_t[:size, :2:1]> (<np.int64_t *> self._data_arr.data))

    cpdef union_(self, long long x, long long y):
        cdef long long x_root = self.find(x)
        cdef long long y_root = self.find(y)

        if self._data[x_root, 1] < self._data[y_root, 1]:
            self._data[x_root, 0] = y_root
        elif self._data[x_root, 1] > self._data[y_root, 1]:
            self._data[y_root, 0] = x_root
        else:
            self._data[y_root, 0] = x_root
            self._data[x_root, 1] += 1

        return

    cpdef find(self, long long x):
        if self._data[x, 0] != x:
            self._data[x, 0] = self.find(self._data[x, 0])
        return self._data[x, 0]

    cpdef np.ndarray[np.int64_t, ndim=1] components(self):
        return self._data_arr.T[0]

cdef class BoruvkaAlgorithm (object):

    cdef object tree
    cdef object dist
    cdef np.ndarray _data
    cdef np.int64_t min_samples
    cdef public np.double_t[::1] core_distance
    cdef public np.double_t[::1] bounds
    cdef public np.int64_t[::1] component_of_point
    cdef public np.int64_t[::1] component_of_node
    cdef public np.int64_t[::1] candidate_neighbor
    cdef public np.int64_t[::1] candidate_point
    cdef public np.double_t[::1] candidate_distance
    cdef object component_union_find
    cdef set edges

    cdef np.ndarray components
    cdef np.ndarray core_distance_arr
    cdef np.ndarray bounds_arr
    cdef np.ndarray _centroid_distances
    cdef np.ndarray component_of_point_arr
    cdef np.ndarray component_of_node_arr
    cdef np.ndarray candidate_point_arr
    cdef np.ndarray candidate_neighbor_arr
    cdef np.ndarray candidate_distance_arr

    def __init__(self, tree, min_samples=5, metric='euclidean', **kwargs):

        cdef np.int64_t num_points = tree.data.shape[0]
        cdef np.int64_t num_nodes = tree.node_data.shape[0]

        self.tree = tree
        self._data = np.array(tree.data)
        self.min_samples = min_samples

        self.dist = dist_metrics.DistanceMetric.get_metric(metric, **kwargs)

        self.components = np.arange(num_points)
        self.bounds_arr = np.empty(num_nodes, np.double)
        self.component_of_point_arr = np.empty(num_points, dtype=np.int64)
        self.component_of_node_arr = np.empty(num_nodes, dtype=np.int64)
        self.candidate_neighbor_arr = np.empty(num_points, dtype=np.int64)
        self.candidate_point_arr = np.empty(num_points, dtype=np.int64)
        self.candidate_distance_arr = np.empty(num_points, dtype=np.double)
        self.component_union_find = BoruvkaUnionFind(num_points)
        self.edges = set([])

        self.bounds = (<np.double_t[:num_nodes:1]> (<np.double_t *> self.bounds_arr.data))
        self.component_of_point = (<np.int64_t[:num_points:1]> (<np.int64_t *> self.component_of_point_arr.data))
        self.component_of_node = (<np.int64_t[:num_nodes:1]> (<np.int64_t *> self.component_of_node_arr.data))
        self.candidate_neighbor = (<np.int64_t[:num_points:1]> (<np.int64_t *> self.candidate_neighbor_arr.data))
        self.candidate_point = (<np.int64_t[:num_points:1]> (<np.int64_t *> self.candidate_point_arr.data))
        self.candidate_distance = (<np.double_t[:num_points:1]> (<np.double_t *> self.candidate_distance_arr.data))

        self._centroid_distances = self.dist.pairwise(tree.node_bounds[0])

        self._compute_bounds()
        self._initialize_components()

    cdef _compute_bounds(self):

        cdef np.int64_t n
        cdef np.int64_t num_points = self.tree.data.shape[0]
        cdef np.int64_t num_nodes = self.tree.node_data.shape[0]

        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.double_t, ndim=1] nn_dist
        cdef np.int64_t[::1] point_indices
        cdef np.int64_t[::1] idx_array = self.tree.idx_array

        cdef np.double_t b1
        cdef np.double_t b2

        cdef np.int64_t child1
        cdef np.int64_t child2

        cdef NodeData_t node_info
        cdef NodeData_t child1_info
        cdef NodeData_t child2_info

        knn_dist = self.tree.query(self.tree.data, max(2, self.min_samples))[0]
        nn_dist = knn_dist[:, 1]
        self.core_distance_arr = knn_dist[:, self.min_samples - 1].copy()
        self.core_distance = (<np.double_t [:num_points:1]> (<np.double_t *> self.core_distance_arr.data))

        for n in range(num_nodes - 1, -1, -1):
            node_info = self.tree.node_data[n]
            if node_info.is_leaf:
                point_indices = idx_array[node_info.idx_start:node_info.idx_end]
                b1 = nn_dist[point_indices].max()
                # b1 = self.core_distance_arr[point_indices].max()
                b2 = (nn_dist[point_indices] + 2 * node_info.radius).min()
                # b2 = (self.core_distance_arr[point_indices] + 2 * node_info.radius).min()
                self.bounds[n] = min(b1, b2)
            else:
                child1 = 2 * n + 1
                child2 = 2 * n + 2
                child1_info = self.tree.node_data[child1]
                child2_info = self.tree.node_data[child2]
                b1 = max(self.bounds[child1], self.bounds[child2])
                b2 = min(self.bounds[child1] + 2 * (node_info.radius - child1_info.radius),
                            self.bounds[child2] + 2 * (node_info.radius - child2_info.radius))
                if b2 > 0:
                    self.bounds[n] = min(b1, b2)
                else:
                    self.bounds[n] = b1

        for n in range(1, num_nodes):
            self.bounds[n] = min(self.bounds[n], self.bounds[(n - 1) // 2])

    cdef _initialize_components(self):

        cdef np.int64_t n
        cdef np.int64_t num_points = self.tree.data.shape[0]
        cdef np.int64_t num_nodes = self.tree.node_data.shape[0]

        for n in range(num_points):
            self.component_of_point[n] = n
            self.candidate_neighbor[n] = -1
            self.candidate_point[n] = -1
            self.candidate_distance[n] = DBL_MAX

        for n in range(num_nodes):
            self.component_of_node[n] = -(n+1)

    cdef update_components(self):

        cdef np.int64_t source
        cdef np.int64_t sink
        cdef np.int64_t c
        cdef np.int64_t component
        cdef np.int64_t n
        cdef np.int64_t i
        cdef np.int64_t p
        cdef np.int64_t current_component
        cdef np.int64_t child1
        cdef np.int64_t child2

        cdef NodeData_t node_info
        cdef np.int64_t[::1] idx_array = self.tree.idx_array

        for c in range(self.components.shape[0]):
            component = self.components[c]
            source, sink = sorted([self.candidate_point[component],
                                   self.candidate_neighbor[component]])
            if source == -1 or sink == -1:
                raise ValueError('Source or sink of edge is None!')
            self.edges.add((source, sink, self.candidate_distance[component]))
            self.component_union_find.union_(source, sink)
            self.candidate_distance[component] = DBL_MAX

        for n in range(self.tree.data.shape[0]):
            self.component_of_point[n] = self.component_union_find.find(n)

        for n in range(self.tree.node_data.shape[0] - 1, -1, -1):
            node_info = self.tree.node_data[n]
            if node_info.is_leaf:
                current_component = self.component_of_point[idx_array[node_info.idx_start]]
                for i in range(node_info.idx_start + 1, node_info.idx_end):
                    p = idx_array[i]
                    if self.component_of_point[p] != current_component:
                        break
                else:
                    self.component_of_node[n] = current_component
            else:
                child1 = 2 * n + 1
                child2 = 2 * n + 2
                if self.component_of_node[child1] == self.component_of_node[child2]:
                    self.component_of_node[n] = self.component_of_node[child1]

        self.components = np.unique(self.component_union_find.components())
        return self.components.shape[0]

    cpdef int dual_tree_traversal(self, np.int64_t node1, np.int64_t node2):

        cdef np.ndarray[np.double_t, ndim=2] distances_arr
        cdef np.double_t[:,::1] distances

        cdef np.ndarray[np.double_t, ndim=2] points1
        cdef np.ndarray[np.double_t, ndim=2] points2
        cdef np.int64_t[::1] point_indices1, point_indices2

        cdef long long i
        cdef long long j

        cdef long long p
        cdef long long q

        cdef long long child1
        cdef long long child2

        cdef double node_dist

        cdef np.int64_t num_points = self.tree.data.shape[0]
        cdef np.int64_t num_nodes = self.tree.node_data.shape[0]
        cdef np.int64_t[::1] idx_array = self.tree.idx_array

        cdef NodeData_t node1_info = self.tree.node_data[node1]
        cdef NodeData_t node2_info = self.tree.node_data[node2]

        cdef np.int64_t *component_of_point_ptr = <np.int64_t *> &self.component_of_point[0]
        cdef np.double_t *candidate_distance_ptr = <np.double_t *> &self.candidate_distance[0]

        cdef np.int64_t component1
        cdef np.int64_t component2

        cdef np.double_t mr_dist

        node_dist = min_dist_dual(node1_info.radius, node2_info.radius,
                                    node1, node2, (<np.double_t [:num_nodes, :num_nodes:1]>
                                                    (<np.double_t *> self._centroid_distances.data)))

        if node_dist < self.bounds[node1]:
            if self.component_of_node[node1] == self.component_of_node[node2] and \
                    self.component_of_node[node1] >= 0:
                return 0
        else:
            return 0


        if node1_info.is_leaf and node2_info.is_leaf:

            point_indices1 = idx_array[node1_info.idx_start:node1_info.idx_end]
            point_indices2 = idx_array[node2_info.idx_start:node2_info.idx_end]

            points1 = self._data[point_indices1]
            points2 = self._data[point_indices2]

            distances_arr = self.dist.pairwise(points1, points2)
            distances = (<np.double_t [:points1.shape[0], :points2.shape[0]:1]> (<np.double_t *> distances_arr.data))

            for i in range(point_indices1.shape[0]):
                for j in range(point_indices2.shape[0]):
                    p = point_indices1[i]
                    q = point_indices2[j]
                    if p != q:
                        component1 = component_of_point_ptr[p]
                        component2 = component_of_point_ptr[q]
                        if component1 != component2:
                            mr_dist = max(distances[i, j], self.core_distance[p], self.core_distance[q])
                            if mr_dist < candidate_distance_ptr[component1]:
                                candidate_distance_ptr[component1] = mr_dist
                                self.candidate_neighbor[component1] = q
                                self.candidate_point[component1] = p

        elif node1_info.is_leaf or (not node2_info.is_leaf
                                    and node2_info.radius > node1_info.radius):
            self.dual_tree_traversal(node1, 2 * node2 + 1)
            self.dual_tree_traversal(node1, 2 * node2 + 2)
        else:
            self.dual_tree_traversal(2 * node1 + 1, node2)
            self.dual_tree_traversal(2 * node1 + 2, node2)

        return 0

    cpdef spanning_tree(self):
        num_components = self.tree.data.shape[0]
        while num_components > 1:
            self.dual_tree_traversal(0, 0)
            num_components = self.update_components()

        return np.array(list(self.edges))