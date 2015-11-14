#cython: boundscheck=False, nonecheck=False, wraparound=False, initializedcheck=False
# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes
# License: 3-clause BSD

cimport cython

import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX
from libc.math cimport fabs, sqrt, exp, cos, pow

# from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import KDTree, BallTree

import dist_metrics as dist_metrics
cimport dist_metrics as dist_metrics

from libc.math cimport fabs, sqrt, exp, cos, pow, log

cdef np.double_t INF = np.inf

#cdef inline np.double_t euclidean_dist(np.double_t* x1, np.double_t* x2,
#                                       np.int64_t size) nogil except -1:
#    cdef np.double_t tmp, d=0
#    cdef np.intp_t j
#    for j in range(size):
#        tmp = x1[j] - x2[j]
#        d += tmp * tmp
#    return sqrt(d)


cdef struct NodeData_t:
    np.int64_t idx_start
    np.int64_t idx_end
    np.int64_t is_leaf
    np.double_t radius


cdef inline np.double_t balltree_min_dist_dual(np.double_t radius1,
                                               np.double_t radius2,
                                               np.int64_t node1,
                                               np.int64_t node2,
                                               np.double_t[:, ::1] centroid_dist):
    cdef np.double_t dist_pt = centroid_dist[node1, node2]
    return max(0, (dist_pt - radius1 - radius2))

cdef inline np.double_t kdtree_min_dist_dual(dist_metrics.DistanceMetric metric,
                                             np.int64_t node1,
                                             np.int64_t node2,
                                             np.double_t[:, :, ::1] node_bounds,
                                             np.int64_t num_features):

    cdef np.double_t d, d1, d2, rdist=0.0
    cdef np.double_t zero = 0.0
    cdef np.int64_t j

    if metric.p == INF:
        for j in range(num_features):
            d1 = (node_bounds[0, node1, j]
                  - node_bounds[1, node2, j])
            d2 = (node_bounds[0, node2, j]
                  - node_bounds[1, node1, j])
            d = (d1 + fabs(d1)) + (d2 + fabs(d2))

            rdist = max(rdist, 0.5 * d)
    else:
        # here we'll use the fact that x + abs(x) = 2 * max(x, 0)
        for j in range(num_features):
            d1 = (node_bounds[0, node1, j]
                  - node_bounds[1, node2, j])
            d2 = (node_bounds[0, node2, j]
                  - node_bounds[1, node1, j])
            d = (d1 + fabs(d1)) + (d2 + fabs(d2))

            rdist += pow(0.5 * d, metric.p)

    return metric._rdist_to_dist(rdist)


cdef class BoruvkaUnionFind (object):

    cdef np.ndarray _data_arr
    cdef np.int64_t[:,::1] _data
    cdef np.ndarray is_component

    def __init__(self, size):
        self._data_arr = np.zeros((size, 2), dtype=np.int64)
        self._data_arr.T[0] = np.arange(size)
        self._data = (<np.int64_t[:size, :2:1]> (<np.int64_t *> self._data_arr.data))
        self.is_component = np.ones(size, dtype=np.bool)

    cdef union_(self, np.int64_t x, np.int64_t y):
        cdef np.int64_t x_root = self.find(x)
        cdef np.int64_t y_root = self.find(y)

        if self._data[x_root, 1] < self._data[y_root, 1]:
            self._data[x_root, 0] = y_root
        elif self._data[x_root, 1] > self._data[y_root, 1]:
            self._data[y_root, 0] = x_root
        else:
            self._data[y_root, 0] = x_root
            self._data[x_root, 1] += 1

        return

    cdef find(self, np.int64_t x):
        if self._data[x, 0] != x:
            self._data[x, 0] = self.find(self._data[x, 0])
            self.is_component[x] = False
        return self._data[x, 0]

    cdef np.ndarray[np.int64_t, ndim=1] components(self):
        return self.is_component.nonzero()[0]

cdef class KDTreeBoruvkaAlgorithm (object):

    cdef object tree
    cdef object core_dist_tree
    cdef dist_metrics.DistanceMetric dist
    cdef np.ndarray _data
    cdef np.double_t[:, ::1] _raw_data
    cdef np.double_t[:, :, ::1] node_bounds
    cdef np.int64_t min_samples
    cdef np.int64_t num_points
    cdef np.int64_t num_nodes
    cdef np.int64_t num_features

    cdef public np.double_t[::1] core_distance
    cdef public np.double_t[::1] bounds
    cdef public np.int64_t[::1] component_of_point
    cdef public np.int64_t[::1] component_of_node
    cdef public np.int64_t[::1] candidate_neighbor
    cdef public np.int64_t[::1] candidate_point
    cdef public np.double_t[::1] candidate_distance
    cdef public np.double_t[:,::1] centroid_distances
    cdef public np.int64_t[::1] idx_array
    cdef public NodeData_t[::1] node_data
    cdef BoruvkaUnionFind component_union_find
    cdef np.ndarray edges
    cdef np.int64_t num_edges

    cdef np.int64_t *component_of_point_ptr
    cdef np.int64_t *component_of_node_ptr
    cdef np.double_t *candidate_distance_ptr
    cdef np.int64_t *candidate_neighbor_ptr
    cdef np.int64_t *candidate_point_ptr
    cdef np.double_t *core_distance_ptr
    cdef np.double_t *bounds_ptr

    cdef np.ndarray components
    cdef np.ndarray core_distance_arr
    cdef np.ndarray bounds_arr
    cdef np.ndarray _centroid_distances_arr
    cdef np.ndarray component_of_point_arr
    cdef np.ndarray component_of_node_arr
    cdef np.ndarray candidate_point_arr
    cdef np.ndarray candidate_neighbor_arr
    cdef np.ndarray candidate_distance_arr

    def __init__(self, tree, min_samples=5, metric='euclidean', **kwargs):

        self.core_dist_tree = tree
        self.tree = KDTree(tree.data, metric=metric, leaf_size=10)
        self._data = np.array(self.tree.data)
        self._raw_data = self.tree.data
        self.node_bounds = self.tree.node_bounds
        self.min_samples = min_samples

        self.num_points = self.tree.data.shape[0]
        self.num_features = self.tree.data.shape[1]
        self.num_nodes = self.tree.node_data.shape[0]

        self.dist = dist_metrics.DistanceMetric.get_metric(metric, **kwargs)

        self.components = np.arange(self.num_points)
        self.bounds_arr = np.empty(self.num_nodes, np.double)
        self.component_of_point_arr = np.empty(self.num_points, dtype=np.int64)
        self.component_of_node_arr = np.empty(self.num_nodes, dtype=np.int64)
        self.candidate_neighbor_arr = np.empty(self.num_points, dtype=np.int64)
        self.candidate_point_arr = np.empty(self.num_points, dtype=np.int64)
        self.candidate_distance_arr = np.empty(self.num_points, dtype=np.double)
        self.component_union_find = BoruvkaUnionFind(self.num_points)

        self.edges = np.empty((self.num_points - 1, 3))
        self.num_edges = 0

        self.idx_array = self.tree.idx_array
        self.node_data = self.tree.node_data

        self.bounds = (<np.double_t[:self.num_nodes:1]> (<np.double_t *> self.bounds_arr.data))
        self.component_of_point = (<np.int64_t[:self.num_points:1]> (<np.int64_t *> self.component_of_point_arr.data))
        self.component_of_node = (<np.int64_t[:self.num_nodes:1]> (<np.int64_t *> self.component_of_node_arr.data))
        self.candidate_neighbor = (<np.int64_t[:self.num_points:1]> (<np.int64_t *> self.candidate_neighbor_arr.data))
        self.candidate_point = (<np.int64_t[:self.num_points:1]> (<np.int64_t *> self.candidate_point_arr.data))
        self.candidate_distance = (<np.double_t[:self.num_points:1]> (<np.double_t *> self.candidate_distance_arr.data))

        #self._centroid_distances_arr = self.dist.pairwise(self.tree.node_bounds[0])
        #self.centroid_distances = (<np.double_t [:self.num_nodes, :self.num_nodes:1]> (<np.double_t *> self._centroid_distances_arr.data))

        self._initialize_components()
        self._compute_bounds()

        # Set up fast pointer access to arrays
        self.component_of_point_ptr = <np.int64_t *> &self.component_of_point[0]
        self.component_of_node_ptr = <np.int64_t *> &self.component_of_node[0]
        self.candidate_distance_ptr = <np.double_t *> &self.candidate_distance[0]
        self.candidate_neighbor_ptr = <np.int64_t *> &self.candidate_neighbor[0]
        self.candidate_point_ptr = <np.int64_t *> &self.candidate_point[0]
        self.core_distance_ptr = <np.double_t *> &self.core_distance[0]
        self.bounds_ptr = <np.double_t *> &self.bounds[0]

    @cython.profile(True)
    cdef _compute_bounds(self):

        cdef np.int64_t n

        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.int64_t, ndim=2] knn_indices

        knn_dist, knn_indices = self.core_dist_tree.query(self.tree.data,
                                                          k=self.min_samples,
                                                          dualtree=True,
                                                          breadth_first=True)
        self.core_distance_arr = knn_dist[:, self.min_samples - 1].copy()
        self.core_distance = (<np.double_t [:self.num_points:1]> (<np.double_t *> self.core_distance_arr.data))

        for n in range(self.num_nodes):
            self.bounds_arr[n] = <np.double_t> DBL_MAX

    cdef _initialize_components(self):

        cdef np.int64_t n

        for n in range(self.num_points):
            self.component_of_point[n] = n
            self.candidate_neighbor[n] = -1
            self.candidate_point[n] = -1
            self.candidate_distance[n] = DBL_MAX

        for n in range(self.num_nodes):
            self.component_of_node[n] = -(n+1)

    cpdef update_components(self):

        cdef np.int64_t source
        cdef np.int64_t sink
        cdef np.int64_t c
        cdef np.int64_t component
        cdef np.int64_t n
        cdef np.int64_t i
        cdef np.int64_t p
        cdef np.int64_t current_component
        cdef np.int64_t current_source_component
        cdef np.int64_t current_sink_component
        cdef np.int64_t child1
        cdef np.int64_t child2

        cdef NodeData_t node_info

        for c in range(self.components.shape[0]):
            component = self.components[c]
            source = self.candidate_point[component]
            sink = self.candidate_neighbor[component]
            if source == -1 or sink == -1:
                continue
                # raise ValueError('Source or sink of edge is not defined!')
            current_source_component = self.component_union_find.find(source)
            current_sink_component = self.component_union_find.find(sink)
            if current_source_component == current_sink_component:
                self.candidate_point[component] = -1
                self.candidate_neighbor[component] = -1
                self.candidate_distance[component] = DBL_MAX
                continue
            self.edges[self.num_edges, 0] = source
            self.edges[self.num_edges, 1] = sink
            self.edges[self.num_edges, 2] = self.candidate_distance[component]
            self.num_edges += 1
            self.component_union_find.union_(source, sink)
            self.candidate_distance[component] = DBL_MAX
            if self.num_edges == self.num_points - 1:
                self.components = self.component_union_find.components()
                return self.components.shape[0]

        for n in range(self.tree.data.shape[0]):
            self.component_of_point[n] = self.component_union_find.find(n)

        for n in range(self.tree.node_data.shape[0] - 1, -1, -1):
            node_info = self.node_data[n]
            if node_info.is_leaf:
                current_component = self.component_of_point[self.idx_array[node_info.idx_start]]
                for i in range(node_info.idx_start + 1, node_info.idx_end):
                    p = self.idx_array[i]
                    if self.component_of_point[p] != current_component:
                        break
                else:
                    self.component_of_node[n] = current_component
            else:
                child1 = 2 * n + 1
                child2 = 2 * n + 2
                if self.component_of_node[child1] == self.component_of_node[child2]:
                    self.component_of_node[n] = self.component_of_node[child1]

        last_num_components = self.components.shape[0]
        self.components = self.component_union_find.components()

        if self.components.shape[0] == last_num_components:
            # Reset bounds
            for n in range(self.num_nodes):
                self.bounds_arr[n] = <np.double_t> DBL_MAX

        return self.components.shape[0]

    cdef int dual_tree_traversal(self, np.int64_t node1, np.int64_t node2):

        cdef np.int64_t[::1] point_indices1, point_indices2

        cdef long long i
        cdef long long j

        cdef np.int64_t p
        cdef np.int64_t q

        cdef np.int64_t parent
        cdef np.int64_t child1
        cdef np.int64_t child2

        cdef double node_dist

        cdef NodeData_t node1_info = self.node_data[node1]
        cdef NodeData_t node2_info = self.node_data[node2]
        cdef NodeData_t parent_info
        cdef NodeData_t left_info
        cdef NodeData_t right_info

        cdef np.int64_t component1
        cdef np.int64_t component2

        cdef np.double_t *raw_data = (<np.double_t *> &self._raw_data[0,0])
        cdef np.double_t d

        cdef np.double_t mr_dist

        cdef np.double_t new_bound
        cdef np.double_t new_upper_bound
        cdef np.double_t new_lower_bound
        cdef np.double_t bound_max
        cdef np.double_t bound_min

        cdef np.int64_t left
        cdef np.int64_t right
        cdef np.double_t left_dist
        cdef np.double_t right_dist


        node_dist = kdtree_min_dist_dual(self.dist,
                                      node1, node2, self.node_bounds, self.num_features)

        if node_dist < self.bounds_ptr[node1]:
            if self.component_of_node_ptr[node1] == self.component_of_node_ptr[node2] and \
                    self.component_of_node_ptr[node1] >= 0:
                return 0
        else:
            return 0


        if node1_info.is_leaf and node2_info.is_leaf:

            new_upper_bound = 0.0
            new_lower_bound = DBL_MAX

            point_indices1 = self.idx_array[node1_info.idx_start:node1_info.idx_end]
            point_indices2 = self.idx_array[node2_info.idx_start:node2_info.idx_end]

            for i in range(point_indices1.shape[0]):

                p = point_indices1[i]
                component1 = self.component_of_point_ptr[p]

                if self.core_distance_ptr[p] > self.candidate_distance_ptr[component1]:
                    continue

                for j in range(point_indices2.shape[0]):

                    q = point_indices2[j]
                    component2 = self.component_of_point_ptr[q]

                    if self.core_distance_ptr[q] > self.candidate_distance_ptr[component1]:
                        continue

                    if component1 != component2:

                        d = self.dist.dist(&raw_data[self.num_features * p],
                                           &raw_data[self.num_features * q],
                                           self.num_features)

                        # mr_dist = max(distances[i, j], self.core_distance_ptr[p], self.core_distance_ptr[q])
                        mr_dist = max(d, self.core_distance_ptr[p], self.core_distance_ptr[q])
                        if mr_dist < self.candidate_distance_ptr[component1]:
                            self.candidate_distance_ptr[component1] = mr_dist
                            self.candidate_neighbor_ptr[component1] = q
                            self.candidate_point_ptr[component1] = p

                new_upper_bound = max(new_upper_bound, self.candidate_distance_ptr[component1])
                new_lower_bound = min(new_lower_bound, self.candidate_distance_ptr[component1])

            new_bound = min(new_upper_bound, new_lower_bound + 2 * node1_info.radius)
            if new_bound < self.bounds_ptr[node1]:
                self.bounds_ptr[node1] = new_bound

                # Propagate bounds up the tree
                while node1 > 0:
                    parent = (node1 - 1) // 2
                    left = 2 * parent + 1
                    right = 2 * parent + 2

                    parent_info = self.node_data[parent]
                    left_info = self.node_data[left]
                    right_info = self.node_data[right]

                    bound_max = max(self.bounds_ptr[left],
                                    self.bounds_ptr[right])
                    bound_min = min(self.bounds_ptr[left] + 2 * (parent_info.radius - left_info.radius),
                                    self.bounds_ptr[right] + 2 * (parent_info.radius - right_info.radius))

                    if bound_min > 0:
                        new_bound = min(bound_max, bound_min)
                    else:
                        new_bound = bound_max

                    if new_bound < self.bounds_ptr[parent]:
                        self.bounds_ptr[parent] = new_bound
                        node1 = parent
                    else:
                        break


        elif node1_info.is_leaf or (not node2_info.is_leaf
                                    and node2_info.radius > node1_info.radius):

            left = 2 * node2 + 1
            right = 2 * node2 + 2

            node2_info = self.node_data[left]

            left_dist = kdtree_min_dist_dual(self.dist,
                                      node1, left, self.node_bounds, self.num_features)

            node2_info = self.node_data[right]

            right_dist = kdtree_min_dist_dual(self.dist,
                                      node1, right, self.node_bounds, self.num_features)

            if left_dist < right_dist:
                self.dual_tree_traversal(node1, left)
                self.dual_tree_traversal(node1, right)
            else:
                self.dual_tree_traversal(node1, right)
                self.dual_tree_traversal(node1, left)
        else:
            left = 2 * node1 + 1
            right = 2 * node1 + 2

            node1_info = self.node_data[left]

            left_dist = kdtree_min_dist_dual(self.dist,
                                      left, node2, self.node_bounds, self.num_features)

            node1_info = self.node_data[right]

            right_dist = kdtree_min_dist_dual(self.dist,
                                      right, node2, self.node_bounds, self.num_features)

            if left_dist < right_dist:
                self.dual_tree_traversal(left, node2)
                self.dual_tree_traversal(right, node2)
            else:
                self.dual_tree_traversal(right, node2)
                self.dual_tree_traversal(left, node2)


        return 0

    cpdef spanning_tree(self):

        cdef np.int64_t num_components
        cdef np.int64_t num_nodes

        num_components = self.tree.data.shape[0]
        num_nodes = self.tree.node_data.shape[0]
        while num_components > 1:
            self.dual_tree_traversal(0, 0)
            num_components = self.update_components()

        return self.edges

cdef class BallTreeBoruvkaAlgorithm (object):

    cdef object tree
    cdef object core_dist_tree
    cdef dist_metrics.DistanceMetric dist
    cdef np.ndarray _data
    cdef np.double_t[:, ::1] _raw_data
    cdef np.int64_t min_samples
    cdef np.int64_t num_points
    cdef np.int64_t num_nodes
    cdef np.int64_t num_features

    cdef public np.double_t[::1] core_distance
    cdef public np.double_t[::1] bounds
    cdef public np.int64_t[::1] component_of_point
    cdef public np.int64_t[::1] component_of_node
    cdef public np.int64_t[::1] candidate_neighbor
    cdef public np.int64_t[::1] candidate_point
    cdef public np.double_t[::1] candidate_distance
    cdef public np.double_t[:,::1] centroid_distances
    cdef public np.int64_t[::1] idx_array
    cdef public NodeData_t[::1] node_data
    cdef BoruvkaUnionFind component_union_find
    cdef np.ndarray edges
    cdef np.int64_t num_edges

    cdef np.int64_t *component_of_point_ptr
    cdef np.int64_t *component_of_node_ptr
    cdef np.double_t *candidate_distance_ptr
    cdef np.int64_t *candidate_neighbor_ptr
    cdef np.int64_t *candidate_point_ptr
    cdef np.double_t *core_distance_ptr
    cdef np.double_t *bounds_ptr

    cdef np.ndarray components
    cdef np.ndarray core_distance_arr
    cdef np.ndarray bounds_arr
    cdef np.ndarray _centroid_distances_arr
    cdef np.ndarray component_of_point_arr
    cdef np.ndarray component_of_node_arr
    cdef np.ndarray candidate_point_arr
    cdef np.ndarray candidate_neighbor_arr
    cdef np.ndarray candidate_distance_arr

    def __init__(self, tree, min_samples=5, metric='euclidean', **kwargs):

        self.core_dist_tree = tree
        self.tree = BallTree(tree.data, metric=metric, leaf_size=10)
        self._data = np.array(self.tree.data)
        self._raw_data = self.tree.data
        self.min_samples = min_samples

        self.num_points = self.tree.data.shape[0]
        self.num_features = self.tree.data.shape[1]
        self.num_nodes = self.tree.node_data.shape[0]


        self.dist = dist_metrics.DistanceMetric.get_metric(metric, **kwargs)

        self.components = np.arange(self.num_points)
        self.bounds_arr = np.empty(self.num_nodes, np.double)
        self.component_of_point_arr = np.empty(self.num_points, dtype=np.int64)
        self.component_of_node_arr = np.empty(self.num_nodes, dtype=np.int64)
        self.candidate_neighbor_arr = np.empty(self.num_points, dtype=np.int64)
        self.candidate_point_arr = np.empty(self.num_points, dtype=np.int64)
        self.candidate_distance_arr = np.empty(self.num_points, dtype=np.double)
        self.component_union_find = BoruvkaUnionFind(self.num_points)

        self.edges = np.empty((self.num_points - 1, 3))
        self.num_edges = 0

        self.idx_array = self.tree.idx_array
        self.node_data = self.tree.node_data

        self.bounds = (<np.double_t[:self.num_nodes:1]> (<np.double_t *> self.bounds_arr.data))
        self.component_of_point = (<np.int64_t[:self.num_points:1]> (<np.int64_t *> self.component_of_point_arr.data))
        self.component_of_node = (<np.int64_t[:self.num_nodes:1]> (<np.int64_t *> self.component_of_node_arr.data))
        self.candidate_neighbor = (<np.int64_t[:self.num_points:1]> (<np.int64_t *> self.candidate_neighbor_arr.data))
        self.candidate_point = (<np.int64_t[:self.num_points:1]> (<np.int64_t *> self.candidate_point_arr.data))
        self.candidate_distance = (<np.double_t[:self.num_points:1]> (<np.double_t *> self.candidate_distance_arr.data))

        self._centroid_distances_arr = self.dist.pairwise(self.tree.node_bounds[0])
        self.centroid_distances = (<np.double_t [:self.num_nodes, :self.num_nodes:1]> (<np.double_t *> self._centroid_distances_arr.data))

        self._initialize_components()
        self._compute_bounds()

        # Set up fast pointer access to arrays
        self.component_of_point_ptr = <np.int64_t *> &self.component_of_point[0]
        self.component_of_node_ptr = <np.int64_t *> &self.component_of_node[0]
        self.candidate_distance_ptr = <np.double_t *> &self.candidate_distance[0]
        self.candidate_neighbor_ptr = <np.int64_t *> &self.candidate_neighbor[0]
        self.candidate_point_ptr = <np.int64_t *> &self.candidate_point[0]
        self.core_distance_ptr = <np.double_t *> &self.core_distance[0]
        self.bounds_ptr = <np.double_t *> &self.bounds[0]

    cdef _compute_bounds(self):

        cdef np.int64_t n

        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.int64_t, ndim=2] knn_indices

        knn_dist, knn_indices = self.core_dist_tree.query(self.tree.data,
                                                          k=self.min_samples,
                                                          dualtree=True,
                                                          breadth_first=True)
        self.core_distance_arr = knn_dist[:, self.min_samples - 1].copy()
        self.core_distance = (<np.double_t [:self.num_points:1]> (<np.double_t *> self.core_distance_arr.data))

        for n in range(self.num_nodes):
            self.bounds_arr[n] = <np.double_t> DBL_MAX

    cdef _initialize_components(self):

        cdef np.int64_t n

        for n in range(self.num_points):
            self.component_of_point[n] = n
            self.candidate_neighbor[n] = -1
            self.candidate_point[n] = -1
            self.candidate_distance[n] = DBL_MAX

        for n in range(self.num_nodes):
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
        cdef np.int64_t current_source_component
        cdef np.int64_t current_sink_component
        cdef np.int64_t child1
        cdef np.int64_t child2

        cdef NodeData_t node_info

        for c in range(self.components.shape[0]):
            component = self.components[c]
            source = self.candidate_point[component]
            sink = self.candidate_neighbor[component]
            if source == -1 or sink == -1:
                continue
                # raise ValueError('Source or sink of edge is not defined!')
            current_source_component = self.component_union_find.find(source)
            current_sink_component = self.component_union_find.find(sink)
            if current_source_component == current_sink_component:
                self.candidate_point[component] = -1
                self.candidate_neighbor[component] = -1
                self.candidate_distance[component] = DBL_MAX
                continue
            self.edges[self.num_edges, 0] = source
            self.edges[self.num_edges, 1] = sink
            self.edges[self.num_edges, 2] = self.candidate_distance[component]
            self.num_edges += 1
            self.component_union_find.union_(source, sink)
            self.candidate_distance[component] = DBL_MAX
            if self.num_edges == self.num_points - 1:
                self.components = self.component_union_find.components()
                return self.components.shape[0]

        for n in range(self.tree.data.shape[0]):
            self.component_of_point[n] = self.component_union_find.find(n)

        for n in range(self.tree.node_data.shape[0] - 1, -1, -1):
            node_info = self.node_data[n]
            if node_info.is_leaf:
                current_component = self.component_of_point[self.idx_array[node_info.idx_start]]
                for i in range(node_info.idx_start + 1, node_info.idx_end):
                    p = self.idx_array[i]
                    if self.component_of_point[p] != current_component:
                        break
                else:
                    self.component_of_node[n] = current_component
            else:
                child1 = 2 * n + 1
                child2 = 2 * n + 2
                if self.component_of_node[child1] == self.component_of_node[child2]:
                    self.component_of_node[n] = self.component_of_node[child1]

        last_num_components = self.components.shape[0]
        self.components = self.component_union_find.components()

        if self.components.shape[0] == last_num_components:
            # Reset bounds
            for n in range(self.num_nodes):
                self.bounds_arr[n] = <np.double_t> DBL_MAX

        return self.components.shape[0]

    cdef int dual_tree_traversal(self, np.int64_t node1, np.int64_t node2):

        cdef np.int64_t[::1] point_indices1, point_indices2

        cdef long long i
        cdef long long j

        cdef np.int64_t p
        cdef np.int64_t q

        cdef np.int64_t parent
        cdef np.int64_t child1
        cdef np.int64_t child2

        cdef double node_dist

        cdef NodeData_t node1_info = self.node_data[node1]
        cdef NodeData_t node2_info = self.node_data[node2]
        cdef NodeData_t parent_info
        cdef NodeData_t left_info
        cdef NodeData_t right_info

        cdef np.int64_t component1
        cdef np.int64_t component2

        cdef np.double_t *raw_data = (<np.double_t *> &self._raw_data[0,0])
        cdef np.double_t d

        cdef np.double_t mr_dist

        cdef np.double_t new_bound
        cdef np.double_t new_upper_bound
        cdef np.double_t new_lower_bound
        cdef np.double_t bound_max
        cdef np.double_t bound_min

        cdef np.int64_t left
        cdef np.int64_t right
        cdef np.double_t left_dist
        cdef np.double_t right_dist

        node_dist = balltree_min_dist_dual(node1_info.radius, node2_info.radius,
                                    node1, node2, self.centroid_distances)

        if node_dist < self.bounds_ptr[node1]:
            if self.component_of_node_ptr[node1] == self.component_of_node_ptr[node2] and \
                    self.component_of_node_ptr[node1] >= 0:
                return 0
        else:
            return 0


        if node1_info.is_leaf and node2_info.is_leaf:

            new_bound = 0.0

            point_indices1 = self.idx_array[node1_info.idx_start:node1_info.idx_end]
            point_indices2 = self.idx_array[node2_info.idx_start:node2_info.idx_end]

            for i in range(point_indices1.shape[0]):

                p = point_indices1[i]
                component1 = self.component_of_point_ptr[p]

                if self.core_distance_ptr[p] > self.candidate_distance_ptr[component1]:
                    continue

                for j in range(point_indices2.shape[0]):

                    q = point_indices2[j]
                    component2 = self.component_of_point_ptr[q]

                    if self.core_distance_ptr[q] > self.candidate_distance_ptr[component1]:
                        continue

                    if component1 != component2:

                        d = self.dist.dist(&raw_data[self.num_features * p],
                                           &raw_data[self.num_features * q],
                                           self.num_features)

                        mr_dist = max(d, self.core_distance_ptr[p], self.core_distance_ptr[q])
                        if mr_dist < self.candidate_distance_ptr[component1]:
                            self.candidate_distance_ptr[component1] = mr_dist
                            self.candidate_neighbor_ptr[component1] = q
                            self.candidate_point_ptr[component1] = p

                new_upper_bound = max(new_upper_bound, self.candidate_distance_ptr[component1])
                new_lower_bound = min(new_lower_bound, self.candidate_distance_ptr[component1])

            new_bound = min(new_upper_bound, new_lower_bound + 2 * node1_info.radius)
            if new_bound < self.bounds_ptr[node1]:
                self.bounds_ptr[node1] = new_bound

                # Propagate bounds up the tree
                while node1 > 0:
                    parent = (node1 - 1) // 2
                    left = 2 * parent + 1
                    right = 2 * parent + 2

                    parent_info = self.node_data[parent]
                    left_info = self.node_data[left]
                    right_info = self.node_data[right]

                    bound_max = max(self.bounds_ptr[left],
                                    self.bounds_ptr[right])
                    bound_min = min(self.bounds_ptr[left] + 2 * (parent_info.radius - left_info.radius),
                                    self.bounds_ptr[right] + 2 * (parent_info.radius - right_info.radius))

                    if bound_min > 0:
                        new_bound = min(bound_max, bound_min)
                    else:
                        new_bound = bound_max

                    if new_bound < self.bounds_ptr[parent]:
                        self.bounds_ptr[parent] = new_bound
                        node1 = parent
                    else:
                        break


        elif node1_info.is_leaf or (not node2_info.is_leaf
                                    and node2_info.radius > node1_info.radius):

            left = 2 * node2 + 1
            right = 2 * node2 + 2

            node2_info = self.node_data[left]

            left_dist = balltree_min_dist_dual(node1_info.radius, node2_info.radius,
                                      node1, left, self.centroid_distances)

            node2_info = self.node_data[right]

            right_dist = balltree_min_dist_dual(node1_info.radius, node2_info.radius,
                                       node1, right, self.centroid_distances)

            if left_dist < right_dist:
                self.dual_tree_traversal(node1, left)
                self.dual_tree_traversal(node1, right)
            else:
                self.dual_tree_traversal(node1, right)
                self.dual_tree_traversal(node1, left)
        else:
            left = 2 * node1 + 1
            right = 2 * node1 + 2

            node1_info = self.node_data[left]

            left_dist = balltree_min_dist_dual(node1_info.radius, node2_info.radius,
                                      left, node2, self.centroid_distances)

            node1_info = self.node_data[right]

            right_dist = balltree_min_dist_dual(node1_info.radius, node2_info.radius,
                                       right, node2, self.centroid_distances)

            if left_dist < right_dist:
                self.dual_tree_traversal(left, node2)
                self.dual_tree_traversal(right, node2)
            else:
                self.dual_tree_traversal(right, node2)
                self.dual_tree_traversal(left, node2)


        return 0

    cpdef spanning_tree(self):

        cdef np.int64_t num_components
        cdef np.int64_t num_nodes

        num_components = self.tree.data.shape[0]
        num_nodes = self.tree.node_data.shape[0]
        while num_components > 1:
            self.dual_tree_traversal(0, 0)
            num_components = self.update_components()

        return self.edges