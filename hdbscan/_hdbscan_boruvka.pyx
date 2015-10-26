#cython: boundscheck=False, nonecheck=False, profile=True
# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes
# License: 3-clause BSD

cimport cython

import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX

from scipy.spatial.distance import cdist, pdist, squareform

cdef points(tree, node, data, indices=False):
    node_data = tree.node_data[node]
    if not node_data['is_leaf']:

        if indices:
            return np.array([]), np.array([])
        else:
            return np.array([])

    else:
        idx_start = node_data['idx_start']
        idx_end = node_data['idx_end']
        selection = tree.idx_array[idx_start:idx_end]
        if indices:
            return data[selection], selection
        else:
            return data[selection]

cdef descendant_points(tree, node, data):
    node_data = tree.node_data[node]
    idx_start = node_data['idx_start']
    idx_end = node_data['idx_end']
    return data[tree.idx_array[idx_start:idx_end]]

cdef inline list children(object tree, long long node):
    node_data = tree.node_data[node]
    if node_data['is_leaf']:
        return []
    else:
        return [2 * node + 1, 2 * node + 2]

cdef inline double min_dist_dual(object tree1,
                                 object tree2,
                                 long long node1,
                                 long long node2,
                                 np.ndarray[double, ndim=2] centroid_dist):
    dist_pt = centroid_dist[node1, node2]
    return max(0, (dist_pt - tree1.node_data[node1]['radius']
                    - tree2.node_data[node2]['radius']))

cdef double max_child_distance(object tree, long long node, np.ndarray data):
    node_points = points(tree, node, data)
    if node_points.shape[0] > 0:
        centroid = tree.node_bounds[0, node]
        point_distances = cdist([centroid], node_points)[0]
        return np.max(point_distances)
    else:
        return 0.0

cdef double max_descendant_distance(object tree, long long node, np.ndarray data):
    node_points = descendant_points(tree, node, data)
    centroid = tree.node_bounds[0, node]
    point_distances = cdist([centroid], node_points)[0]
    return np.max(point_distances)

cdef class BoruvkaUnionFind (object):

    cdef np.ndarray _data

    def __init__(self, size):
        self._data = np.zeros((size, 2))
        self._data.T[0] = np.arange(size)

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
        return self._data.T[0]

cdef class BoruvkaAlgorithm (object):

    cdef object tree
    cdef np.ndarray _data
    cdef np.ndarray bounds
    cdef dict component_of_point
    cdef dict component_of_node
    cdef dict candidate_neighbor
    cdef dict candidate_point
    cdef dict candidate_distance
    cdef object component_union_find
    cdef set edges

    cdef np.ndarray _centroid_distances

    def __init__(self, tree):
        self.tree = tree
        self._data = np.array(tree.data)
        self.bounds = np.zeros(tree.node_bounds[0].shape[0])
        self.component_of_point = {}
        self.component_of_node = {}
        self.candidate_neighbor = {}
        self.candidate_point = {}
        self.candidate_distance = {}
        self.component_union_find = BoruvkaUnionFind(tree.data.shape[0])
        self.edges = set([])


        self._centroid_distances = squareform(pdist(tree.node_bounds[0]))
        self._compute_bounds()
        self._initialize_components()

    cdef _compute_bounds(self):
        nn_dist = self.tree.query(self.tree.data, 2)[0][:,-1]

        for n in range(self.tree.node_data.shape[0] - 1, -1, -1):
            if self.tree.node_data[n]['is_leaf']:
                node_points, node_point_indices = points(self.tree, n, self._data, indices=True)
                b1 = nn_dist[node_point_indices].max()
                b2 = (nn_dist[node_point_indices] + 2 * max_descendant_distance(self.tree, n, self._data)).min()
                self.bounds[n] = min(b1, b2)
            else:
                child_nodes = children(self.tree, n)
                lambda_children = np.array([max_descendant_distance(self.tree, c, self._data) for c in child_nodes])
                b1 = self.bounds[child_nodes].max()
                b2 = (self.bounds[child_nodes] + 2 * (max_descendant_distance(self.tree, n, self._data) - lambda_children)).min()
                if b2 > 0:
                    self.bounds[n] = min(b1, b2)
                else:
                    self.bounds[n] = b1

        for n in range(1, self.tree.node_data.shape[0]):
            self.bounds[n] = min(self.bounds[n], self.bounds[(n - 1) // 2])

    cdef _initialize_components(self):
        self.component_of_point = {n:n for n in range(self.tree.data.shape[0])}
        self.component_of_node = {n:-(n+1) for n in range(self.tree.node_data.shape[0])}
        self.candidate_neighbor = {n:None for n in range(self.tree.data.shape[0])}
        self.candidate_point = {n:None for n in range(self.tree.data.shape[0])}
        self.candidate_distance = {n:np.infty for n in range(self.tree.data.shape[0])}

    cpdef score(self, node1, node2):
        node_dist = min_dist_dual(self.tree, self.tree, node1, node2, self._centroid_distances)
        if node_dist < self.bounds[node1]:
            if self.component_of_node[node1] == self.component_of_node[node2] and \
                    self.component_of_node[node1] >= 0 and self.component_of_node[node2] >= 0:
                return np.infty
            else:
                return node_dist
        else:
            return np.infty

    cdef base_case(self, long long p, long long q, double point_distance):

        cdef long long component

        component = self.component_of_point[p]
        if component != self.component_of_point[q] and \
                point_distance < self.candidate_distance[component]:
            self.candidate_distance[component] = point_distance
            self.candidate_neighbor[component] = q
            self.candidate_point[component] = p

        return point_distance

    cdef update_components(self):

        cdef long long source
        cdef long long sink

        components = np.unique(self.component_union_find.components())
        for component in components:
            source, sink = sorted([self.candidate_point[component],
                                   self.candidate_neighbor[component]])
            if source is None or sink is None:
                raise ValueError('Source or sink of edge is None!')
            self.edges.add((source, sink, self.candidate_distance[component]))
            self.component_union_find.union_(source, sink)
            self.candidate_distance[component] = np.infty

        for n in range(self.tree.data.shape[0]):
            self.component_of_point[n] = self.component_union_find.find(n)

        for n in range(self.tree.node_data.shape[0] - 1, -1, -1):
            if self.tree.node_data[n]['is_leaf']:
                components_of_points = np.array([self.component_of_point[p] for p in points(self.tree, n, self._data, indices=True)[1]])
                if np.all(components_of_points == components_of_points[0]):
                    self.component_of_node[n] = components_of_points[0]
            else:
                child1, child2 = children(self.tree, n)
                if self.component_of_node[child1] == self. component_of_node[child2]:
                    self.component_of_node[n] = self.component_of_node[child1]

        components = np.unique(self.component_union_find.components())
        return components.shape[0]

    cdef void dual_tree_traversal(self, long long node1, long long node2):

        cdef np.ndarray[np.double_t, ndim=2] distances

        cdef np.ndarray points1, point2
        # cdef np.ndarray point_indices1, point_indices2

        cdef long long i
        cdef long long j

        cdef long long p
        cdef long long q

        cdef long long child1
        cdef long long child2

        cdef double node_dist

        if np.isinf(self.score(node1, node2)):
            return
        # node_dist = min_dist_dual(self.tree, self.tree, node1, node2, self._centroid_distances)
        # if node_dist < self.bounds[node1]:
        #     if self.component_of_node[node1] == self.component_of_node[node2] and \
        #             self.component_of_node[node1] >= 0 and self.component_of_node[node2] >= 0:
        #         return
        # else:
        #     return


        if self.tree.node_data[node1]['is_leaf'] and self.tree.node_data[node2]['is_leaf']:
            points1, point_indices1 = points(self.tree, node1, self._data, indices=True)
            points2, point_indices2 = points(self.tree, node2, self._data, indices=True)

            distances = cdist(points1, points2)
            for i in range(point_indices1.shape[0]):
                for j in range(point_indices2.shape[0]):
                    if distances[i, j] > 0:
                        p = point_indices1[i]
                        q = point_indices2[j]
                        self.base_case(p, q, distances[i, j])
        else:
            for child1 in children(self.tree, node1):
                for child2 in children(self.tree, node2):
                    self.dual_tree_traversal(child1, child2)

    cpdef spanning_tree(self):
        num_components = self.tree.data.shape[0]
        while num_components > 1:
            self.dual_tree_traversal(0, 0)
            num_components = self.update_components()

        return np.array(list(self.edges))