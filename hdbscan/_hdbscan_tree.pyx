#cython: boundscheck=False, nonecheck=False, initializedcheck=False, profile=True
# Tree handling (condensing, finding stable clusters) for hdbscan
# Authors: Leland McInnes
# License: 3-clause BSD

import numpy as np
cimport numpy as np

import itertools

cpdef np.ndarray get_points(np.ndarray[np.double_t, ndim=2] hierarchy):
    """
    Extract original point information from the single linkage hierarchy
    providing a list of points for each node in the hierarchy tree.
    """

    cdef long long max_node
    cdef long long num_points
    cdef long long dim
    cdef long long node
    cdef long long parent
    cdef np.ndarray[tuple, ndim=1] result
    cdef long long i
    cdef long long left_child
    cdef long long right_child

    dim = hierarchy.shape[0]
    max_node = 2 * dim
    num_points = max_node - (dim - 1)

    result = np.empty(max_node + 1, dtype=object)

    for node in range(num_points):
        result[node] = (node,)

    for i in range(dim):
        parent = i + num_points
        left_child = <long long> hierarchy[i,0]
        right_child = <long long> hierarchy[i,1]
        result[parent] = result[left_child] + result[right_child]

    return result

cdef list bfs_from_hierarchy(np.ndarray[np.double_t, ndim=2] hierarchy, long long bfs_root):

    """
    Perform a breadth first search on a tree in scipy hclust format.
    """
    
    cdef list to_process
    cdef long long max_node
    cdef long long num_points
    cdef long long dim

    dim = hierarchy.shape[0]
    max_node = 2 * dim
    num_points = max_node - dim + 1

    to_process = [bfs_root]
    result = []

    while to_process:
        result.extend(to_process)
        to_process = [x - num_points for x in 
                          to_process if x >= num_points]
        if to_process:
            to_process = hierarchy[to_process,:2].flatten().astype(np.int64).tolist()

    return result
        
cpdef np.ndarray condense_tree(np.ndarray[np.double_t, ndim=2] hierarchy,
                          long long min_cluster_size=10):

    cdef long long root
    cdef long long num_points
    cdef long long next_label
    cdef list node_list
    cdef list result_list

    cdef np.ndarray[np.int64_t, ndim=1] relabel
    cdef np.ndarray[np.int_t, ndim=1] ignore
    cdef np.ndarray[np.double_t, ndim=1] children

    cdef long long node
    cdef long long sub_node
    cdef long long left
    cdef long long right
    cdef double lambda_value
    cdef long long left_count
    cdef long long right_count
    
    root = 2 * hierarchy.shape[0]
    num_points = root // 2 + 1
    next_label = num_points + 1
    
    node_list = bfs_from_hierarchy(hierarchy, root)
    
    relabel = np.empty(len(node_list), dtype=np.int64)
    relabel[root] = num_points
    result_list = []
    ignore = np.zeros(len(node_list), dtype=np.int)
    
    for node in node_list:
        if ignore[node] or node < num_points:
            continue
            
        children = hierarchy[node - num_points]
        left = <long long> children[0]
        right = <long long> children[1]
        lambda_value = 1.0 / children[2] if children[2] > 0.0 else np.inf
        left_count = <long long> (hierarchy[left - num_points][3] 
                            if left >= num_points else 1)
        right_count = <long long> (hierarchy[right - num_points][3] 
                             if right >= num_points else 1)
        
        if left_count > min_cluster_size and right_count > min_cluster_size:
            relabel[left] = next_label
            next_label += 1
            result_list.append((relabel[node], relabel[left], lambda_value, left_count))
            
            relabel[right] = next_label
            next_label += 1
            result_list.append((relabel[node], relabel[right], lambda_value, right_count))
            
        elif left_count <= min_cluster_size and right_count <= min_cluster_size:
            for sub_node in bfs_from_hierarchy(hierarchy, left):
                if sub_node < num_points:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                ignore[sub_node] = True
                
            for sub_node in bfs_from_hierarchy(hierarchy, right):
                if sub_node < num_points:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                ignore[sub_node] = True
                
        elif left_count <= min_cluster_size:
            relabel[right] = relabel[node]
            for sub_node in bfs_from_hierarchy(hierarchy, left):
                if sub_node < num_points:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                ignore[sub_node] = True
                
        else:
            relabel[left] = relabel[node]
            for sub_node in bfs_from_hierarchy(hierarchy, right):
                if sub_node < num_points:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                ignore[sub_node] = True
                
    return np.array(result_list, dtype=[
                                        ('parent', np.int64),
                                        ('child', np.int64),
                                        ('lambda', float),
                                        ('child_size', np.int64)
                                       ])

cpdef dict compute_stability(np.ndarray condensed_tree):

    cdef np.ndarray[np.double_t, ndim=1] result_arr
    cdef np.ndarray sorted_child_data
    cdef np.ndarray[np.int64_t, ndim=1] sorted_children
    cdef np.ndarray[np.double_t, ndim=1] sorted_lambdas

    cdef np.int64_t child
    cdef np.int64_t current_child
    cdef np.float64_t lambda_
    cdef np.float64_t min_lambda

    cdef np.ndarray[np.double_t, ndim=1] births_arr
    cdef np.double_t *births

    cdef np.int64_t largest_child = condensed_tree['child'].max()
    cdef np.int64_t smallest_cluster = condensed_tree['parent'].min()
    cdef np.int64_t num_clusters = condensed_tree['parent'].max() - smallest_cluster + 1

    sorted_child_data = np.sort(condensed_tree[['child', 'lambda']], axis=0)
    births_arr = np.nan * np.ones(largest_child + 1, dtype=np.double)
    births = (<np.double_t *> births_arr.data)
    sorted_children = sorted_child_data['child']
    sorted_lambdas = sorted_child_data['lambda']

    current_child = -1
    min_lambda = 0

    for row in range(sorted_child_data.shape[0]):
        child = <np.int64_t> sorted_children[row]
        lambda_ = sorted_lambdas[row]

        if child == current_child:
            min_lambda = min(min_lambda, lambda_)
        elif current_child != -1:
            births[current_child] = min_lambda
            current_child = child
            min_lambda = lambda_
        else:
            # Initialize
            current_child = child
            min_lambda = lambda_

    result_arr = np.zeros(num_clusters, dtype=np.double)

    for i in range(condensed_tree.shape[0]):
        parent = condensed_tree['parent'][i]
        child = condensed_tree['child'][i]
        lambda_ = condensed_tree['lambda'][i]
        child_size = condensed_tree['child_size'][i]
        result_index = parent - smallest_cluster

        result_arr[result_index] += (lambda_ - births[parent]) * child_size

    return dict(np.vstack((np.arange(smallest_cluster, condensed_tree['parent'].max() + 1), result_arr)).T)

cdef list bfs_from_cluster_tree(np.ndarray tree, long long bfs_root):

    cdef list result
    cdef np.ndarray[np.int64_t, ndim=1] to_process

    result = []
    to_process = np.array([bfs_root], dtype=np.int64)
    
    while to_process.shape[0] > 0:
        result.extend(to_process.tolist())
        to_process = tree['child'][np.in1d(tree['parent'], to_process)]

    return result

cdef max_lambdas(np.ndarray tree):

    cdef np.ndarray sorted_parent_data
    cdef np.ndarray[np.int64_t, ndim=1] sorted_parents
    cdef np.ndarray[np.double_t, ndim=1] sorted_lambdas

    cdef np.int64_t parent
    cdef np.int64_t current_parent
    cdef np.float64_t lambda_
    cdef np.float64_t max_lambda

    cdef np.ndarray[np.double_t, ndim=1] deaths_arr
    cdef np.double_t *deaths

    cdef np.int64_t largest_parent = tree['parent'].max()

    sorted_parent_data = np.sort(tree[['parent', 'lambda']], axis=0)
    deaths_arr = np.zeros(largest_parent + 1, dtype=np.double)
    deaths = (<np.double_t *> deaths_arr.data)
    sorted_parents = sorted_parent_data['parent']
    sorted_lambdas = sorted_parent_data['lambda']

    current_parent = -1
    max_lambda = 0

    for row in range(sorted_parent_data.shape[0]):
        parent = <np.int64_t> sorted_parents[row]
        lambda_ = sorted_lambdas[row]

        if parent == current_parent:
            max_lambda = max(max_lambda, lambda_)
        elif current_parent != -1:
            deaths[current_parent] = max_lambda
            current_parent = parent
            max_lambda = lambda_
        else:
            # Initialize
            current_parent = parent
            max_lambda = lambda_

    return deaths_arr

cdef tuple get_cluster_points(np.ndarray tree, long long cluster, long long num_points):

    cdef list result
    cdef np.ndarray[np.int64_t, ndim=1] result_arr
    cdef np.int64_t *result_ptr
    cdef np.int64_t num_result_points = 0
    cdef np.ndarray[np.int64_t, ndim=1] to_process
    cdef np.ndarray[np.int64_t, ndim=1] next_to_process
    cdef np.int64_t num_to_process
    cdef np.int64_t next_num_to_process
    cdef np.int64_t in_process

    cdef np.ndarray[np.int64_t, ndim=1] children = tree['child']
    cdef np.ndarray[np.int64_t, ndim=1] parents = tree['parent']
    cdef np.ndarray[np.int64_t, ndim=1] sizes = tree['child_size']

    cdef np.int64_t i
    cdef np.int64_t j

    result = []
    #to_process = [cluster]
    #next_to_process = []
    result_arr = np.empty(num_points, dtype=np.int64)
    result_ptr = (<np.int64_t *> result_arr.data)

    to_process = np.empty(num_points, dtype=np.int64)
    to_process[0] = cluster
    num_to_process = 1

    next_to_process = np.empty(num_points, dtype=np.int64)
    next_num_to_process = 0

    while num_to_process > 0:

        for i in range(num_to_process):
            in_process = to_process[i]
            if in_process < num_points:
                result_ptr[num_result_points] = in_process
                num_result_points += 1
            else:
                for j in range(parents.shape[0]):
                    if parents[j] == in_process:
                        next_to_process[next_num_to_process] = children[j]
                        next_num_to_process += 1

        to_process[:next_num_to_process] = next_to_process[:next_num_to_process]
        num_to_process = next_num_to_process
        next_num_to_process = 0

    result = list(result_arr[:num_result_points])

    # deaths = max_lambdas(tree, set([]))

    cluster_split_selection = (parents == cluster) & (sizes > 1)
    if np.sum(cluster_split_selection) > 0:
        max_cluster_lambda = tree[cluster_split_selection]['lambda'][0]
    else:
        max_cluster_lambda = tree['lambda'][result].max()

    #if deaths[cluster] != max_cluster_lambda:
    #    pass
    #    #print cluster, deaths[cluster], max_cluster_lambda
    #prob = tree['lambda'][result]
    #prob = np.where(prob <= max_cluster_lambda, prob, max_cluster_lambda)
    #prob = prob / max_cluster_lambda
    prob = np.ones(len(result))

    return result, prob

cdef class TreeUnionFind (object):

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

cpdef np.ndarray[np.int64_t, ndim=1] do_labelling(np.ndarray tree, set clusters, dict cluster_label_map):

    cdef np.int64_t root_cluster
    cdef np.ndarray[np.int64_t, ndim=1] result_arr
    cdef np.int64_t *result
    cdef TreeUnionFind union_find
    cdef np.int64_t n
    cdef np.int64_t cluster

    root_cluster = tree['parent'].min()
    result_arr = np.empty(root_cluster, dtype=np.int64)
    result = (<np.int64_t *> result_arr.data)

    union_find = TreeUnionFind(tree['parent'].max() + 1)

    for row in tree:
        if row['child'] not in clusters:
            union_find.union_(row['parent'], row['child'])

    for n in range(root_cluster):
        cluster = union_find.find(n)
        if cluster <= root_cluster:
            result[n] = -1
        else:
            result[n] = cluster_label_map[cluster]

    return result_arr

cpdef get_probabilities(np.ndarray tree, dict cluster_map, np.ndarray labels):

    cdef np.ndarray[np.double_t, ndim=1] result
    cdef np.ndarray[np.double_t, ndim=1] deaths
    cdef np.int64_t root_cluster
    cdef np.int64_t point
    cdef np.int64_t cluster_num
    cdef np.int64_t cluster
    cdef np.double_t max_lambda
    cdef np.double_t lambda_

    result = np.zeros(labels.shape[0])
    deaths = max_lambdas(tree)
    root_cluster = tree['parent'].min()

    for row in tree:
        point = row['child']
        if point >= root_cluster:
            continue

        cluster_num = labels[point]

        if cluster_num == -1:
            continue

        cluster = cluster_map[cluster_num]
        max_lambda = deaths[cluster]
        lambda_ = min(row['lambda'], max_lambda)
        result[point] = lambda_ / max_lambda

    return result


cpdef tuple get_clusters(np.ndarray tree, dict stability):
    """
    The tree is assumed to have numeric node ids such that a reverse numeric
    sort is equivalent to a topological sort.
    """
    cdef list node_list
    cdef np.ndarray cluster_tree
    cdef np.ndarray child_selection
    cdef dict is_cluster
    cdef float subtree_stability
    cdef long long node
    cdef long long sub_node
    cdef long long cluster
    cdef long long num_points
    cdef np.ndarray labels

    # Assume clusters are ordered by numeric id equivalent to
    # a topological sort of the tree; This is valid given the
    # current implementation above, so don't change that ... or
    # if you do, change this accordingly!
    node_list = sorted(stability.keys(), reverse=True)[:-1] # (exclude root)
    cluster_tree = tree[tree['child_size'] > 1]
    is_cluster = {cluster:True for cluster in node_list}
    num_points = np.max(tree[tree['child_size'] == 1]['child']) + 1

    for node in node_list:
        child_selection = (cluster_tree['parent'] == node)
        subtree_stability = np.sum([stability[child] for 
                                    child in cluster_tree['child'][child_selection]])
        if subtree_stability > stability[node]:
            is_cluster[node] = False
            stability[node] = subtree_stability
        else:
            for sub_node in bfs_from_cluster_tree(cluster_tree, node):
                if sub_node != node:
                    is_cluster[sub_node] = False

    clusters = set([c for c in is_cluster if is_cluster[c]])
    cluster_map = {c:n for n, c in enumerate(clusters)}
    reverse_cluster_map = {n:c for n, c in enumerate(clusters)}

    labels = do_labelling(tree, clusters, cluster_map)
    probs = get_probabilities(tree, reverse_cluster_map, labels)

    return (labels, probs)
    
    
    
       
