#cython: boundscheck=False, nonecheck=False, initializedcheck=False
# Utility routines in cython for prediction in hdbscan
# Authors: Leland McInnes
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from dist_metrics cimport DistanceMetric

from libc.float cimport DBL_MAX

cpdef get_tree_row_with_child(np.ndarray tree, np.intp_t child):

    cdef int i
    cdef np.ndarray[np.intp_t, ndim = 1] child_array = tree['child']

    for i in range(tree.shape[0]):
        if child_array[i] == child:
            return tree[i]

    return tree[0]

cdef np.float64_t min_dist_to_exemplar(
                        np.ndarray[np.float64_t, ndim=1] point,
                        np.ndarray[np.float64_t, ndim=2] cluster_exemplars,
                        DistanceMetric dist_metric):

    cdef np.intp_t i
    cdef np.float64_t result = DBL_MAX
    cdef np.float64_t distance
    cdef np.float64_t *point_ptr = (<np.float64_t *> point.data)
    cdef np.float64_t[:, ::1] exemplars_view = \
        (<np.float64_t [:cluster_exemplars.shape[0], :cluster_exemplars.shape[1]:1]> (<np.float64_t *> cluster_exemplars.data))
    cdef np.float64_t *exemplars_ptr = \
        (<np.float64_t *> &exemplars_view[0, 0])
    cdef np.intp_t num_features = point.shape[0]

    for i in range(cluster_exemplars.shape[0]):
        distance = dist_metric.dist(point_ptr,
                                    &cluster_exemplars_ptr[num_features * i],
                                    num_features)
        if distance < result:
            result = distance

    return result

cdef np.ndarray[np.float64_t, ndim=1] dist_vector(
                    np.ndarray[np.float64_t, ndim=1] point,
                    list exemplars_list):

    cdef np.intp_t i
    cdef np.ndarray[np.float64_t, ndim=2] exemplars
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(len(exemplars_list))


    for i in range(len(exemplars_list)):
        exemplars = exemplars_list[i]
        result[i] = min_dist_to_exemplar(point, exemplars)

    return result

cpdef np.ndarray[np.float64_t, ndim=1] dist_membership_vector(
                    np.ndarray[np.float64_t, ndim=1] point,
                    list exemplars_list,
                    softmax=False):

    cdef np.intp_t i
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(len(exemplars_list))
    cdef np.ndarray[np.float64_t, ndim=1] vector
    cdef np.float64_t sum

    vector = dist_vector(point, exemplars_list)

    if softmax:
        for i in range(vector.shape[0]):
            if vector[i] != 0:
                result[i] = np.exp(1.0 / vector[i])
            else:
                result[i] = DBL_MAX / vector.shape[0]
            sum += result[i]

    else:
        for i in range(vector.shape[0]):
            if vector[i] != 0:
                result[i] = 1.0 / vector[i]
            else:
                result[i] = DBL_MAX / vector.shape[0]
            sum += result[i]

    for i in range(result.shape[0]):
        result[i] = result[i] / sum

    return result