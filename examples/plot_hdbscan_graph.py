import numpy
import numpy as np
import sklearn.metrics
from scipy import sparse
import igraph
import networkx as nx
import time
from hdbscan import (
    HDBSCAN,
    hdbscan,
    validity_index,
    approximate_predict,
    approximate_predict_scores,
    membership_vector,
    all_points_membership_vectors,
)


def create_distance_matrix(graph):
    """
    Creates a distance matrix from the given graph using the igraph shortest path algorithm.
    :param graph: An igraph graph object.
    :return: Scipy csr matrix based on the graph.
    """

    # create a distance matrix based of the graph
    # create variables
    path_weight, vertex_from_list, vertex_to_list, vertex_from = [], [], [], 0

    for vertex in graph.vs:
        list_edges_shortest_path = graph.get_shortest_paths(vertex, to=None, weights="weight", mode='out',
                                                            output="epath")
        vertex_to = 0

        for edge_list in list_edges_shortest_path:
            if edge_list:
                vertex_from_list.append(vertex_from)
                vertex_to_list.append(vertex_to)
                path_weight.append(sum(graph.es.select(edge_list)["weight"]))
            else:
                vertex_from_list.append(vertex_from)
                vertex_to_list.append(vertex_to)
                path_weight.append(0)

            vertex_to += 1
        vertex_from += 1

    distance_matrix = sparse.csr_matrix((path_weight, (vertex_from_list, vertex_to_list)))

    return distance_matrix


def hdbscan_graph():
    """
    Creates a weighted stochastic_block_model graph to compare the newly created graph function of HDBSCAN
    to the precomputed metric using a distance matrix created for the graph.
    """
    # measure time
    start_build_graph = time.time()

    # set parameters graph and edges
    number_communities = 4
    edge_weight_in_community = 0.1
    edge_weight_out_community = 1

    # create graph
    community_sizes = np.random.randint(low=30, high=70, size=number_communities)
    matrix_prob = np.random.rand(number_communities, number_communities)
    matrix_prob = (np.tril(matrix_prob) + np.tril(matrix_prob, -1).T) * 0.5
    numpy.fill_diagonal(matrix_prob, 0.7)
    sbm_graph = nx.stochastic_block_model(community_sizes, matrix_prob, seed=0)

    # convert to igraph object
    graph = igraph.Graph(n=sbm_graph.number_of_nodes(), directed=False)
    graph.add_edges(sbm_graph.edges())

    # check for double edges and loops and delete those
    graph.simplify()
    graph.vs.select(_degree=0).delete()

    # run community detection to assign edge weights
    community_detection = graph.community_multilevel()

    # add edge weights
    weight_list = []
    for edge in graph.es:
        vertex_1 = edge.source
        vertex_2 = edge.target
        edge_weight_added = False
        for subgraph in community_detection:
            if vertex_1 in subgraph and vertex_2 in subgraph:
                weight_list.append(edge_weight_in_community)
                edge_weight_added = True
        if not edge_weight_added:
            weight_list.append(edge_weight_out_community)

    graph.es["weight"] = weight_list

    print("Graph created:", time.time() - start_build_graph)

    # run HDBSCAN on graph distance matrix
    start_distance_matrix = time.time()

    # create a distance matrix from the graph
    distance_matrix = create_distance_matrix(graph)

    # run HDBSCAN on the created distance matrix
    clusterer = HDBSCAN(metric="precomputed").fit(distance_matrix)
    labels_distance_matrix = clusterer.labels_

    # measure time
    print("HDBSCAN distance matrix:", time.time() - start_distance_matrix)

    # plot graph clustering using iGraph
    graph.vs["label_distance_matrix"] = labels_distance_matrix
    vclustering = igraph.clustering.VertexClustering.FromAttribute(graph, "label_distance_matrix")
    igraph.plot(vclustering)

    """
    Convert the iGraph graph into a csr sparse matrix, which the modified HDBSCAN function accepts and 
    transforms into a scipy csgraph. 
    """
    # run HDBSCAN using the graph metric
    start_hdbscan_graph = time.time()

    # create adjacency matrix from the graph, csr sparse matrix format
    adjacency = graph.get_adjacency_sparse(attribute="weight")

    clusterer = HDBSCAN(metric="graph").fit(adjacency)
    labels_hdbscan_graph = clusterer.labels_

    print("HDBSCAN graph:", time.time() - start_hdbscan_graph)

    # plot clustering labels using iGraph
    graph.vs["label_hdbscan_graph"] = labels_hdbscan_graph
    vclustering = igraph.clustering.VertexClustering.FromAttribute(graph, "label_hdbscan_graph")
    igraph.plot(vclustering)

    # print the AMI and ARI for the labels
    print("AMI:", sklearn.metrics.adjusted_mutual_info_score(labels_distance_matrix, labels_hdbscan_graph))
    print("ARI:", sklearn.metrics.adjusted_rand_score(labels_distance_matrix, labels_hdbscan_graph))


"""
run the example function displaying the graph feature of HDBSCAN
"""
hdbscan_graph()
