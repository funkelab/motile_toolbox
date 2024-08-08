import networkx as nx
from motile import TrackGraph


def graph_to_nx(graph: TrackGraph) -> nx.DiGraph:
    """Convert a motile TrackGraph into a networkx DiGraph. Hyper edges
    are "flattened" into individual edges and the attributes from the hyper edge
    are put on all individual edges.

    Args:
        graph (TrackGraph): TrackGraph to be converted to networkx

    Returns:
        nx.DiGraph: Directed networkx graph with same nodes, edges, and attributes.
    """
    nx_graph = nx.DiGraph()
    nodes_list = list(graph.nodes.items())
    nx_graph.add_nodes_from(nodes_list)
    edges_list = []
    for edge_id, data in graph.edges.items():
        if graph.is_hyperedge(edge_id):
            for source in edge_id[0]:
                for target in edge_id[1]:
                    edges_list.append((source, target, data))
        else:
            edges_list.append((edge_id[0], edge_id[1], data))
    nx_graph.add_edges_from(edges_list)
    return nx_graph
