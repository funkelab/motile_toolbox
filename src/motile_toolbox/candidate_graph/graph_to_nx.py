import networkx as nx
from motile import TrackGraph


def graph_to_nx(graph: TrackGraph, flatten_hyperedges=True) -> nx.DiGraph:
    """Convert a motile TrackGraph into a networkx DiGraph.

    Args:
        graph (TrackGraph): TrackGraph to be converted to networkx
        flatten_hyperedges (bool, optional): If True, include one edge for each
            (source, target) combo in a hyperedge. If False, introduce a new
            hypernode to represent hyperedges. Defaults to True.

    Returns:
        nx.DiGraph: Directed networkx graph with same nodes, edges, and attributes.
    """
    nx_graph = nx.DiGraph()
    nodes_list = list(graph.nodes.items())
    nx_graph.add_nodes_from(nodes_list)
    edges_list = []
    for edge, data in graph.edges.items():
        if graph.is_hyperedge(edge):
            us, vs = edge
            if flatten_hyperedges:
                # flatten the hyperedges into multiple normal edges
                for u in us:
                    for v in vs:
                        edges_list.append((u, v, data))
            else:
                # add a hypernode to connect all in nodes with all out nodes
                hypernode_id = "_".join(list(map(str, us)) + list(map(str, vs)))
                for u in us:
                    edges_list.append((u, hypernode_id, data))
                for v in vs:
                    edges_list.append((hypernode_id, v, data))
        else:
            u, v = edge
            edges_list.append((u, v, data))

    nx_graph.add_edges_from(edges_list)
    return nx_graph
