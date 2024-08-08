import networkx as nx
import numpy as np

from motile_toolbox.candidate_graph import NodeAttr


def assign_tracklet_ids(graph: nx.DiGraph) -> nx.DiGraph:
    """Add a tracklet_id attribute to a graph by removing division edges,
    assigning one id to each connected component.
    Designed as a helper for visualizing the graph in the napari Tracks layer.

    Args:
        graph (nx.DiGraph): A networkx graph with a tracking solution

    Returns:
        nx.DiGraph: The same graph with the tracklet_id assigned. Probably
        occurrs in place but returned just to be clear.
    """
    graph_copy = graph.copy()

    parents = [node for node, degree in graph.out_degree() if degree >= 2]
    intertrack_edges = []

    # Remove all intertrack edges from a copy of the original graph
    for parent in parents:
        daughters = [child for p, child in graph.out_edges(parent)]
        for daughter in daughters:
            graph_copy.remove_edge(parent, daughter)
            intertrack_edges.append((parent, daughter))

    track_id = 1
    for tracklet in nx.weakly_connected_components(graph_copy):
        nx.set_node_attributes(
            graph, {node: {"tracklet_id": track_id} for node in tracklet}
        )
        track_id += 1
    return graph, intertrack_edges


def _get_location(graph, node, location_key):
    if isinstance(location_key, list) or isinstance(location_key, tuple):
        return [graph.nodes[node][dim] for dim in location_key]
    else:
        return graph.nodes[node][location_key]


def to_napari_tracks_layer(
    graph, frame_key=NodeAttr.TIME.value, location_key=NodeAttr.POS.value, properties=()
):
    """Function to take a networkx graph and return the data needed to add to
    a napari tracks layer.

    Args:
        graph (nx.DiGraph): _description_
        frame_key (str, optional): Key in graph attributes containing time frame.
            Defaults to NodeAttr.TIME.value.
        location_key (str, optional): Key in graph node attributes containing
            location. Defaults to NodeAttr.POS.value.
        properties (tuple, optional): Keys in graph node attributes to add
            to the visualization layer. Defaults to (). NOTE: not working now :(

    Returns:
        data : array (N, D+1)
            Coordinates for N points in D+1 dimensions. ID,T,(Z),Y,X. The first
            axis is the integer ID of the track. D is either 3 or 4 for planar
            or volumetric timeseries respectively.
        properties : dict {str: array (N,)}
            Properties for each point. Each property should be an array of length N,
            where N is the number of points.
        graph : dict {int: list}
            Graph representing associations between tracks. Dictionary defines the
            mapping between a track ID and the parents of the track. This can be
            one (the track has one parent, and the parent has >=1 child) in the
            case of track splitting, or more than one (the track has multiple
            parents, but only one child) in the case of track merging.
    """
    ndim = len(_get_location(graph, next(iter(graph.nodes)), location_key))
    napari_data = np.zeros((graph.number_of_nodes(), ndim + 2))
    napari_properties = {prop: np.zeros(graph.number_of_nodes()) for prop in properties}
    napari_edges = {}
    graph, intertrack_edges = assign_tracklet_ids(graph)
    for index, node in enumerate(graph.nodes(data=True)):
        node_id, data = node
        location = _get_location(graph, node_id, location_key)
        napari_data[index] = [data["tracklet_id"], data[frame_key], *location]
        for prop in properties:
            if prop in data:
                napari_properties[prop][index] = data[prop]
    napari_edges = {}
    for parent, child in intertrack_edges:
        parent_track_id = graph.nodes[parent]["tracklet_id"]
        child_track_id = graph.nodes[child]["tracklet_id"]
        if child_track_id in napari_edges:
            napari_edges[child_track_id].append(parent_track_id)
        else:
            napari_edges[child_track_id] = [parent_track_id]
    return napari_data, napari_properties, napari_edges
