import networkx as nx
from skimage.measure import regionprops
import numpy as np
from typing import Iterable
from tqdm import tqdm
import logging
import math

logger = logging.getLogger(__name__)


def get_location(node_data, loc_keys=("z", "y", "x")):
    return [node_data[k] for k in loc_keys]


def graph_from_segmentation(
    segmentation: np.ndarray,
    max_edge_distance: float,
    attributes: tuple[str, ...] | list[str] = ("distance",),
    position_keys: tuple[str, ...] | list[str] = ("y", "x"),
    frame_key: str = "t",
):
    """Construct a candidate graph from a segmentation array. Nodes are placed at the centroid
    of each segmentation and edges are added for all nodes in adjacent frames within
    max_edge_distance. The specified attributes are computed during construction.
    Node ids are strings with format "{time}_{label id}".

    Args:
        segmentation (np.ndarray): A 3 or 4 dimensional numpy array with integer labels
            (0 is background, all pixels with value 1 belong to one cell, etc.).
            The time dimension is first, followed by two or three position dimensions.
            If the position dims are not (y, x), use `position_keys` to specify the names of
            the dimensions.
        max_edge_distance (float): Maximum distance that objects can travel between frames. All
            nodes within this distance in adjacent frames will by connected with a candidate edge.
        attributes (tuple[str, ...], optional): Set of attributes to compute and add to graph.
            Valid attributes are: "distance". Defaults to ("distance",).
        position_keys (tuple[str, ...], optional): What to label the position dimensions in the
            candidate graph. The order of the names corresponds to the order of the dimensions
            in `segmentation`. Defaults to ("y", "x").
        frame_key (str, optional): What to label the time dimension in the candidate graph.
            Defaults to 't'.

    Returns:
        nx.DiGraph: A candidate graph that can be passed to the motile solver.

    Raises:
        ValueError: if unsupported attribute strings are passed in to the attributes argument,
            or if the number of position keys provided does not match the number of position dimensions.
    """
    valid_attributes = ["distance"]
    for attr in attributes:
        if attr not in valid_attributes:
            raise ValueError(
                f"Invalid attribute {attr} (supported attributes: {valid_attributes})"
            )
    if len(position_keys) != segmentation.ndim - 1:
        raise ValueError(
            f"Position labels {position_keys} does not match number of spatial dims ({segmentation.ndim - 1})"
        )
    # add nodes
    node_frame_dict = (
        {}
    )  # construct a dictionary from time frame to node_id for efficiency
    cand_graph = nx.DiGraph()

    for t in range(len(segmentation)):
        nodes_in_frame = []
        props = regionprops(segmentation[t])
        for i, regionprop in enumerate(props):
            node_id = f"{t}_{regionprop.label}"
            attrs = {
                frame_key: t,
                "segmentation_id": regionprop.label,
            }
            centroid = regionprop.centroid  # [z,] y, x
            for label, value in zip(position_keys, centroid):
                attrs[label] = value
            cand_graph.add_node(node_id, **attrs)
            nodes_in_frame.append(node_id)
        node_frame_dict[t] = nodes_in_frame

    logger.info(f"Candidate nodes: {cand_graph.number_of_nodes()}")

    # add edges
    frames = sorted(node_frame_dict.keys())
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_nodes = node_frame_dict[frame + 1]
        next_locs = [
            get_location(cand_graph.nodes[n], loc_keys=position_keys)
            for n in next_nodes
        ]
        for node in node_frame_dict[frame]:
            loc = get_location(cand_graph.nodes[node], loc_keys=position_keys)
            for next_id, next_loc in zip(next_nodes, next_locs):
                dist = math.dist(next_loc, loc)
                attrs = {}
                if "distance" in attributes:
                    attrs["distance"] = dist
                if dist < max_edge_distance:
                    cand_graph.add_edge(node, next_id, **attrs)

    logger.info(f"Candidate edges: {cand_graph.number_of_edges()}")
    return cand_graph
