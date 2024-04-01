import logging
import math
from typing import Any

import networkx as nx
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

from .graph_attributes import EdgeAttr, NodeAttr
from .iou import add_iou

logger = logging.getLogger(__name__)


def _get_node_id(time: int, label_id: int, hypothesis_id: int | None = None) -> str:

    if hypothesis_id:
        return f"{time}_{hypothesis_id}_{label_id}"
    else:
        return f"{time}_{label_id}"


def nodes_from_segmentation(
    segmentation: np.ndarray,
) -> tuple[nx.DiGraph, dict[int, list[Any]]]:
    """Extract candidate nodes from a segmentation. Also computes specified attributes.
    Returns a networkx graph with only nodes, and also a dictionary from frames to
    node_ids for efficient edge adding.

    Args:
        segmentation (np.ndarray): A 3 or 4 dimensional numpy array with integer labels
            (0 is background, all pixels with value 1 belong to one cell, etc.). The
            time dimension is first, followed by two or three position dimensions.

    Returns:
        tuple[nx.DiGraph, dict[int, list[Any]]]: A candidate graph with only nodes,
            and a mapping from time frames to node ids.
    """
    cand_graph = nx.DiGraph()
    # also construct a dictionary from time frame to node_id for efficiency
    node_frame_dict = {}
    print("Extracting nodes from segmentaiton")
    for t in tqdm(range(len(segmentation))):
        nodes_in_frame = []
        props = regionprops(segmentation[t])
        for regionprop in props:
            node_id = _get_node_id(t, regionprop.label)
            attrs = {
                NodeAttr.TIME.value: t,
            }
            attrs[NodeAttr.SEG_ID.value] = regionprop.label
            centroid = regionprop.centroid  # [z,] y, x
            attrs[NodeAttr.POS.value] = centroid
            cand_graph.add_node(node_id, **attrs)
            nodes_in_frame.append(node_id)
        if nodes_in_frame:
            node_frame_dict[t] = nodes_in_frame
    return cand_graph, node_frame_dict


def _compute_node_frame_dict(
    cand_graph: nx.DiGraph
) -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (nx.DiGraph): A networkx graph

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        t = data[NodeAttr.TIME.value]
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node)
    return node_frame_dict


def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    node_frame_dict: None | dict[int, list[Any]] = None,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """
    print("Extracting candidate edges")
    if not node_frame_dict:
        node_frame_dict = _compute_node_frame_dict(cand_graph)

    frames = sorted(node_frame_dict.keys())
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_nodes = node_frame_dict[frame + 1]
        next_locs = [cand_graph.nodes[n][NodeAttr.POS.value] for n in next_nodes]
        for node in node_frame_dict[frame]:
            loc = cand_graph.nodes[node][NodeAttr.POS.value]
            for next_id, next_loc in zip(next_nodes, next_locs):
                dist = math.dist(next_loc, loc)
                if dist <= max_edge_distance:
                    attrs = {EdgeAttr.DISTANCE.value: dist}
                    cand_graph.add_edge(node, next_id, **attrs)


def graph_from_segmentation(
    segmentation: np.ndarray,
    max_edge_distance: float,
    iou: bool = False,
) -> nx.DiGraph:
    """Construct a candidate graph from a segmentation array. Nodes are placed at the
    centroid of each segmentation and edges are added for all nodes in adjacent frames
    within max_edge_distance. The specified attributes are computed during construction.
    Node ids are strings with format "{time}_{label id}".

    Args:
        segmentation (np.ndarray): A 3 or 4 dimensional numpy array with integer labels
            (0 is background, all pixels with value 1 belong to one cell, etc.). The
            time dimension is first, followed by two or three position dimensions. If
            the position dims are not (y, x), use `position_keys` to specify the names
            of the dimensions.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        iou (bool, optional): Whether to include IOU on the candidate graph.
            Defaults to False.

    Returns:
        nx.DiGraph: A candidate graph that can be passed to the motile solver.
    """
    # add nodes
    cand_graph, node_frame_dict = nodes_from_segmentation(segmentation)
    logger.info(f"Candidate nodes: {cand_graph.number_of_nodes()}")

    # add edges
    add_cand_edges(
        cand_graph,
        max_edge_distance=max_edge_distance,
        node_frame_dict=node_frame_dict,
    )
    if iou:
        add_iou(cand_graph, segmentation, node_frame_dict)

    logger.info(f"Candidate edges: {cand_graph.number_of_edges()}")
    return cand_graph
