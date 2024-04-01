import logging

import networkx as nx
import numpy as np

from .conflict_sets import compute_conflict_sets
from .iou import add_iou, add_multihypo_iou
from .utils import add_cand_edges, nodes_from_segmentation

logger = logging.getLogger(__name__)


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


def compute_multi_seg_graph(
    segmentation: np.ndarray,
    max_edge_distance: float,
    iou: bool = False,
) -> tuple[nx.DiGraph, list[set]]:
    """Create a candidate graph from multi hypothesis segmentation. This is not
    tailored for agglomeration approaches with hierarchical merge graphs, it simply
    creates a conflict set for any nodes that overlap in the same time frame.

    Args:
        segmentations (np.ndarray): Multiple hypothesis segmentation. Dimensions
            are (t, h, [z], y, x), where h is the number of hypotheses.

    Returns:
        nx.DiGraph: _description_
    """
    # for each segmentation, get nodes using same method as graph_from_segmentation
    # add them all to one big graph
    cand_graph = nx.DiGraph()
    node_frame_dict = {}
    num_hypotheses = segmentation.shape[1]
    for hypo_id in range(num_hypotheses):
        hypothesis = segmentation[:, hypo_id]
        node_graph, frame_dict = nodes_from_segmentation(hypothesis, hypo_id=hypo_id)
        cand_graph.update(node_graph)
        node_frame_dict.update(frame_dict)

    # Compute conflict sets between segmentations
    # can use same method as IOU (without the U) to compute conflict sets
    conflicts = []
    for time, segs in enumerate(segmentation):
        conflicts.extend(compute_conflict_sets(segs, time))

    # add edges with same method as before, with slightly different implementation
    add_cand_edges(cand_graph, max_edge_distance, node_frame_dict)
    if iou:
        add_multihypo_iou(cand_graph, segmentation, node_frame_dict)

    return cand_graph, conflicts
