import logging
from typing import Any

import networkx as nx
import numpy as np

from .conflict_sets import compute_conflict_sets
from .iou import add_iou, add_multihypo_iou
from .utils import add_cand_edges, nodes_from_segmentation

logger = logging.getLogger(__name__)


def get_candidate_graph(
    segmentation: np.ndarray,
    max_edge_distance: float,
    iou: bool = False,
    multihypo: bool = False,
) -> tuple[nx.DiGraph, list[set[Any]] | None]:
    """Construct a candidate graph from a segmentation array. Nodes are placed at the
    centroid of each segmentation and edges are added for all nodes in adjacent frames
    within max_edge_distance. If segmentation contains multiple hypotheses, will also
    return a list of conflicting node ids that cannot be selected together.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, [h], [z], y, x), where h is the number of hypotheses.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes with centroids within this distance in adjacent frames
            will by connected with a candidate edge.
        iou (bool, optional): Whether to include IOU on the candidate graph.
            Defaults to False.
        multihypo (bool, optional): Whether the segmentation contains multiple
            hypotheses. Defaults to False.

    Returns:
        tuple[nx.DiGraph, list[set[Any]] | None]: A candidate graph that can be passed
        to the motile solver, and a list of conflicting node ids.
    """
    # add nodes
    if multihypo:
        cand_graph = nx.DiGraph()
        num_frames = segmentation.shape[0]
        node_frame_dict = {t: [] for t in range(num_frames)}
        num_hypotheses = segmentation.shape[1]
        for hypo_id in range(num_hypotheses):
            hypothesis = segmentation[:, hypo_id]
            node_graph, frame_dict = nodes_from_segmentation(
                hypothesis, hypo_id=hypo_id
            )
            cand_graph.update(node_graph)
            for t in range(num_frames):
                if t in frame_dict:
                    node_frame_dict[t].extend(frame_dict[t])
    else:
        cand_graph, node_frame_dict = nodes_from_segmentation(segmentation)
    logger.info(f"Candidate nodes: {cand_graph.number_of_nodes()}")

    # add edges
    add_cand_edges(
        cand_graph,
        max_edge_distance=max_edge_distance,
        node_frame_dict=node_frame_dict,
    )
    if iou:
        if multihypo:
            add_multihypo_iou(cand_graph, segmentation, node_frame_dict)
        else:
            add_iou(cand_graph, segmentation, node_frame_dict)

    logger.info(f"Candidate edges: {cand_graph.number_of_edges()}")

    # Compute conflict sets between segmentations
    if multihypo:
        conflicts = []
        for time, segs in enumerate(segmentation):
            conflicts.extend(compute_conflict_sets(segs, time))
    else:
        conflicts = None

    return cand_graph, conflicts
