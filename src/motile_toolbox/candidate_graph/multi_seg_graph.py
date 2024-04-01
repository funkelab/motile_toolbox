
from itertools import combinations

import networkx as nx
import numpy as np

from .graph_from_segmentation import (
    _get_node_id,
    add_cand_edges,
    nodes_from_segmentation,
)
from .iou import add_multihypo_iou


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
        hypothesis = segmentation[:,hypo_id]
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


def compute_conflict_sets(segmentation_frame: np.ndarray, time: int) -> list[set]:
    """Segmentation in one frame only. Return

    Args:
        segmentation_frame (np.ndarray):  One frame of the multiple hypothesis
            segmentation. Dimensions are (h, [z], y, x), where h is the number of
            hypotheses.
        time (int): Time frame, for computing node_ids.

    Returns:
        list[set]: list of sets of node ids that overlap
    """
    flattened_segs = [seg.flatten() for seg in segmentation_frame]

    # get locations where at least two hypotheses have labels
    # This approach may be inefficient, but likely doesn't matter compared to np.unique
    conflict_indices = np.zeros(flattened_segs[0].shape, dtype=bool)
    for seg1, seg2 in combinations(flattened_segs, 2):
        non_zero_indices = np.logical_and(seg1, seg2)
        conflict_indices = np.logical_or(conflict_indices, non_zero_indices)

    flattened_stacked = np.array([seg[conflict_indices] for seg in flattened_segs])
    values = np.unique(flattened_stacked, axis=1)

    conflict_sets = []
    for conflicting_labels in values:
        id_set = set()
        for hypo_id, label in enumerate(conflicting_labels):
            if label != 0:
                id_set.add(_get_node_id(time, label, hypo_id))
            conflict_sets.append(id_set)
    return conflict_sets
