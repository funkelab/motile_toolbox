from itertools import combinations

import networkx as nx
import numpy as np
from tqdm import tqdm

from .graph_attributes import EdgeAttr, NodeAttr
from .graph_from_segmentation import _get_node_id


def compute_ious(frame1: np.ndarray, frame2: np.ndarray) -> dict[int, dict[int, float]]:
    """Compute label IOUs between two label arrays of the same shape. Ignores background
    (label 0).

    Args:
        frame1 (np.ndarray): Array with integer labels
        frame2 (np.ndarray): Array with integer labels

    Returns:
        dict[int, dict[int, float]]: Dictionary from labels in frame 1 to labels in
            frame 2 to iou values. Nodes that have no overlap are not included.
    """
    frame1 = frame1.flatten()
    frame2 = frame2.flatten()
    # get indices where both are not zero (ignore background)
    # this speeds up computation significantly
    non_zero_indices = np.logical_and(frame1, frame2)
    flattened_stacked = np.array([frame1[non_zero_indices], frame2[non_zero_indices]])

    values, counts = np.unique(flattened_stacked, axis=1, return_counts=True)
    frame1_values, frame1_counts = np.unique(frame1, return_counts=True)
    frame1_label_sizes = dict(zip(frame1_values, frame1_counts))
    frame2_values, frame2_counts = np.unique(frame2, return_counts=True)
    frame2_label_sizes = dict(zip(frame2_values, frame2_counts))
    iou_dict: dict[int, dict[int, float]] = {}
    for index in range(values.shape[1]):
        pair = values[:, index]
        intersection = counts[index]
        id1, id2 = pair
        union = frame1_label_sizes[id1] + frame2_label_sizes[id2] - intersection
        if id1 not in iou_dict:
            iou_dict[id1] = {}
        iou_dict[id1][id2] = intersection / union
    return iou_dict


def add_iou(cand_graph: nx.DiGraph, segmentation: np.ndarray, node_frame_dict) -> None:
    """Add IOU to the candidate graph.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with nodes and edges already populated
        segmentation (np.ndarray): segmentation that was used to create cand_graph
    """
    frames = sorted(node_frame_dict.keys())
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        ious = compute_ious(segmentation[frame], segmentation[frame + 1])
        next_nodes = node_frame_dict[frame + 1]
        for node_id in node_frame_dict[frame]:
            node_seg_id = cand_graph.nodes[node_id][NodeAttr.SEG_ID.value]
            for next_id in next_nodes:
                next_seg_id = cand_graph.nodes[next_id][NodeAttr.SEG_ID.value]
                iou = ious.get(node_seg_id, {}).get(next_seg_id, 0)
                cand_graph.edges[(node_id, next_id)][EdgeAttr.IOU.value] = iou


def add_multihypo_iou(
    cand_graph: nx.DiGraph, segmentation: np.ndarray, node_frame_dict
) -> None:
    """Add IOU to the candidate graph for multi-hypothesis segmentations.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with nodes and edges already populated
        segmentation (np.ndarray): Multiple hypothesis segmentation. Dimensions
            are (t, h, [z], y, x), where h is the number of hypotheses.
    """
    frames = sorted(node_frame_dict.keys())
    num_hypotheses = segmentation.shape[1]
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        # construct dictionary of ious between node_ids in frame 1 and frame 2
        ious: dict[str, dict[str, float]] = {}
        for hypo1, hypo2 in combinations(range(num_hypotheses), 2):
            hypo_ious = compute_ious(
                segmentation[frame][hypo1], segmentation[frame + 1][hypo2]
            )
            for segid, intersecting_labels in hypo_ious.items():
                node_id = _get_node_id(frame, segid, hypo1)
                ious[node_id] = {}
                for segid2, iou in intersecting_labels.items():
                    next_id = _get_node_id(frame + 1, segid2, hypo2)
                    ious[node_id][next_id] = iou
        next_nodes = node_frame_dict[frame + 1]
        for node_id in node_frame_dict[frame]:
            for next_id in next_nodes:
                iou = ious.get(node_id, {}).get(next_id, 0)
                cand_graph.edges[(node_id, next_id)][EdgeAttr.IOU.value] = iou
