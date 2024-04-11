from itertools import product
from typing import Any

import networkx as nx
import numpy as np
from tqdm import tqdm

from .graph_attributes import EdgeAttr
from .utils import _compute_node_frame_dict, get_node_id


def _compute_ious(
    frame1: np.ndarray, frame2: np.ndarray
) -> list[tuple[int, int, float]]:
    """Compute label IOUs between two label arrays of the same shape. Ignores background
    (label 0).

    Args:
        frame1 (np.ndarray): Array with integer labels
        frame2 (np.ndarray): Array with integer labels

    Returns:
        list[tuple[int, int, float]]: List of tuples of label in frame 1, label in
            frame 2, and iou values. Labels that have no overlap are not included.
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
    ious: list[tuple[int, int, float]] = []
    for index in range(values.shape[1]):
        pair = values[:, index]
        intersection = counts[index]
        id1, id2 = pair
        union = frame1_label_sizes[id1] + frame2_label_sizes[id2] - intersection
        ious.append((id1, id2, intersection / union))
    return ious


def _get_iou_dict(segmentation) -> dict[str, dict[str, float]]:
    """Get all ious values for the provided segmentation (all frames).
    Will return as map from node_id -> dict[node_id] -> iou for easy
    navigation when adding to candidate graph.

    Args:
        segmentation (np.ndarray): Segmentation that was used to create cand_graph.
            Has shape (t, h, [z], y, x), where h is the number of hypotheses.

    Returns:
        dict[str, dict[str, float]]: A map from node id to another dictionary, which
            contains node_ids to iou values.
    """
    iou_dict: dict[str, dict[str, float]] = {}
    hypo_pairs: list[tuple[int | None, ...]]
    num_hypotheses = segmentation.shape[1]
    if num_hypotheses > 1:
        hypo_pairs = list(product(range(num_hypotheses), repeat=2))
    else:
        hypo_pairs = [(None, None)]

    for frame in range(len(segmentation) - 1):
        for hypo1, hypo2 in hypo_pairs:
            seg1 = segmentation[frame][hypo1]
            seg2 = segmentation[frame + 1][hypo2]
            ious = _compute_ious(seg1, seg2)
            for label1, label2, iou in ious:
                node_id1 = get_node_id(frame, label1, hypo1)
                if node_id1 not in iou_dict:
                    iou_dict[node_id1] = {}
                node_id2 = get_node_id(frame + 1, label2, hypo2)
                iou_dict[node_id1][node_id2] = iou
    return iou_dict


def add_iou(
    cand_graph: nx.DiGraph,
    segmentation: np.ndarray,
    node_frame_dict: dict[int, list[Any]] | None = None,
) -> None:
    """Add IOU to the candidate graph.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with nodes and edges already populated
        segmentation (np.ndarray): segmentation that was used to create cand_graph.
            Has shape (t, h, [z], y, x), where h is the number of hypotheses.
        node_frame_dict(dict[int, list[Any]] | None, optional): A mapping from
            time frames to nodes in that frame. Will be computed if not provided,
            but can be provided for efficiency (e.g. after running
            nodes_from_segmentation). Defaults to None.
    """
    if node_frame_dict is None:
        node_frame_dict = _compute_node_frame_dict(cand_graph)
    frames = sorted(node_frame_dict.keys())
    ious = _get_iou_dict(segmentation)
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict.keys():
            continue
        next_nodes = node_frame_dict[frame + 1]
        for node_id in node_frame_dict[frame]:
            for next_id in next_nodes:
                iou = ious.get(node_id, {}).get(next_id, 0)
                if (node_id, next_id) in cand_graph.edges:
                    cand_graph.edges[(node_id, next_id)][EdgeAttr.IOU.value] = iou
