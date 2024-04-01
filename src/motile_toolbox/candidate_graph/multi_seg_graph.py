import numpy as np
import networkx as nx
from typing import Any

from .graph_attributes import EdgeAttr, NodeAttr, add_iou
from .graph_from_segmentation import nodes_from_segmentation, add_cand_edges


def compute_multi_seg_graph(segmentations: list[np.ndarray]) -> tuple[nx.DiGraph, list[set]]:
    """Create a candidate graph from multi hypothesis segmentations. This is not 
    tailored for agglomeration approaches with hierarchical merge graphs, it simply 
    creates a conflict set for any nodes that overlap in the same time frame.

    Args:
        segmentations (list[np.ndarray]): 

    Returns:
        nx.DiGraph: _description_
    """
    # for each segmentation, get nodes using same method as graph_from_segmentation
    # add them all to one big graph
    cand_graph, frame_dict = nodes_from_multi_segmentation(segmentations) # TODO: other args

    # Compute conflict sets between segmentations
    # can use same method as IOU (without the U) to compute conflict sets
    conflicts = []
    for time, segs in enumerate(segmentations):
        conflicts.append(compute_conflict_sets(segs, time))

    # add edges with same method as before, with slightly different implementation
    add_cand_edges(cand_graph) # TODO: other args
    if EdgeAttr.IOU in edge_attributes:
        # TODO: cross product when calling (need to re-organize add_iou to not assume stuff)
        add_iou(cand_graph, segmentation)

    return cand_graph
        
    




def nodes_from_multi_segmentation(
    segmentations: list[np.ndarray],
    attributes: tuple[NodeAttr, ...] | list[NodeAttr] = (NodeAttr.SEG_ID,),
    position_keys: tuple[str, ...] | list[str] = ("y", "x"),
    frame_key: str = "t",
) -> tuple[nx.DiGraph, dict[int, list[Any]]]:
    multi_hypo_node_graph = nx.DiGraph()
    multi_frame_dict = {}
    for layer_id, segmentation in enumerate(segmentations):
        node_graph, frame_dict = nodes_from_segmentation(segmentation, layer_id)
        # TODO: pass attributes, etc.
        # TODO: add multi segmentation attribute to nodes_from_segmentation
        #   (use in node id and add to attributes)
        multi_hypo_node_graph.update(node_graph)
        multi_frame_dict.update(frame_dict)
        # TODO: Make sure there is no node-id collision

    return multi_hypo_node_graph, multi_frame_dict



def compute_conflict_sets(segmenations: np.ndarray, time: int) -> list[set]:
    """ Segmentations in one frame only. Return list of sets of node ids that conflict."""
    # This will look a lot like the IOU code
    pass
