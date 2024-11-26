import networkx as nx
import numpy as np

from motile_toolbox.candidate_graph import NodeAttr


def relabel_segmentation(
    solution_nx_graph: nx.DiGraph,
    segmentation: np.ndarray,
) -> np.ndarray:
    """Relabel a segmentation based on tracking results so that nodes in same
    track share the same id. IDs do change at division.

    Args:
        solution_nx_graph (nx.DiGraph): Networkx graph with the solution to use
            for relabeling. Nodes not in graph will be removed from seg. Original
            segmentation ids have to be stored in the graph so we
            can map them back.
        segmentation (np.ndarray): Original segmentation with dimensions (t, [z], y, x)

    Returns:
        np.ndarray: Relabeled segmentation array where nodes in same track share same
            id with shape (t,[z],y,x)
    """
    tracked_masks = np.zeros_like(segmentation)
    id_counter = 1
    parent_nodes = [n for (n, d) in solution_nx_graph.out_degree() if d > 1]
    soln_copy = solution_nx_graph.copy()
    for parent_node in parent_nodes:
        out_edges = solution_nx_graph.out_edges(parent_node)
        soln_copy.remove_edges_from(out_edges)
    for node_set in nx.weakly_connected_components(soln_copy):
        for node in node_set:
            time_frame = solution_nx_graph.nodes[node][NodeAttr.TIME.value]
            previous_seg_id = solution_nx_graph.nodes[node][NodeAttr.SEG_ID.value]
            previous_seg_mask = segmentation[time_frame] == previous_seg_id
            tracked_masks[time_frame][previous_seg_mask] = id_counter
        id_counter += 1
    return tracked_masks
