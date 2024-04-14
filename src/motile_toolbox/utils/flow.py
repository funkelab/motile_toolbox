import math

import networkx as nx
import numpy as np
from skimage.registration import phase_cross_correlation

from motile_toolbox.candidate_graph.graph_attributes import EdgeAttr, NodeAttr


def compute_pcc_flow(candidate_graph: nx.DiGraph, images: np.ndarray):
    """This calculates the flow using phase cross correlation
    for the image cropped around an object
    at `t` and the same region of interest at `t+1`,
    and updates the `NodeAttr.FLOW`.

    Args:
        candidate_graph (nx.DiGraph): Existing candidate graph with nodes.

        images (np.ndarray): Raw images (t, c, [z], y, x).

    """
    for node in candidate_graph.nodes(data=True):
        frame = node[1][NodeAttr.TIME.value]
        if frame + 1 >= len(images):
            continue
        loc = node[1][NodeAttr.POS.value]
        bbox = node[1][NodeAttr.BBOX.value]
        if len(loc) == 2:
            reference_image = images[frame][
                0, bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
            ]
            shifted_image = images[frame + 1][
                0, bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
            ]
        elif len(loc) == 3:
            reference_image = (
                images[frame][
                    0,
                    bbox[0] : bbox[3] + 1,
                    bbox[1] : bbox[4] + 1,
                    bbox[2] : bbox[5] + 1,
                ],
            )
            shifted_image = images[frame + 1][
                0,
                bbox[0] : bbox[3] + 1,
                bbox[1] : bbox[4] + 1,
                bbox[2] : bbox[5] + 1,
            ]
        shift, _, _ = phase_cross_correlation(reference_image, shifted_image)
        node[1][NodeAttr.FLOW.value] = shift


def correct_edge_distance(candidate_graph: nx.DiGraph):
    """This corrects for the edge distance in case the flow at a segmentation
    node is available. The EdgeAttr.DISTANCE.value is set equal to
    the L2 norm of (pos@t+1 - (flow + pos@t).


    Args:
        candidate_graph (nx.DiGraph): Existing candidate graph with nodes and
        edges.

    Returns:
        candidate_graph (nx.DiGraph): Updated candidate graph. (Edge
        distance attribute is updated, by taking flow into account).

    """
    for edge in candidate_graph.edges(data=True):
        in_node = candidate_graph.nodes[edge[0]]
        out_node = candidate_graph.nodes[edge[1]]
        dist = math.dist(
            out_node[NodeAttr.POS.value],
            in_node[NodeAttr.POS.value] + in_node[NodeAttr.FLOW.value],
        )
        edge[2][EdgeAttr.DISTANCE.value] = dist

    return candidate_graph
