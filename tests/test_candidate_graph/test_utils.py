from collections import Counter

import networkx as nx
import numpy as np
from motile_toolbox.candidate_graph import (
    NodeAttr,
    add_cand_edges,
    get_node_id,
    nodes_from_segmentation,
)
from motile_toolbox.candidate_graph.utils import (
    _compute_node_frame_dict,
    nodes_from_points_list,
)


# nodes_from_segmentation
def test_nodes_from_segmentation_empty():
    # test with empty segmentation
    empty_graph, node_frame_dict = nodes_from_segmentation(
        np.zeros((3, 1, 10, 10), dtype="int32")
    )
    assert Counter(empty_graph.nodes) == Counter([])
    assert node_frame_dict == {}


def test_nodes_from_segmentation_2d(segmentation_2d):
    # test with 2D segmentation
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_2d,
    )
    assert Counter(list(node_graph.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert node_graph.nodes["1_1"][NodeAttr.SEG_ID.value] == 1
    assert node_graph.nodes["1_1"][NodeAttr.TIME.value] == 1
    assert node_graph.nodes["1_1"][NodeAttr.AREA.value] == 305
    assert node_graph.nodes["1_1"][NodeAttr.POS.value] == (20, 80)

    assert node_frame_dict[0] == ["0_1"]
    assert Counter(node_frame_dict[1]) == Counter(["1_1", "1_2"])

    # test with scaling
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_2d, scale=[1, 1, 2]
    )
    assert Counter(list(node_graph.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert node_graph.nodes["1_1"][NodeAttr.SEG_ID.value] == 1
    assert node_graph.nodes["1_1"][NodeAttr.TIME.value] == 1
    assert node_graph.nodes["1_1"][NodeAttr.AREA.value] == 610
    assert node_graph.nodes["1_1"][NodeAttr.POS.value] == (20, 160)

    assert node_frame_dict[0] == ["0_1"]
    assert Counter(node_frame_dict[1]) == Counter(["1_1", "1_2"])


def test_nodes_from_segmentation_2d_hypo(
    multi_hypothesis_segmentation_2d, multi_hypothesis_graph_2d
):
    # test with 2D segmentation
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=multi_hypothesis_segmentation_2d
    )
    assert Counter(list(node_graph.nodes)) == Counter(
        list(multi_hypothesis_graph_2d.nodes)
    )
    assert node_graph.nodes["1_0_1"][NodeAttr.SEG_ID.value] == 1
    assert node_graph.nodes["1_0_1"][NodeAttr.SEG_HYPO.value] == 0
    assert node_graph.nodes["1_0_1"][NodeAttr.TIME.value] == 1
    assert node_graph.nodes["1_0_1"][NodeAttr.AREA.value] == 305
    assert node_graph.nodes["1_0_1"][NodeAttr.POS.value] == (20, 80)

    assert Counter(node_frame_dict[0]) == Counter(["0_0_1", "0_1_1"])
    assert Counter(node_frame_dict[1]) == Counter(["1_0_1", "1_0_2", "1_1_1", "1_1_2"])


def test_nodes_from_segmentation_3d(segmentation_3d):
    # test with 3D segmentation
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_3d,
    )
    assert Counter(list(node_graph.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert node_graph.nodes["1_1"][NodeAttr.SEG_ID.value] == 1
    assert node_graph.nodes["1_1"][NodeAttr.TIME.value] == 1
    assert node_graph.nodes["1_1"][NodeAttr.AREA.value] == 4169
    assert node_graph.nodes["1_1"][NodeAttr.POS.value] == (20, 50, 80)

    assert node_frame_dict[0] == ["0_1"]
    assert Counter(node_frame_dict[1]) == Counter(["1_1", "1_2"])

    # test with scaling
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_3d, scale=[1, 1, 4.5, 1]
    )
    assert Counter(list(node_graph.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert node_graph.nodes["1_1"][NodeAttr.SEG_ID.value] == 1
    assert node_graph.nodes["1_1"][NodeAttr.AREA.value] == 4169 * 4.5
    assert node_graph.nodes["1_1"][NodeAttr.TIME.value] == 1
    assert node_graph.nodes["1_1"][NodeAttr.POS.value] == (20.0, 225.0, 80.0)

    assert node_frame_dict[0] == ["0_1"]
    assert Counter(node_frame_dict[1]) == Counter(["1_1", "1_2"])


# add_cand_edges
def test_add_cand_edges_2d(graph_2d):
    cand_graph = nx.create_empty_copy(graph_2d)
    add_cand_edges(cand_graph, max_edge_distance=50)
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_2d.edges))


def test_add_cand_edges_3d(graph_3d):
    cand_graph = nx.create_empty_copy(graph_3d)
    add_cand_edges(cand_graph, max_edge_distance=15)
    graph_3d.remove_edge("0_1", "1_1")
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_3d.edges))


def test_get_node_id():
    assert get_node_id(0, 2) == "0_2"
    assert get_node_id(2, 10, 3) == "2_3_10"


def test_compute_node_frame_dict(graph_2d):
    node_frame_dict = _compute_node_frame_dict(graph_2d)
    expected = {
        0: [
            "0_1",
        ],
        1: ["1_1", "1_2"],
    }
    assert node_frame_dict == expected


def test_nodes_from_points_list_2d():
    points_list = np.array(
        [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
            [1, 2, 3, 4],
        ]
    )
    cand_graph, node_frame_dict = nodes_from_points_list(points_list)
    assert Counter(list(cand_graph.nodes)) == Counter([0, 1, 2])
    assert cand_graph.nodes[0][NodeAttr.TIME.value] == 0
    assert (cand_graph.nodes[0][NodeAttr.POS.value] == np.array([1, 2, 3])).all()
    assert cand_graph.nodes[1][NodeAttr.TIME.value] == 2
    assert (cand_graph.nodes[1][NodeAttr.POS.value] == np.array([3, 4, 5])).all()
