from collections import Counter

import networkx as nx
import numpy as np
import pytest
from motile_toolbox.candidate_graph import (
    EdgeAttr, NodeAttr,
    add_cand_edges,
    graph_from_segmentation,
    nodes_from_segmentation,
)


# nodes_from_segmentation
def test_nodes_from_segmentation_empty():
    # test with empty segmentation
    empty_graph, node_frame_dict = nodes_from_segmentation(
        np.zeros((3, 10, 10), dtype="int32")
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
    assert node_graph.nodes["1_1"][NodeAttr.POS.value] == (20, 80)

    assert node_frame_dict[0] == ["0_1"]
    assert Counter(node_frame_dict[1]) == Counter(["1_1", "1_2"])


def test_nodes_from_segmentation_2d_hypo(segmentation_2d):
    # test with 2D segmentation
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_2d, hypo_id=0
    )
    assert Counter(list(node_graph.nodes)) == Counter(["0_0_1", "1_0_1", "1_0_2"])
    assert node_graph.nodes["1_0_1"][NodeAttr.SEG_ID.value] == 1
    assert node_graph.nodes["1_0_1"][NodeAttr.SEG_HYPO.value] == 0
    assert node_graph.nodes["1_0_1"][NodeAttr.TIME.value] == 1
    assert node_graph.nodes["1_0_1"][NodeAttr.POS.value] == (20, 80)

    assert node_frame_dict[0] == ["0_0_1"]
    assert Counter(node_frame_dict[1]) == Counter(["1_0_1", "1_0_2"])


def test_nodes_from_segmentation_3d(segmentation_3d):
    # test with 3D segmentation
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_3d,
    )
    assert Counter(list(node_graph.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert node_graph.nodes["1_1"][NodeAttr.SEG_ID.value] == 1
    assert node_graph.nodes["1_1"][NodeAttr.TIME.value] == 1
    assert node_graph.nodes["1_1"][NodeAttr.POS.value] == (20, 50, 80)

    assert node_frame_dict[0] == ["0_1"]
    assert Counter(node_frame_dict[1]) == Counter(["1_1", "1_2"])


# add_cand_edges
def test_add_cand_edges_2d(graph_2d):
    cand_graph = nx.create_empty_copy(graph_2d)
    add_cand_edges(cand_graph, max_edge_distance=50)
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_2d.edges))
    for edge in cand_graph.edges:
        assert (
            pytest.approx(cand_graph.edges[edge][EdgeAttr.DISTANCE.value], abs=0.01)
            == graph_2d.edges[edge][EdgeAttr.DISTANCE.value]
        )


def test_add_cand_edges_3d(graph_3d):
    cand_graph = nx.create_empty_copy(graph_3d)
    add_cand_edges(cand_graph, max_edge_distance=15)
    graph_3d.remove_edge("0_1", "1_1")
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_3d.edges))
    for edge in cand_graph.edges:
        assert pytest.approx(cand_graph.edges[edge], abs=0.01) == graph_3d.edges[edge]


def test_graph_from_segmentation_2d(segmentation_2d, graph_2d):
    # test with 2D segmentation
    cand_graph = graph_from_segmentation(
        segmentation=segmentation_2d,
        max_edge_distance=100,
        iou=True,
    )
    assert Counter(list(cand_graph.nodes)) == Counter(list(graph_2d.nodes))
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_2d.edges))
    for node in cand_graph.nodes:
        assert Counter(cand_graph.nodes[node]) == Counter(graph_2d.nodes[node])
    for edge in cand_graph.edges:
        print(cand_graph.edges[edge])
        assert (
            pytest.approx(cand_graph.edges[edge][EdgeAttr.DISTANCE.value], abs=0.01)
            == graph_2d.edges[edge][EdgeAttr.DISTANCE.value]
        )
        assert (
            pytest.approx(cand_graph.edges[edge][EdgeAttr.IOU.value], abs=0.01)
            == graph_2d.edges[edge][EdgeAttr.IOU.value]
        )

    # lower edge distance
    cand_graph = graph_from_segmentation(
        segmentation=segmentation_2d,
        max_edge_distance=15,
    )
    assert Counter(list(cand_graph.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert Counter(list(cand_graph.edges)) == Counter([("0_1", "1_2")])
    assert cand_graph.edges[("0_1", "1_2")][EdgeAttr.DISTANCE.value] == pytest.approx(
        11.18, abs=0.01
    )


def test_graph_from_segmentation_3d(segmentation_3d, graph_3d):
    # test with 3D segmentation
    cand_graph = graph_from_segmentation(
        segmentation=segmentation_3d,
        max_edge_distance=100,
    )
    assert Counter(list(cand_graph.nodes)) == Counter(list(graph_3d.nodes))
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_3d.edges))
    for node in cand_graph.nodes:
        assert Counter(cand_graph.nodes[node]) == Counter(graph_3d.nodes[node])
    for edge in cand_graph.edges:
        assert pytest.approx(cand_graph.edges[edge], abs=0.01) == graph_3d.edges[edge]
