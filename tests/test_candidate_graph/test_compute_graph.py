from collections import Counter

import numpy as np
import pytest
from motile_toolbox.candidate_graph import EdgeAttr, get_candidate_graph
from motile_toolbox.candidate_graph.compute_graph import (
    get_candidate_graph_from_points_list,
)
from motile_toolbox.candidate_graph.graph_attributes import NodeAttr


def test_graph_from_segmentation_2d(segmentation_2d, graph_2d):
    # test with 2D segmentation
    cand_graph, _ = get_candidate_graph(
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
            pytest.approx(cand_graph.edges[edge][EdgeAttr.IOU.value], abs=0.01)
            == graph_2d.edges[edge][EdgeAttr.IOU.value]
        )

    # lower edge distance
    cand_graph, _ = get_candidate_graph(
        segmentation=segmentation_2d,
        max_edge_distance=15,
    )
    assert Counter(list(cand_graph.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert Counter(list(cand_graph.edges)) == Counter([("0_1", "1_2")])


def test_graph_from_segmentation_3d(segmentation_3d, graph_3d):
    # test with 3D segmentation
    cand_graph, _ = get_candidate_graph(
        segmentation=segmentation_3d,
        max_edge_distance=100,
    )
    assert Counter(list(cand_graph.nodes)) == Counter(list(graph_3d.nodes))
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_3d.edges))
    for node in cand_graph.nodes:
        assert Counter(cand_graph.nodes[node]) == Counter(graph_3d.nodes[node])
    for edge in cand_graph.edges:
        assert pytest.approx(cand_graph.edges[edge], abs=0.01) == graph_3d.edges[edge]


def test_graph_from_multi_segmentation_2d(
    multi_hypothesis_segmentation_2d, multi_hypothesis_graph_2d
):
    # test with 2D segmentation
    cand_graph, conflict_set = get_candidate_graph(
        segmentation=multi_hypothesis_segmentation_2d,
        max_edge_distance=100,
        iou=True,
    )
    assert Counter(list(cand_graph.nodes)) == Counter(
        list(multi_hypothesis_graph_2d.nodes)
    )
    assert Counter(list(cand_graph.edges)) == Counter(
        list(multi_hypothesis_graph_2d.edges)
    )
    for node in cand_graph.nodes:
        assert Counter(cand_graph.nodes[node]) == Counter(
            multi_hypothesis_graph_2d.nodes[node]
        )
    for edge in cand_graph.edges:
        assert (
            pytest.approx(cand_graph.edges[edge][EdgeAttr.IOU.value], abs=0.01)
            == multi_hypothesis_graph_2d.edges[edge][EdgeAttr.IOU.value]
        )
    # TODO: Test conflict set

    # lower edge distance
    cand_graph, _ = get_candidate_graph(
        segmentation=multi_hypothesis_segmentation_2d,
        max_edge_distance=14,
    )
    assert Counter(list(cand_graph.nodes)) == Counter(
        list(multi_hypothesis_graph_2d.nodes)
    )
    assert Counter(list(cand_graph.edges)) == Counter(
        [("0_0_1", "1_0_2"), ("0_0_1", "1_1_2"), ("0_1_1", "1_1_2")]
    )


def test_graph_from_points_list():
    points_list = np.array(
        [
            # t, z, y, x
            [0, 1, 1, 1],
            [2, 3, 3, 3],
            [1, 2, 2, 2],
            [2, 6, 6, 6],
            [2, 1, 1, 1],
        ]
    )
    cand_graph = get_candidate_graph_from_points_list(points_list, max_edge_distance=3)
    assert cand_graph.number_of_edges() == 3
    assert len(cand_graph.in_edges(3)) == 0

    # test scale
    cand_graph = get_candidate_graph_from_points_list(
        points_list, max_edge_distance=3, scale=[1, 1, 1, 5]
    )
    assert cand_graph.number_of_edges() == 0
    assert len(cand_graph.in_edges(3)) == 0
    assert cand_graph.nodes[0][NodeAttr.POS.value] == [1, 1, 5]
    assert cand_graph.nodes[0][NodeAttr.TIME.value] == 0
