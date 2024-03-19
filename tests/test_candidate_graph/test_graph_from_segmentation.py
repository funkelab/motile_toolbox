from collections import Counter

import networkx as nx
import numpy as np
import pytest
from motile_toolbox.candidate_graph import EdgeAttr, NodeAttr
from motile_toolbox.candidate_graph.graph_from_segmentation import (
    add_cand_edges,
    compute_ious,
    graph_from_segmentation,
    nodes_from_segmentation,
)
from skimage.draw import disk


@pytest.fixture
def segmentation_2d():
    frame_shape = (100, 100)
    total_shape = (2, *frame_shape)
    segmentation = np.zeros(total_shape, dtype="int32")
    # make frame with one cell in center with label 1
    rr, cc = disk(center=(50, 50), radius=20, shape=(100, 100))
    segmentation[0][rr, cc] = 1

    # make frame with two cells
    # first cell centered at (20, 80) with label 1
    # second cell centered at (60, 45) with label 2
    rr, cc = disk(center=(20, 80), radius=10, shape=frame_shape)
    segmentation[1][rr, cc] = 1
    rr, cc = disk(center=(60, 45), radius=15, shape=frame_shape)
    segmentation[1][rr, cc] = 2

    return segmentation


@pytest.fixture
def graph_2d():
    graph = nx.DiGraph()
    nodes = [
        ("0_1", {"y": 50, "x": 50, "t": 0, "segmentation_id": 1}),
        ("1_1", {"y": 20, "x": 80, "t": 1, "segmentation_id": 1}),
        ("1_2", {"y": 60, "x": 45, "t": 1, "segmentation_id": 2}),
    ]
    edges = [
        ("0_1", "1_1", {"distance": 42.43, "iou": 0.0}),
        ("0_1", "1_2", {"distance": 11.18, "iou": 0.395}),
    ]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def sphere(center, radius, shape):
    assert len(center) == len(shape)
    indices = np.moveaxis(np.indices(shape), 0, -1)  # last dim is the index
    distance = np.linalg.norm(np.subtract(indices, np.asarray(center)), axis=-1)
    mask = distance <= radius
    return mask


@pytest.fixture
def segmentation_3d():
    frame_shape = (100, 100, 100)
    total_shape = (2, *frame_shape)
    segmentation = np.zeros(total_shape, dtype="int32")
    # make frame with one cell in center with label 1
    mask = sphere(center=(50, 50, 50), radius=20, shape=frame_shape)
    segmentation[0][mask] = 1

    # make frame with two cells
    # first cell centered at (20, 50, 80) with label 1
    # second cell centered at (60, 50, 45) with label 2
    mask = sphere(center=(20, 50, 80), radius=10, shape=frame_shape)
    segmentation[1][mask] = 1
    mask = sphere(center=(60, 50, 45), radius=15, shape=frame_shape)
    segmentation[1][mask] = 2

    return segmentation


@pytest.fixture
def graph_3d():
    graph = nx.DiGraph()
    nodes = [
        ("0_1", {"z": 50, "y": 50, "x": 50, "t": 0, "segmentation_id": 1}),
        ("1_1", {"z": 20, "y": 50, "x": 80, "t": 1, "segmentation_id": 1}),
        ("1_2", {"z": 60, "y": 50, "x": 45, "t": 1, "segmentation_id": 2}),
    ]
    edges = [
        # math.dist([50, 50], [20, 80])
        ("0_1", "1_1", {"distance": 42.43}),
        # math.dist([50, 50], [60, 45])
        ("0_1", "1_2", {"distance": 11.18}),
    ]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


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
    assert node_graph.nodes["1_1"]["segmentation_id"] == 1
    assert node_graph.nodes["1_1"]["t"] == 1
    assert node_graph.nodes["1_1"]["y"] == 20
    assert node_graph.nodes["1_1"]["x"] == 80

    assert node_frame_dict[0] == ["0_1"]
    assert Counter(node_frame_dict[1]) == Counter(["1_1", "1_2"])

    # remove attrs
    node_graph, _ = nodes_from_segmentation(
        segmentation=segmentation_2d,
        attributes=[],
    )
    assert Counter(list(node_graph.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert "segmentation_id" not in node_graph.nodes["0_1"]


def test_nodes_from_segmentation_3d(segmentation_3d):
    # test with 3D segmentation
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_3d,
        attributes=[NodeAttr.SEG_ID],
        position_keys=("pos_z", "pos_y", "pos_x"),
    )
    assert Counter(list(node_graph.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert node_graph.nodes["1_1"]["segmentation_id"] == 1
    assert node_graph.nodes["1_1"]["t"] == 1
    assert node_graph.nodes["1_1"]["pos_z"] == 20
    assert node_graph.nodes["1_1"]["pos_y"] == 50
    assert node_graph.nodes["1_1"]["pos_x"] == 80

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
    add_cand_edges(cand_graph, max_edge_distance=15, position_keys=("z", "y", "x"))
    graph_3d.remove_edge("0_1", "1_1")
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_3d.edges))
    for edge in cand_graph.edges:
        assert pytest.approx(cand_graph.edges[edge], abs=0.01) == graph_3d.edges[edge]


# graph_from_segmentation
def test_graph_from_segmentation_invalid():
    # test invalid attributes
    with pytest.raises(ValueError):
        graph_from_segmentation(
            np.zeros((3, 10, 10, 10), dtype="int32"),
            10,
            edge_attributes=["invalid"],
        )
    with pytest.raises(ValueError):
        graph_from_segmentation(
            np.zeros((3, 10, 10, 10), dtype="int32"),
            10,
            node_attributes=["invalid"],
        )

    with pytest.raises(ValueError):
        graph_from_segmentation(
            np.zeros((3, 10, 10), dtype="int32"), 100, position_keys=["z", "y", "x"]
        )


def test_graph_from_segmentation_2d(segmentation_2d, graph_2d):
    # test with 2D segmentation
    cand_graph = graph_from_segmentation(
        segmentation=segmentation_2d,
        max_edge_distance=100,
        edge_attributes=[EdgeAttr.DISTANCE, EdgeAttr.IOU],
    )
    assert Counter(list(cand_graph.nodes)) == Counter(list(graph_2d.nodes))
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_2d.edges))
    for node in cand_graph.nodes:
        assert Counter(cand_graph.nodes[node]) == Counter(graph_2d.nodes[node])
    for edge in cand_graph.edges:
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
    assert cand_graph.edges[("0_1", "1_2")]["distance"] == pytest.approx(
        11.18, abs=0.01
    )


def test_graph_from_segmentation_3d(segmentation_3d, graph_3d):
    # test with 3D segmentation
    cand_graph = graph_from_segmentation(
        segmentation=segmentation_3d,
        max_edge_distance=100,
        position_keys=("z", "y", "x"),
    )
    assert Counter(list(cand_graph.nodes)) == Counter(list(graph_3d.nodes))
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_3d.edges))
    for node in cand_graph.nodes:
        assert Counter(cand_graph.nodes[node]) == Counter(graph_3d.nodes[node])
    for edge in cand_graph.edges:
        assert pytest.approx(cand_graph.edges[edge], abs=0.01) == graph_3d.edges[edge]


def test_compute_ious_2d(segmentation_2d):
    ious = compute_ious(segmentation_2d[0], segmentation_2d[1])
    expected = {1: {2: 555.46 / 1408.0}}
    assert ious.keys() == expected.keys()
    assert ious[1].keys() == expected[1].keys()
    assert ious[1][2] == pytest.approx(expected[1][2], abs=0.1)

    ious = compute_ious(segmentation_2d[1], segmentation_2d[1])
    expected = {1: {1: 1.0}, 2: {2: 1.0}}
    assert ious.keys() == expected.keys()
    assert ious[1].keys() == expected[1].keys()
    assert ious[1][1] == pytest.approx(expected[1][1], abs=0.1)
    assert ious[2].keys() == expected[2].keys()
    assert ious[2][2] == pytest.approx(expected[2][2], abs=0.1)


def test_compute_ious_3d(segmentation_3d):
    ious = compute_ious(segmentation_3d[0], segmentation_3d[1])
    expected = {1: {2: 0.30}}
    assert ious.keys() == expected.keys()
    assert ious[1].keys() == expected[1].keys()
    assert ious[1][2] == pytest.approx(expected[1][2], abs=0.1)

    ious = compute_ious(segmentation_3d[1], segmentation_3d[1])
    expected = {1: {1: 1.0}, 2: {2: 1.0}}
    assert ious.keys() == expected.keys()
    assert ious[1].keys() == expected[1].keys()
    assert ious[1][1] == pytest.approx(expected[1][1], abs=0.1)
    assert ious[2].keys() == expected[2].keys()
    assert ious[2][2] == pytest.approx(expected[2][2], abs=0.1)
