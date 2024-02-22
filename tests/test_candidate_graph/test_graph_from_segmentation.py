from motile_toolbox.candidate_graph import graph_from_segmentation
import pytest
import numpy as np
from skimage.draw import disk
from collections import Counter
import math


@pytest.fixture
def segmentation_2d():
    frame_shape = (100, 100)
    total_shape = (2,) + frame_shape
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


def sphere(center, radius, shape):
    distance = np.linalg.norm(
        np.subtract(np.indices(shape).T, np.asarray(center)), axis=len(center)
    )
    mask = distance <= radius
    return mask


@pytest.fixture
def segmentation_3d():
    frame_shape = (100, 100, 100)
    total_shape = (2,) + frame_shape
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


def test_graph_from_segmentation_invalid():
    # test invalid attributes
    with pytest.raises(ValueError):
        graph_from_segmentation(
            np.zeros((3, 10, 10, 10), dtype="int32"),
            10,
            attributes=["invalid"],
        )

    with pytest.raises(ValueError):
        graph_from_segmentation(
            np.zeros((3, 10, 10), dtype="int32"), 100, position_keys=["z", "y", "x"]
        )


def test_graph_from_segmentation_empty():
    empty_graph = graph_from_segmentation(np.zeros((3, 10, 10, 10), dtype="int32"), 10)
    assert Counter(empty_graph.nodes) == Counter([])


def test_graph_from_segmentation_2d(segmentation_2d):
    # test with 2D segmentation
    graph_2d = graph_from_segmentation(
        segmentation=segmentation_2d,
        max_edge_distance=100,
    )
    assert Counter(list(graph_2d.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert Counter(list(graph_2d.edges)) == Counter([("0_1", "1_1"), ("0_1", "1_2")])
    assert graph_2d.nodes["0_1"]["segmentation_id"] == 1
    assert graph_2d.nodes["0_1"]["t"] == 0
    assert graph_2d.nodes["0_1"]["y"] == 50
    assert graph_2d.nodes["0_1"]["x"] == 50
    assert graph_2d.edges[("0_1", "1_1")]["distance"] == pytest.approx(42.43, abs=0.01)
    # math.dist([50, 50], [20, 80])
    assert graph_2d.edges[("0_1", "1_2")]["distance"] == pytest.approx(11.18, abs=0.01)
    # math.dist([50, 50], [60, 45])

    # lower edge distance
    graph_2d = graph_from_segmentation(
        segmentation=segmentation_2d,
        max_edge_distance=15,
    )
    assert Counter(list(graph_2d.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert Counter(list(graph_2d.edges)) == Counter([("0_1", "1_2")])
    assert graph_2d.edges[("0_1", "1_2")]["distance"] == pytest.approx(11.18, abs=0.01)


def test_graph_from_segmentation_3d(segmentation_3d):
    # test with 3D segmentation
    graph_3d = graph_from_segmentation(
        segmentation=segmentation_3d,
        max_edge_distance=100,
        position_keys=("z", "y", "x"),
    )
    assert Counter(list(graph_3d.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert Counter(list(graph_3d.edges)) == Counter([("0_1", "1_1"), ("0_1", "1_2")])
    assert graph_3d.nodes["0_1"]["segmentation_id"] == 1
    assert graph_3d.nodes["0_1"]["t"] == 0
    assert graph_3d.nodes["0_1"]["y"] == 50
    assert graph_3d.nodes["0_1"]["x"] == 50
    assert graph_3d.edges[("0_1", "1_1")]["distance"] == pytest.approx(42.43, abs=0.01)
    # math.dist([50, 50], [20, 80])
    assert graph_3d.edges[("0_1", "1_2")]["distance"] == pytest.approx(11.18, abs=0.01)
    # math.dist([50, 50], [60, 45])

    # lower edge distance
    graph_3d = graph_from_segmentation(
        segmentation=segmentation_3d,
        max_edge_distance=15,
        position_keys=("z", "y", "x"),
    )
    assert Counter(list(graph_3d.nodes)) == Counter(["0_1", "1_1", "1_2"])
    assert Counter(list(graph_3d.edges)) == Counter([("0_1", "1_2")])
    assert graph_3d.edges[("0_1", "1_2")]["distance"] == pytest.approx(11.18, abs=0.01)
    # math.dist([50, 50], [60, 45])
