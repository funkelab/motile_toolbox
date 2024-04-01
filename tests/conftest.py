import networkx as nx
import numpy as np
import pytest
from motile_toolbox.candidate_graph import EdgeAttr, NodeAttr
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
        (
            "0_1",
            {
                NodeAttr.POS.value: (50, 50),
                NodeAttr.TIME.value: 0,
                NodeAttr.SEG_ID.value: 1,
            },
        ),
        (
            "1_1",
            {
                NodeAttr.POS.value: (20, 80),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_ID.value: 1,
            },
        ),
        (
            "1_2",
            {
                NodeAttr.POS.value: (60, 45),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_ID.value: 2,
            },
        ),
    ]
    edges = [
        ("0_1", "1_1", {EdgeAttr.DISTANCE.value: 42.43, EdgeAttr.IOU.value: 0.0}),
        ("0_1", "1_2", {EdgeAttr.DISTANCE.value: 11.18, EdgeAttr.IOU.value: 0.395}),
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
        (
            "0_1",
            {
                NodeAttr.POS.value: (50, 50, 50),
                NodeAttr.TIME.value: 0,
                NodeAttr.SEG_ID.value: 1,
            },
        ),
        (
            "1_1",
            {
                NodeAttr.POS.value: (20, 50, 80),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_ID.value: 1,
            },
        ),
        (
            "1_2",
            {
                NodeAttr.POS.value: (60, 50, 45),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_ID.value: 2,
            },
        ),
    ]
    edges = [
        # math.dist([50, 50], [20, 80])
        ("0_1", "1_1", {EdgeAttr.DISTANCE.value: 42.43}),
        # math.dist([50, 50], [60, 45])
        ("0_1", "1_2", {EdgeAttr.DISTANCE.value: 11.18}),
    ]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph
