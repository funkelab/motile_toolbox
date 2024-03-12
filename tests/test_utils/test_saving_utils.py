import networkx as nx
import numpy as np
import pytest
from motile_toolbox.utils import relabel_segmentation
from numpy.testing import assert_array_equal
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
    # first cell centered at (20, 80) with label 2
    # second cell centered at (60, 45) with label 3
    rr, cc = disk(center=(20, 80), radius=10, shape=frame_shape)
    segmentation[1][rr, cc] = 2
    rr, cc = disk(center=(60, 45), radius=15, shape=frame_shape)
    segmentation[1][rr, cc] = 3

    return segmentation


@pytest.fixture
def graph_2d():
    graph = nx.DiGraph()
    nodes = [
        ("0_1", {"y": 50, "x": 50, "t": 0, "segmentation_id": 1}),
        ("1_1", {"y": 20, "x": 80, "t": 1, "segmentation_id": 2}),
    ]
    edges = [
        ("0_1", "1_1", {"distance": 42.43}),
    ]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def test_relabel_segmentation(segmentation_2d, graph_2d):
    frame_shape = (100, 100)
    expected = np.zeros(segmentation_2d.shape, dtype="int32")
    # make frame with one cell in center with label 1
    rr, cc = disk(center=(50, 50), radius=20, shape=(100, 100))
    expected[0][rr, cc] = 1

    # make frame with cell centered at (20, 80) with label 1
    rr, cc = disk(center=(20, 80), radius=10, shape=frame_shape)
    expected[1][rr, cc] = 1

    relabeled_seg = relabel_segmentation(graph_2d, segmentation_2d)
    print(f"Nonzero relabeled: {np.count_nonzero(relabeled_seg)}")
    print(f"Nonzero expected: {np.count_nonzero(expected)}")
    print(f"Max relabeled: {np.max(relabeled_seg)}")
    print(f"Max expected: {np.max(expected)}")

    assert_array_equal(relabeled_seg, expected)
