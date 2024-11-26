import motile
import networkx as nx
import numpy as np
import pytest
from motile_toolbox.candidate_graph.graph_attributes import EdgeAttr, NodeAttr
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
def multi_hypothesis_segmentation_2d():
    """
    Creates a multi-hypothesis version of the `segmentation_2d` fixture defined above.

    """
    frame_shape = (100, 100)
    total_shape = (2, 2, *frame_shape)  # 2 hypotheses, 2 time points, H, W
    segmentation = np.zeros(total_shape, dtype="int32")
    # make frame with one cell in center with label 1 (hypo 0)
    rr0, cc0 = disk(center=(50, 50), radius=20, shape=frame_shape)
    # make frame with one cell at (45, 45) with label 2 (hypo 1)
    rr1, cc1 = disk(center=(45, 45), radius=15, shape=frame_shape)

    segmentation[0, 0][rr0, cc0] = 1
    segmentation[1, 0][rr1, cc1] = 2

    # make frame with two cells
    # first cell centered at (20, 80) with label 3 (hypo0) and 4 (hypo1)
    rr0, cc0 = disk(center=(20, 80), radius=10, shape=frame_shape)
    rr1, cc1 = disk(center=(15, 75), radius=15, shape=frame_shape)

    segmentation[0, 1][rr0, cc0] = 3
    segmentation[1, 1][rr1, cc1] = 4

    # second cell centered at (60, 45) with label 5(hypo0) and 6 (hypo1)
    rr0, cc0 = disk(center=(60, 45), radius=15, shape=frame_shape)
    rr1, cc1 = disk(center=(55, 40), radius=20, shape=frame_shape)

    segmentation[0, 1][rr0, cc0] = 5
    segmentation[1, 1][rr1, cc1] = 6

    return segmentation


@pytest.fixture
def graph_2d():
    graph = nx.DiGraph()
    nodes = [
        (
            1,
            {
                NodeAttr.POS.value: (50, 50),
                NodeAttr.TIME.value: 0,
                NodeAttr.SEG_ID.value: 1,
                NodeAttr.AREA.value: 1245,
            },
        ),
        (
            2,
            {
                NodeAttr.POS.value: (20, 80),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_ID.value: 2,
                NodeAttr.AREA.value: 305,
            },
        ),
        (
            3,
            {
                NodeAttr.POS.value: (60, 45),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_ID.value: 3,
                NodeAttr.AREA.value: 697,
            },
        ),
    ]
    edges = [
        (1, 2, {EdgeAttr.IOU.value: 0.0}),
        (1, 3, {EdgeAttr.IOU.value: 0.395}),
    ]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


@pytest.fixture
def multi_hypothesis_graph_2d():
    graph = nx.DiGraph()
    nodes = [
        (
            1,
            {
                NodeAttr.POS.value: (50, 50),
                NodeAttr.TIME.value: 0,
                NodeAttr.SEG_HYPO.value: 0,
                NodeAttr.SEG_ID.value: 1,
                NodeAttr.AREA.value: 1245,
            },
        ),
        (
            2,
            {
                NodeAttr.POS.value: (45, 45),
                NodeAttr.TIME.value: 0,
                NodeAttr.SEG_HYPO.value: 1,
                NodeAttr.SEG_ID.value: 2,
                NodeAttr.AREA.value: 697,
            },
        ),
        (
            3,
            {
                NodeAttr.POS.value: (20, 80),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_HYPO.value: 0,
                NodeAttr.SEG_ID.value: 3,
                NodeAttr.AREA.value: 305,
            },
        ),
        (
            4,
            {
                NodeAttr.POS.value: (15, 75),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_HYPO.value: 1,
                NodeAttr.SEG_ID.value: 4,
                NodeAttr.AREA.value: 697,
            },
        ),
        (
            5,
            {
                NodeAttr.POS.value: (60, 45),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_HYPO.value: 0,
                NodeAttr.SEG_ID.value: 5,
                NodeAttr.AREA.value: 697,
            },
        ),
        (
            6,
            {
                NodeAttr.POS.value: (55, 40),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_HYPO.value: 1,
                NodeAttr.SEG_ID.value: 6,
                NodeAttr.AREA.value: 1245,
            },
        ),
    ]

    edges = [
        (1, 3, {EdgeAttr.IOU.value: 0.0}),
        (1, 4, {EdgeAttr.IOU.value: 0.0}),
        (
            1,
            5,
            {EdgeAttr.IOU.value: 0.3931},
        ),
        (
            1,
            6,
            {EdgeAttr.IOU.value: 0.4768},
        ),
        (2, 3, {EdgeAttr.IOU.value: 0.0}),
        (2, 4, {EdgeAttr.IOU.value: 0.0}),
        (2, 5, {EdgeAttr.IOU.value: 0.2402}),
        (
            2,
            6,
            {EdgeAttr.IOU.value: 0.3931},
        ),
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
    # first cell centered at (20, 50, 80) with label 2
    # second cell centered at (60, 50, 45) with label 3
    mask = sphere(center=(20, 50, 80), radius=10, shape=frame_shape)
    segmentation[1][mask] = 2
    mask = sphere(center=(60, 50, 45), radius=15, shape=frame_shape)
    segmentation[1][mask] = 3

    return segmentation


@pytest.fixture
def multi_hypothesis_segmentation_3d():
    """
    Creates a multi-hypothesis version of the `segmentation_3d` fixture defined above.

    """
    frame_shape = (100, 100, 100)
    total_shape = (2, 2, *frame_shape)  # 2 time points, 2 hypotheses
    segmentation = np.zeros(total_shape, dtype="int32")
    # make first frame with one cell in center with label 1
    mask = sphere(center=(50, 50, 50), radius=20, shape=frame_shape)
    segmentation[0, 0][mask] = 1

    # make second hypothesis first frame with one cell in center with label 1
    mask = sphere(center=(45, 50, 55), radius=20, shape=frame_shape)
    segmentation[1, 0][mask] = 1

    # make second frame, first hypothesis with two cells
    # first cell centered at (20, 50, 80) with label 1
    # second cell centered at (60, 50, 45) with label 2
    mask = sphere(center=(20, 50, 80), radius=10, shape=frame_shape)
    segmentation[0, 1][mask] = 1
    mask = sphere(center=(60, 50, 45), radius=15, shape=frame_shape)
    segmentation[0, 1][mask] = 2

    # make second frame, second hypothesis with one cell
    # first cell centered at (15, 50, 70) with label 1
    mask = sphere(center=(15, 50, 70), radius=10, shape=frame_shape)
    segmentation[1, 1][mask] = 1

    return segmentation


@pytest.fixture
def graph_3d():
    graph = nx.DiGraph()
    nodes = [
        (
            1,
            {
                NodeAttr.POS.value: (50, 50, 50),
                NodeAttr.TIME.value: 0,
                NodeAttr.SEG_ID.value: 1,
                NodeAttr.AREA.value: 33401,
            },
        ),
        (
            2,
            {
                NodeAttr.POS.value: (20, 50, 80),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_ID.value: 2,
                NodeAttr.AREA.value: 4169,
            },
        ),
        (
            3,
            {
                NodeAttr.POS.value: (60, 50, 45),
                NodeAttr.TIME.value: 1,
                NodeAttr.SEG_ID.value: 3,
                NodeAttr.AREA.value: 14147,
            },
        ),
    ]
    edges = [
        # math.dist([50, 50], [20, 80])
        (1, 2),
        # math.dist([50, 50], [60, 45])
        (1, 3),
    ]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


@pytest.fixture
def arlo_graph_nx() -> nx.DiGraph:
    """Create the "Arlo graph", a simple toy graph for testing.

       x
       |
    200|           6
       |         /
    150|   1---3---5
       |     x   x
    100|   0---2---4
        ------------------------------------ t
           0   1   2
    """
    cells = [
        {"id": 0, "t": 0, "x": 101, "score": 1.0},
        {"id": 1, "t": 0, "x": 150, "score": 1.0},
        {"id": 2, "t": 1, "x": 100, "score": 1.0},
        {"id": 3, "t": 1, "x": 151, "score": 1.0},
        {"id": 4, "t": 2, "x": 102, "score": 1.0},
        {"id": 5, "t": 2, "x": 149, "score": 1.0},
        {"id": 6, "t": 2, "x": 200, "score": 1.0},
    ]

    edges = [
        {"source": 0, "target": 2, "prediction_distance": 1.0},
        {"source": 1, "target": 3, "prediction_distance": 1.0},
        {"source": 0, "target": 3, "prediction_distance": 50.0},
        {"source": 1, "target": 2, "prediction_distance": 50.0},
        {"source": 2, "target": 4, "prediction_distance": 2.0},
        {"source": 3, "target": 5, "prediction_distance": 2.0},
        {"source": 2, "target": 5, "prediction_distance": 49.0},
        {"source": 3, "target": 4, "prediction_distance": 49.0},
        {"source": 3, "target": 6, "prediction_distance": 3.0},
    ]

    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from([(cell["id"], cell) for cell in cells])
    nx_graph.add_edges_from([(edge["source"], edge["target"], edge) for edge in edges])
    return nx_graph


@pytest.fixture
def arlo_graph(arlo_graph_nx) -> motile.TrackGraph:
    """Return the "Arlo graph" as a :class:`motile.TrackGraph` instance."""
    return motile.TrackGraph(arlo_graph_nx)
