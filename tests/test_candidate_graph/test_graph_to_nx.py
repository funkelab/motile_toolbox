import networkx as nx
import pytest
from motile import TrackGraph
from motile_toolbox.candidate_graph import graph_to_nx
from networkx.utils import graphs_equal


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


def test_graph_to_nx(graph_3d: nx.DiGraph):
    track_graph = TrackGraph(nx_graph=graph_3d, frame_attribute="t")
    nx_graph = graph_to_nx(track_graph)
    assert graphs_equal(graph_3d, nx_graph)
