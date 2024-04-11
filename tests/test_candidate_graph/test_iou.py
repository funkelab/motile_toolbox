import networkx as nx
import pytest
from motile_toolbox.candidate_graph import EdgeAttr, add_iou
from motile_toolbox.candidate_graph.iou import _compute_ious


def test_compute_ious_2d(segmentation_2d):
    ious = _compute_ious(segmentation_2d[0], segmentation_2d[1])
    expected = [
        (1, 2, 555.46 / 1408.0),
    ]
    for iou, expected_iou in zip(ious, expected):
        assert iou == pytest.approx(expected_iou, abs=0.01)

    ious = _compute_ious(segmentation_2d[1], segmentation_2d[1])
    expected = [(1, 1, 1.0), (2, 2, 1.0)]
    for iou, expected_iou in zip(ious, expected):
        assert iou == pytest.approx(expected_iou, abs=0.01)


def test_compute_ious_3d(segmentation_3d):
    ious = _compute_ious(segmentation_3d[0], segmentation_3d[1])
    expected = [(1, 2, 0.30)]
    for iou, expected_iou in zip(ious, expected):
        assert iou == pytest.approx(expected_iou, abs=0.01)

    ious = _compute_ious(segmentation_3d[1], segmentation_3d[1])
    expected = [(1, 1, 1.0), (2, 2, 1.0)]
    for iou, expected_iou in zip(ious, expected):
        assert iou == pytest.approx(expected_iou, abs=0.01)


def test_add_iou_2d(segmentation_2d, graph_2d):
    expected = graph_2d
    input_graph = graph_2d.copy()
    nx.set_edge_attributes(input_graph, -1, name=EdgeAttr.IOU.value)
    add_iou(input_graph, segmentation_2d)
    for s, t, attrs in expected.edges(data=True):
        assert (
            pytest.approx(attrs[EdgeAttr.IOU.value], abs=0.01)
            == input_graph.edges[(s, t)][EdgeAttr.IOU.value]
        )


def test_multi_hypo_iou_2d(multi_hypothesis_segmentation_2d, multi_hypothesis_graph_2d):
    expected = multi_hypothesis_graph_2d
    input_graph = multi_hypothesis_graph_2d.copy()
    nx.set_edge_attributes(input_graph, -1, name=EdgeAttr.IOU.value)
    add_iou(input_graph, multi_hypothesis_segmentation_2d)
    for s, t, attrs in expected.edges(data=True):
        print(s, t)
        assert (
            pytest.approx(attrs[EdgeAttr.IOU.value], abs=0.01)
            == input_graph.edges[(s, t)][EdgeAttr.IOU.value]
        )
