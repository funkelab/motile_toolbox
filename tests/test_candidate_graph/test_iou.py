import pytest
from motile_toolbox.candidate_graph.iou import _compute_ious


def test_compute_ious_2d(segmentation_2d):
    ious = _compute_ious(segmentation_2d[0], segmentation_2d[1])
    expected = {1: {2: 555.46 / 1408.0}}
    assert ious.keys() == expected.keys()
    assert ious[1].keys() == expected[1].keys()
    assert ious[1][2] == pytest.approx(expected[1][2], abs=0.1)

    ious = _compute_ious(segmentation_2d[1], segmentation_2d[1])
    expected = {1: {1: 1.0}, 2: {2: 1.0}}
    assert ious.keys() == expected.keys()
    assert ious[1].keys() == expected[1].keys()
    assert ious[1][1] == pytest.approx(expected[1][1], abs=0.1)
    assert ious[2].keys() == expected[2].keys()
    assert ious[2][2] == pytest.approx(expected[2][2], abs=0.1)


def test_compute_ious_3d(segmentation_3d):
    ious = _compute_ious(segmentation_3d[0], segmentation_3d[1])
    expected = {1: {2: 0.30}}
    assert ious.keys() == expected.keys()
    assert ious[1].keys() == expected[1].keys()
    assert ious[1][2] == pytest.approx(expected[1][2], abs=0.1)

    ious = _compute_ious(segmentation_3d[1], segmentation_3d[1])
    expected = {1: {1: 1.0}, 2: {2: 1.0}}
    assert ious.keys() == expected.keys()
    assert ious[1].keys() == expected[1].keys()
    assert ious[1][1] == pytest.approx(expected[1][1], abs=0.1)
    assert ious[2].keys() == expected[2].keys()
    assert ious[2][2] == pytest.approx(expected[2][2], abs=0.1)
