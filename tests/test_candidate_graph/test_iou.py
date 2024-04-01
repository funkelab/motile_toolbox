from motile_toolbox.candidate_graph.iou import compute_ious
import pytest
import numpy as np
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