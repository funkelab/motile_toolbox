import numpy as np
from motile_toolbox.utils import (
    ensure_unique_labels,
    relabel_segmentation_with_track_id,
)
from numpy.testing import assert_array_equal
from skimage.draw import disk


def test_relabel_segmentation(segmentation_2d, graph_2d):
    frame_shape = (100, 100)
    expected = np.zeros(segmentation_2d.shape, dtype="int32")
    # make frame with one cell in center with label 1
    rr, cc = disk(center=(50, 50), radius=20, shape=(100, 100))
    expected[0][rr, cc] = 1

    # make frame with cell centered at (20, 80) with label 1
    rr, cc = disk(center=(20, 80), radius=10, shape=frame_shape)
    expected[1][rr, cc] = 1

    graph_2d.remove_node("1_2")
    relabeled_seg = relabel_segmentation_with_track_id(graph_2d, segmentation_2d)
    print(f"Nonzero relabeled: {np.count_nonzero(relabeled_seg)}")
    print(f"Nonzero expected: {np.count_nonzero(expected)}")
    print(f"Max relabeled: {np.max(relabeled_seg)}")
    print(f"Max expected: {np.max(expected)}")

    assert_array_equal(relabeled_seg, expected)


def test_ensure_unique_labels_2d(segmentation_2d):
    expected = segmentation_2d.copy().astype(np.uint64)
    frame = expected[1]
    frame[frame == 2] = 3
    frame[frame == 1] = 2
    expected[1] = frame

    print(np.unique(expected[1], return_counts=True))
    result = ensure_unique_labels(segmentation_2d)
    assert_array_equal(expected, result)


def test_ensure_unique_labels_2d_multiseg(multi_hypothesis_segmentation_2d):
    expected = multi_hypothesis_segmentation_2d.copy().astype(np.uint64)

    # add 1 to the first hypothesis second frame
    h0f1 = expected[0, 1]
    h0f1[h0f1 == 2] = 3
    h0f1[h0f1 == 1] = 2
    expected[0, 1] = h0f1
    # add 3 to the second hypothesis first frame
    h1f0 = expected[1, 0]
    h1f0[h1f0 == 1] = 4
    expected[1, 0] = h1f0
    # add 4 to the second hypothesis second frame
    h1f1 = expected[1, 1]
    h1f1[h1f1 == 1] = 5
    h1f1[h1f1 == 2] = 6
    expected[1, 1] = h1f1
    result = ensure_unique_labels(multi_hypothesis_segmentation_2d, multiseg=True)
    assert_array_equal(expected, result)
