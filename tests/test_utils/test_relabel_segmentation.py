import numpy as np
from motile_toolbox.utils import relabel_segmentation
from numpy.testing import assert_array_equal
from skimage.draw import disk


def test_relabel_segmentation(segmentation_2d, graph_2d):
    frame_shape = (100, 100)
    expected = np.zeros(segmentation_2d.shape, dtype="int32")
    # make frame with one cell in center with label 1
    rr, cc = disk(center=(50, 50), radius=20, shape=(100, 100))
    expected[0, 0][rr, cc] = 1

    # make frame with cell centered at (20, 80) with label 1
    rr, cc = disk(center=(20, 80), radius=10, shape=frame_shape)
    expected[1, 0][rr, cc] = 1

    graph_2d.remove_node("1_2")
    relabeled_seg = relabel_segmentation(graph_2d, segmentation_2d)
    print(f"Nonzero relabeled: {np.count_nonzero(relabeled_seg)}")
    print(f"Nonzero expected: {np.count_nonzero(expected)}")
    print(f"Max relabeled: {np.max(relabeled_seg)}")
    print(f"Max expected: {np.max(expected)}")

    assert_array_equal(relabeled_seg, expected)
