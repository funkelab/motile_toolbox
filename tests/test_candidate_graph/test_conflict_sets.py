import numpy as np
from motile_toolbox.candidate_graph.conflict_sets import compute_conflict_sets
from pytest_unordered import unordered


def test_conflict_sets_2d(multi_hypothesis_segmentation_2d):
    for t in range(multi_hypothesis_segmentation_2d.shape[0]):
        conflict_set = compute_conflict_sets(multi_hypothesis_segmentation_2d[:, t])
        if t == 0:
            expected = [{2, 1}]
            assert len(conflict_set) == 1
            assert conflict_set == unordered(expected)
        elif t == 1:
            expected = [{3, 4}, {5, 6}]
            assert len(conflict_set) == 2
            assert conflict_set == unordered(expected)


def test_conflict_sets_2d_reshaped(multi_hypothesis_segmentation_2d):
    """Reshape segmentation array just to provide a slightly difficult example."""

    reshaped = np.asarray(
        [
            multi_hypothesis_segmentation_2d[0, 0],  # hypothesis 0
            multi_hypothesis_segmentation_2d[0, 1],  # hypothesis 1
            multi_hypothesis_segmentation_2d[
                1, 1
            ],  # hypothesis 2 (time 1 hypothesis 1)
        ]
    )  # this is simulating one frame of multi hypothesis data
    conflict_set = compute_conflict_sets(reshaped)
    # note the expected ids are not really there since the
    # reshaped array is artifically constructed

    expected = [
        {1, 5, 6},
        {3, 4},
        {1, 5},
        {5, 6},
        {1, 6},
    ]
    assert conflict_set == unordered(expected)
