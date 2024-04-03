import numpy as np
from motile_toolbox.candidate_graph.conflict_sets import compute_conflict_sets
from pytest_unordered import unordered


def test_conflict_sets_2d(multi_hypothesis_segmentation_2d):
    for t in range(multi_hypothesis_segmentation_2d.shape[0]):
        conflict_set = compute_conflict_sets(multi_hypothesis_segmentation_2d[t], t)
        if t == 0:
            expected = [{"0_1_1", "0_0_1"}]
            assert len(conflict_set) == 1
            assert conflict_set == unordered(expected)
        elif t == 1:
            expected = [{"1_0_2", "1_1_2"}, {"1_0_1", "1_1_1"}]
            assert len(conflict_set) == 2
            assert conflict_set == unordered(expected)


def test_conflict_sets_2d_reshaped(multi_hypothesis_segmentation_2d):
    """Reshape segmentation array just to provide a slightly difficult example."""

    reshaped = np.asarray(
        [
            multi_hypothesis_segmentation_2d[0, 0],  # hypothesis 0
            multi_hypothesis_segmentation_2d[1, 0],  # hypothesis 1
            multi_hypothesis_segmentation_2d[1, 1],
        ]
    )  # hypothesis 2
    conflict_set = compute_conflict_sets(reshaped, 0)
    # note the expected ids are not really there since the
    # reshaped array is artifically constructed
    expected = [
        {"0_0_1", "0_1_2", "0_2_2"},
        {"0_1_1", "0_2_1"},
        {"0_0_1", "0_1_2"},
        {"0_1_2", "0_2_2"},
        {"0_0_1", "0_2_2"},
    ]
    assert conflict_set == unordered(expected)
