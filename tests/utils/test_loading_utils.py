from collections import Counter
from pathlib import Path

from motile_toolbox.utils import loading_utils


def test_load_csv_tracks():
    tests_path = Path(__file__).parent.parent
    tracks = loading_utils.load_csv_tracks(tests_path / "data/sample_tracks.csv")
    assert tracks.number_of_nodes() == 107
    assert tracks.number_of_edges() == 105
    cell_42 = tracks.nodes[42]
    assert cell_42["t"] == 11
    assert cell_42["z"] == 100.0
    assert cell_42["y"] == 208.0
    assert cell_42["x"] == 204.0
    assert cell_42["cell_id"] == 42
    assert cell_42["parent_id"] == 38
    assert cell_42["track_id"] == 0
    assert cell_42["radius"] == 18.5
    assert cell_42["name"] == "ABa"
    assert cell_42["div_state"] == 1
    assert Counter(list(tracks.in_edges(42))) == Counter([(38, 42)])
    assert Counter(list(tracks.out_edges(42))) == Counter([(42, 46), (42, 51)])

    # limit frames
    tracks = loading_utils.load_csv_tracks(
        tests_path / "data/sample_tracks.csv", frames=(11, 13)
    )
    assert tracks.number_of_nodes() == 10
    assert tracks.number_of_edges() == 6
    assert Counter(list(tracks.in_edges(42))) == Counter([])
    assert Counter(list(tracks.out_edges(42))) == Counter([(42, 46), (42, 51)])
