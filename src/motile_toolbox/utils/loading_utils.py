import csv
import logging
from typing import Any, MutableMapping

import networkx as nx
import numpy as np
import zarr

logger = logging.getLogger(__name__)


def load_csv_tracks(
    tracks_path: str, frames: tuple[int, int] | None = None
) -> nx.DiGraph:
    """Load tracks from a csv to a networkx graph.
    Expects the following tab-separated columns (from mskcc-confocal):
        t z y x cell_id parent_id track_id radius name div_state
    But everything after parent_id is optional, as technically are z y and x.

    Args:
        tracks_path (str): path to tracks file
        frames (tuple, optional): Tuple of start frame, end frame to limit the tracks to
            these time points. Includes start frame, excludes end frame. Defaults to
            None.

    Returns: nx.DiGraph where nodes have all the attributes from the columns of the csv,
        and edges have no attributes. Edges go forward in time from parent to child.
    """
    graph = nx.DiGraph()
    with open(tracks_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        # t	z	y	x	cell_id	parent_id	track_id	radius	name	div_state
        for row in reader:
            cell = _convert_types(row)
            if frames:
                time = cell["t"]
                if time < frames[0] or time >= frames[1]:
                    continue
            cell_id = cell["cell_id"]
            graph.add_node(cell["cell_id"], **cell)
            parent_id = cell["parent_id"]
            if parent_id != -1:
                # only add edge if it falls within the specified frames
                if (not frames) or time > frames[0]:
                    graph.add_edge(parent_id, cell_id)
    logger.info(
        f"Loaded {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges "
        f"from {tracks_path}."
    )
    return graph


def _convert_types(row: dict[str, str]) -> MutableMapping[str, Any]:
    """Helper function for loading the tracks csv with correct types
    Converts certain columns to int and others to float - unlisted
    columns are left as strings.

    Args:
        row (dict[str, str]): Row from csv.DictReader

    Returns:
        dict: Same row with the types converted from strings to ints/floats
        for the appropriate keys.
    """
    int_vals = ["t", "cell_id", "parent_id", "track_id", "div_state"]
    float_vals = ["z", "y", "x", "radius"]
    converted_row: MutableMapping[str, Any] = {}
    for key, val in row.items():
        if key in int_vals:
            converted_row[key] = int(val)
        elif key in float_vals:
            converted_row[key] = float(val)
        else:
            converted_row[key] = val
    return converted_row


def load_cellulus_results(
    path_to_zarr: str,
    image_group: str = "test/raw",
    seg_group: str = "post-processed-segmentation",
    seg_channel: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load images and segmentations from custom cellulus zarr.

    Args:
        path_to_zarr (str): Path to zarr containing cellulus results.
        image_group (str, optional): Zarr group containing raw images. Defaults to
            "test/raw".
        seg_group (str, optional): Zarr group containing segmentation.
            Defaults to "post-processed-segmentation".
        seg_channel (int, optional): Channel of segmentation to use. Segmentation
            channels correspond to levels in multi-hypothesis cellulus. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: an array with t, z, y, x for raw images and
            segmentation results.
    """
    base = zarr.open(path_to_zarr, "r")
    images = base[image_group]
    segmentation = base[seg_group][
        :, seg_channel
    ]  # orginally t, c, z, y, x. want to select channel

    # should return (t, z, y, x) for both
    return np.squeeze(images), np.squeeze(segmentation)
