from typing import Any


def _get_location(
    node_data: dict[str, Any], position_keys: tuple[str, ...] | list[str]
) -> list[Any]:
    
    # TODO: Remove this function by storing positions in one attribute called position
    """Convenience function to get the location of a networkx node when each dimension
    is stored in a different attribute.

    Args:
        node_data (dict[str, Any]): Dictionary of attributes of a networkx node.
            Assumes the provided position keys are in the dictionary.
        position_keys (tuple[str, ...] | list[str], optional): Keys to use to get
            location information from node_data (assumes they are present in node_data).
            Defaults to ("z", "y", "x").

    Returns:
        list: _description_
    Raises:
        KeyError if position keys not in node_data
    """
    return [node_data[k] for k in position_keys]

def _get_node_id(time: int, label_id: int, hypothesis_id: int | None) -> str:
   
    if hypothesis_id:
        return f"{time}_{hypothesis_id}_{label_id}"
    else:
        return f"{time}_{label_id}"