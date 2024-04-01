from enum import Enum


class NodeAttr(Enum):
    """Node attributes that can be added to candidate graph using the toolbox.
    Note: Motile can flexibly support any custom attributes. The toolbox provides
    implementations of commonly used ones, listed here.
    """
    POS = "position"
    SEG_ID = "segmentation_id"  # TODO: Seg?
    SEG_HYPOTHESIS = "seg_hypothesis"


class EdgeAttr(Enum):
    """Edge attributes that can be added to candidate graph using the toolbox.
    Note: Motile can flexibly support any custom attributes. The toolbox provides
    implementations of commonly used ones, listed here.
    """

    DISTANCE = "distance"
    IOU = "iou"


def add_iou(cand_graph, segmentation) -> None:
    # TODO: implement
    pass

