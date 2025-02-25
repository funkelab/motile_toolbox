from enum import Enum


class NodeAttr(Enum):
    """Node attributes that can be added to candidate graph using the toolbox.
    Note: Motile can flexibly support any custom attributes. The toolbox provides
    implementations of commonly used ones, listed here.
    """

    POS: str = "pos"
    TIME: str = "time"
    SEG_ID: str = "seg_id"
    SEG_HYPO: str = "seg_hypo"
    TRACK_ID: str = "track_id"

    AREA: str = "area"
    INTENSITY_MEAN: str = "intensity_mean"
    AXIS_MINOR_LENGTH: str = "axis_minor_length"
    AXIS_MAJOR_LENGTH: str = "axis_major_length"
    AXIS_SEMI_MINOR_LENGTH: str = "axis_semi_minor_length"
    PERIMETER: str = "perimeter"
    PIXEL_COUNT: str = "pixel_count"
    CIRCULARITY: str = "circularity"
    VOLUME: str = "volume"
    VOXEL_COUNT: str = "voxel_count"
    SURFACE_AREA: str = "surface_area"
    SPHERICITY: str = "sphericity"


class EdgeAttr(Enum):
    """Edge attributes that can be added to candidate graph using the toolbox.
    Note: Motile can flexibly support any custom attributes. The toolbox provides
    implementations of commonly used ones, listed here.
    """

    IOU: str = "iou"
