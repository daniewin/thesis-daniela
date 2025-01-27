# code adapted, original source: ISEAD project (HITeC e.V.)

import torch
from enum import Enum
from dataclasses import dataclass, field


class ObjectClass(Enum):
    BIRD = "bird"
    MAMMMAL = "mammal"
    ANTHRO = "anthro"
    MISC = "misc"
    UNKNOWN = "unknown"


@dataclass
class BBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    obj_id: int
    obj_class: ObjectClass
    confidence: float

    def __post_init__(self):
        self._check_bbox_coords()

    def get_bbox_dict(self) -> dict:
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "obj_id": self.obj_id,
            "obj_class": self.obj_class,
            "confidence": self.confidence,
        }

    def _check_bbox_coords(self) -> None:
        if self.xmin > self.xmax:
            raise ValueError(f"xmin {self.xmin} > xmax {self.xmax}")
        if self.ymin > self.ymax:
            raise ValueError(f"ymin {self.ymin} > ymax {self.ymax}")


@dataclass
class Segmentation(BBox):
    mask: torch.Tensor
    confidence: float
    xmin: float = field(init=False)
    xmax: float = field(init=False)
    ymin: float = field(init=False)
    ymax: float = field(init=False)

    def __post_init__(self):
        # compute bounding box - relative coordinates
        object_indices = self.mask.nonzero(as_tuple=True)
        self.xmin = object_indices[1].min().item() / self.mask.shape[1]
        self.xmax = object_indices[1].max().item() / self.mask.shape[1]
        self.ymin = object_indices[0].min().item() / self.mask.shape[0]
        self.ymax = object_indices[0].max().item() / self.mask.shape[0]

        # check coords
        self._check_bbox_coords()
