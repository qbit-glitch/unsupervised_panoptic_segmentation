#!/usr/bin/env python3
"""IDD-style taxonomy for unstructured South-Asian traffic.

Adapted from the India Driving Dataset 4-level hierarchy (Varma et al., WACV 2019,
arXiv:1811.10200). This is a STARTING POINT — edit CLASSES / THING_IDS to match the
classes you actually annotate. Keeping the thing/stuff split here means the merge
and QA stages need no per-class hardcoding.
"""

from dataclasses import dataclass
from typing import Dict, List, Set

__all__ = ["TaxonomyEntry", "CLASSES", "THING_IDS", "STUFF_IDS", "name_to_id", "is_thing"]


@dataclass(frozen=True)
class TaxonomyEntry:
    id: int
    name: str
    is_thing: bool
    color: tuple  # RGB for visualization


# id, name, is_thing, color  — ids are contiguous from 0; 255 reserved for void.
_RAW: List[tuple] = [
    (0, "road", False, (128, 64, 128)),
    (1, "drivable fallback", False, (81, 0, 81)),      # IDD: unpaved drivable area
    (2, "sidewalk", False, (244, 35, 232)),
    (3, "non-drivable fallback", False, (152, 251, 152)),
    (4, "building", False, (70, 70, 70)),
    (5, "wall", False, (102, 102, 156)),
    (6, "fence", False, (190, 153, 153)),
    (7, "vegetation", False, (107, 142, 35)),
    (8, "sky", False, (70, 130, 180)),
    (9, "pole", False, (153, 153, 153)),
    (10, "traffic sign", False, (220, 220, 0)),
    (11, "billboard", False, (220, 220, 100)),         # common in Indian scenes
    (12, "person", True, (220, 20, 60)),
    (13, "rider", True, (255, 0, 0)),
    (14, "motorcycle", True, (0, 0, 230)),
    (15, "bicycle", True, (119, 11, 32)),
    (16, "auto rickshaw", True, (255, 204, 54)),        # the key local class
    (17, "car", True, (0, 0, 142)),
    (18, "truck", True, (0, 0, 70)),
    (19, "bus", True, (0, 60, 100)),
    (20, "vehicle fallback", True, (136, 143, 153)),    # carts, trailers, oddities
    (21, "animal", True, (160, 82, 45)),                # cows/dogs on road
    (22, "traffic light", False, (250, 170, 30)),
]

CLASSES: Dict[int, TaxonomyEntry] = {
    r[0]: TaxonomyEntry(id=r[0], name=r[1], is_thing=r[2], color=r[3]) for r in _RAW
}

THING_IDS: Set[int] = {cid for cid, e in CLASSES.items() if e.is_thing}
STUFF_IDS: Set[int] = {cid for cid, e in CLASSES.items() if not e.is_thing}
VOID_ID: int = 255

_NAME_TO_ID: Dict[str, int] = {e.name: cid for cid, e in CLASSES.items()}


def name_to_id(name: str) -> int:
    """Map a prompt/class name to a taxonomy id (raises if unknown)."""
    key = name.strip().lower()
    if key not in _NAME_TO_ID:
        raise KeyError(f"'{name}' not in taxonomy; add it to taxonomy._RAW")
    return _NAME_TO_ID[key]


def is_thing(class_id: int) -> bool:
    return class_id in THING_IDS
