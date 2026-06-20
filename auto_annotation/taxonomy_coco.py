"""COCO-133 panoptic taxonomy (contiguous idx 0..132).

Single source of truth = ``panoptic_val2017.json`` categories sorted by ``id``.
Pure stdlib + numpy so it imports in BOTH ``.venv`` and ``.venv_cups_cpu`` (no
detectron2 dependency, which would break the CPU auto-label venv).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_GT_JSON = (Path(os.environ.get("COCO_ROOT", "/Volumes/code_files/datasets/coco"))
            / "annotations/panoptic_val2017.json")
VOID_IDX = 255


@dataclass(frozen=True)
class CocoClass:
    idx: int
    name: str
    is_thing: bool
    color: tuple


def _color(idx: int) -> tuple:
    """Deterministic, dependency-free RGB for visualization (offset off pure black)."""
    return (40 + (idx * 97) % 216, 40 + (idx * 57) % 216, 40 + (idx * 137) % 216)


@lru_cache(maxsize=1)
def _load() -> dict:
    cats = json.loads(_GT_JSON.read_text())["categories"]
    cats = sorted(cats, key=lambda c: c["id"])
    return {
        i: CocoClass(i, c["name"], bool(c["isthing"]), _color(i))
        for i, c in enumerate(cats)
    }


COCO_CLASSES = _load()
_NAME2IDX = {c.name: i for i, c in COCO_CLASSES.items()}
THING_IDXS = frozenset(i for i, c in COCO_CLASSES.items() if c.is_thing)
STUFF_IDXS = frozenset(i for i, c in COCO_CLASSES.items() if not c.is_thing)


def name_to_idx(name: str) -> int:
    return _NAME2IDX[name]


def is_thing(idx: int) -> bool:
    return idx in THING_IDXS
