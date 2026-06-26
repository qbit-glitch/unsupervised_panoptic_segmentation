#!/usr/bin/env python3
"""Merge a semantic label map + thing-instances into a COCO-style panoptic map.

Encoding: panoptic_id = class_id * label_divisor + instance_id  (stuff -> inst 0).
Rules:
  - Stuff pixels keep their semantic class (instance 0).
  - Thing pixels in the semantic map that NO instance covers become VOID — panoptic
    things must have an instance, so an uncovered 'car region' with no mask is not a
    valid segment.
  - Instances paint over stuff/each-other in ASCENDING score order, so the highest-
    scoring mask wins on overlap.
"""

import logging
from typing import Dict

import numpy as np

from ..config import PipelineConfig
from ..schemas import InstanceResult, PanopticResult, SemanticResult
from ..taxonomy import STUFF_IDS, THING_IDS, VOID_ID, is_thing

logger = logging.getLogger(__name__)

__all__ = ["merge_panoptic"]


def merge_panoptic(semantic: SemanticResult, instances: InstanceResult,
                   cfg: PipelineConfig) -> PanopticResult:
    div = cfg.label_divisor
    sem = semantic.label_map.astype(np.int32)
    pan = sem * div  # stuff get instance 0

    # things without an instance are invalid -> void (filled in by instances below)
    thing_pixels = np.isin(sem, list(THING_IDS))
    pan[thing_pixels] = VOID_ID * div

    # paint instances low-score-first so the best mask survives on overlap
    thing_meta: Dict[int, dict] = {}
    per_class_count: Dict[int, int] = {}
    for inst in sorted(instances.instances, key=lambda i: i.score):
        if not cfg.stuff_overlap_is_thing and not is_thing(inst.class_id):
            continue
        per_class_count[inst.class_id] = per_class_count.get(inst.class_id, 0) + 1
        inst_id = per_class_count[inst.class_id]
        pid = inst.class_id * div + inst_id
        pan[inst.mask] = pid
        thing_meta[pid] = {"score": float(inst.score), "track_id": inst.track_id}

    segments_info = _build_segments(pan, div, thing_meta)
    logger.debug("merged panoptic: %d segments (%d things)",
                 len(segments_info), len(thing_meta))
    return PanopticResult(pan_map=pan, segments_info=segments_info, label_divisor=div)


def _build_segments(pan: np.ndarray, div: int, thing_meta: Dict[int, dict]) -> list:
    """Derive segment metadata from the FINAL panoptic map (exact areas)."""
    segments = []
    pids, counts = np.unique(pan, return_counts=True)
    for pid, area in zip(pids.tolist(), counts.tolist()):
        cid = pid // div
        if cid == VOID_ID:
            continue
        seg = {
            "id": int(pid),
            "category_id": int(cid),
            "isthing": is_thing(cid),
            "area": int(area),
        }
        if pid in thing_meta:
            seg.update(thing_meta[pid])
        segments.append(seg)
    return segments
