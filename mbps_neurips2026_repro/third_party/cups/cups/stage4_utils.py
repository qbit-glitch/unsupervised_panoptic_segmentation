from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from yacs.config import CfgNode

from cups.data import CITYSCAPES_CLASSNAMES

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage4ClassIds:
    """Resolved class ids for Stage-4 losses.

    ``rare_stuff_targets`` are semantic-head target ids where 0 is the shared
    thing-region class and 1..S are the current stuff pseudo classes.
    ``rare_thing_targets`` are ROI-head target ids in the current contiguous
    thing-class space.
    """

    rare_original_ids: Tuple[int, ...]
    rare_stuff_targets: Tuple[int, ...]
    rare_thing_targets: Tuple[int, ...]
    unresolved: Tuple[str, ...]


def _normalise_name(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")


def _as_int_tuple(values: Iterable[object] | None) -> Tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(v) for v in values)


def resolve_stage4_class_ids(
    config: CfgNode,
    thing_pseudo_classes: Sequence[int] | None,
    stuff_pseudo_classes: Sequence[int] | None,
) -> Stage4ClassIds:
    """Resolve Stage-4 rare classes into current CUPS target spaces.

    The common 27-class CAUSE/Cityscapes case resolves from class names. For
    overclustered k-class training, callers can provide explicit
    ``STAGE4.RARE_*_PSEUDO_CLASSES`` ids.
    """

    stage4 = getattr(config, "STAGE4", None)
    if stage4 is None or not getattr(stage4, "ENABLED", False):
        return Stage4ClassIds((), (), (), ())

    thing_pseudo_classes = tuple(int(c) for c in (thing_pseudo_classes or ()))
    stuff_pseudo_classes = tuple(int(c) for c in (stuff_pseudo_classes or ()))

    name_to_id = {_normalise_name(name): idx for idx, name in enumerate(CITYSCAPES_CLASSNAMES)}
    rare_original_ids = []
    unresolved = []
    for name in getattr(stage4, "RARE_CLASSES", ()):
        key = _normalise_name(name)
        if key in name_to_id:
            rare_original_ids.append(name_to_id[key])
        else:
            unresolved.append(str(name))

    explicit_stuff = _as_int_tuple(getattr(stage4, "RARE_STUFF_PSEUDO_CLASSES", ()))
    explicit_things = _as_int_tuple(getattr(stage4, "RARE_THING_PSEUDO_CLASSES", ()))

    rare_stuff_targets = []
    rare_thing_targets = []
    for class_id in rare_original_ids:
        if class_id in stuff_pseudo_classes:
            rare_stuff_targets.append(stuff_pseudo_classes.index(class_id) + 1)
        elif class_id in thing_pseudo_classes:
            rare_thing_targets.append(thing_pseudo_classes.index(class_id))
        elif class_id not in explicit_stuff and class_id not in explicit_things:
            unresolved.append(CITYSCAPES_CLASSNAMES[class_id])

    # Explicit ids are interpreted as current CUPS target-space ids:
    # stuff semantic target ids are 1..S, ROI thing ids are 0..T-1.
    rare_stuff_targets.extend(explicit_stuff)
    rare_thing_targets.extend(explicit_things)

    ids = Stage4ClassIds(
        rare_original_ids=tuple(dict.fromkeys(rare_original_ids)),
        rare_stuff_targets=tuple(dict.fromkeys(int(v) for v in rare_stuff_targets)),
        rare_thing_targets=tuple(dict.fromkeys(int(v) for v in rare_thing_targets)),
        unresolved=tuple(dict.fromkeys(unresolved)),
    )
    if ids.unresolved:
        log.warning(
            "Stage-4 rare classes unresolved in current CUPS split: %s. "
            "Use STAGE4.RARE_*_PSEUDO_CLASSES for k-way overclustered runs.",
            ", ".join(ids.unresolved),
        )
    return ids


def apply_stage4_detectron_cfg(cfg: CfgNode, stage4: CfgNode | None, ids: Stage4ClassIds | None = None) -> None:
    """Attach Stage-4 knobs to the Detectron2 cfg before model construction."""

    if stage4 is None:
        return
    ids = ids or Stage4ClassIds((), (), (), ())
    enabled = bool(getattr(stage4, "ENABLED", False))

    cfg.MODEL.ROI_BOX_HEAD.USE_EQLV2 = enabled and bool(getattr(stage4, "USE_EQLV2", False))
    cfg.MODEL.ROI_BOX_HEAD.EQLV2_GAMMA = float(getattr(stage4, "EQLV2_GAMMA", 12.0))
    cfg.MODEL.ROI_BOX_HEAD.EQLV2_MU = float(getattr(stage4, "EQLV2_MU", 0.8))
    cfg.MODEL.ROI_BOX_HEAD.EQLV2_ALPHA = float(getattr(stage4, "EQLV2_ALPHA", 4.0))
    cfg.MODEL.ROI_BOX_HEAD.USE_SEESAW_LOSS = enabled and bool(getattr(stage4, "USE_SEESAW_LOSS", False))
    cfg.MODEL.ROI_BOX_HEAD.SEESAW_P = float(getattr(stage4, "SEESAW_P", 0.8))
    cfg.MODEL.ROI_BOX_HEAD.SEESAW_Q = float(getattr(stage4, "SEESAW_Q", 2.0))
    cfg.MODEL.ROI_BOX_HEAD.RARE_CLASSES = ids.rare_thing_targets
    cfg.MODEL.ROI_BOX_HEAD.RARE_LOSS_WEIGHT = float(getattr(stage4, "RARE_LOSS_WEIGHT", 1.0))
    cfg.MODEL.ROI_BOX_HEAD.OHEM_ENABLED = enabled and bool(getattr(stage4, "OHEM_ENABLED", False))
    cfg.MODEL.ROI_BOX_HEAD.OHEM_FRACTION = float(getattr(stage4, "OHEM_FRACTION", 0.25))
    cfg.MODEL.ROI_BOX_HEAD.OHEM_MIN_KEPT = int(getattr(stage4, "OHEM_MIN_KEPT", 16))
    cfg.MODEL.ROI_BOX_HEAD.OHEM_RARE_MIN_KEPT = int(getattr(stage4, "OHEM_RARE_MIN_KEPT", 2))

    cfg.MODEL.ROI_HEADS.STAGE4_RARE_RETAIN_ENABLED = (
        enabled and bool(getattr(stage4, "RARE_RETAIN_ENABLED", False))
    )
    cfg.MODEL.ROI_HEADS.STAGE4_RARE_CLASSES = ids.rare_thing_targets
    cfg.MODEL.ROI_HEADS.STAGE4_RARE_RETAIN_MIN_ROIS = int(getattr(stage4, "RARE_RETAIN_MIN_ROIS", 2))
    cfg.MODEL.ROI_HEADS.STAGE4_RARE_RETAIN_RELAXED_IOU = float(
        getattr(stage4, "RARE_RETAIN_RELAXED_IOU", 0.35)
    )

    cfg.MODEL.SEM_SEG_HEAD.RARE_STUFF_CLASSES = ids.rare_stuff_targets
    cfg.MODEL.SEM_SEG_HEAD.RARE_FOCAL_WEIGHT = float(getattr(stage4, "RARE_FOCAL_WEIGHT", 0.0)) if enabled else 0.0
    cfg.MODEL.SEM_SEG_HEAD.RARE_FOCAL_GAMMA = float(getattr(stage4, "RARE_FOCAL_GAMMA", 2.0))
    cfg.MODEL.SEM_SEG_HEAD.RARE_BOUNDARY_WEIGHT = (
        float(getattr(stage4, "RARE_BOUNDARY_WEIGHT", 0.0)) if enabled else 0.0
    )
    cfg.MODEL.SEM_SEG_HEAD.RARE_BOUNDARY_WIDTH = int(getattr(stage4, "RARE_BOUNDARY_WIDTH", 3))


def apply_model_long_tail_detectron_cfg(
    cfg: CfgNode,
    roi_box_head: CfgNode | None = None,
    sem_seg_head: CfgNode | None = None,
) -> None:
    """Attach non-Stage-4 long-tail knobs to the Detectron2 cfg."""

    if roi_box_head is not None:
        use_eqlv2 = bool(getattr(roi_box_head, "USE_EQLV2", False)) or bool(
            getattr(cfg.MODEL.ROI_BOX_HEAD, "USE_EQLV2", False)
        )
        use_seesaw = bool(getattr(roi_box_head, "USE_SEESAW_LOSS", False)) or bool(
            getattr(cfg.MODEL.ROI_BOX_HEAD, "USE_SEESAW_LOSS", False)
        )
        if use_eqlv2 and use_seesaw:
            raise ValueError("MODEL.ROI_BOX_HEAD.USE_EQLV2 and USE_SEESAW_LOSS are mutually exclusive.")
        cfg.MODEL.ROI_BOX_HEAD.USE_EQLV2 = use_eqlv2
        cfg.MODEL.ROI_BOX_HEAD.EQLV2_GAMMA = float(getattr(roi_box_head, "EQLV2_GAMMA", 12.0))
        cfg.MODEL.ROI_BOX_HEAD.EQLV2_MU = float(getattr(roi_box_head, "EQLV2_MU", 0.8))
        cfg.MODEL.ROI_BOX_HEAD.EQLV2_ALPHA = float(getattr(roi_box_head, "EQLV2_ALPHA", 4.0))
        cfg.MODEL.ROI_BOX_HEAD.USE_SEESAW_LOSS = use_seesaw
        cfg.MODEL.ROI_BOX_HEAD.SEESAW_P = float(getattr(roi_box_head, "SEESAW_P", 0.8))
        cfg.MODEL.ROI_BOX_HEAD.SEESAW_Q = float(getattr(roi_box_head, "SEESAW_Q", 2.0))

    if sem_seg_head is not None:
        cfg.MODEL.SEM_SEG_HEAD.LDAM_ENABLED = bool(getattr(sem_seg_head, "LDAM_ENABLED", False))
        cfg.MODEL.SEM_SEG_HEAD.LDAM_MAX_MARGIN = float(getattr(sem_seg_head, "LDAM_MAX_MARGIN", 0.5))
        cfg.MODEL.SEM_SEG_HEAD.LDAM_S = float(getattr(sem_seg_head, "LDAM_S", 30.0))
        cfg.MODEL.SEM_SEG_HEAD.LDAM_CLASS_FREQ = tuple(getattr(sem_seg_head, "LDAM_CLASS_FREQ", ()))
