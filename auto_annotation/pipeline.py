#!/usr/bin/env python3
"""End-to-end orchestrator: raw video -> panoptic labels + review manifest.

Flow per the deep-research design:
    video -> keyframes -> [semantic (INSID3)] + [instances (SAM3)]
          -> SAM-snap boundaries -> panoptic merge -> QA score
          -> route (auto train / human-verify) -> persist + manifest
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .io_utils import extract_keyframes, save_panoptic, save_semantic_png
from .schemas import InstanceResult, SemanticResult
from .stages import build_instance, build_semantic
from .stages.boundary_refine import snap_to_sam
from .stages.panoptic_merge import merge_panoptic
from .stages.quality import score_frame

logger = logging.getLogger(__name__)

__all__ = ["run"]


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def run(cfg: PipelineConfig) -> Dict:
    """Execute the full pipeline. Returns the manifest dict (also written to disk)."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    frames = extract_keyframes(cfg.video_path, cfg.frames_dir,
                               cfg.keyframe_stride, cfg.max_frames)
    if not frames:
        raise RuntimeError(f"no frames extracted from {cfg.video_path}")

    semantic_net = build_semantic(cfg)
    instance_net = build_instance(cfg)

    images = [_load_rgb(p) for p in frames]
    sem_results: List[SemanticResult] = [semantic_net.predict(im) for im in images]
    inst_results: List[InstanceResult] = [instance_net.predict(im) for im in images]
    # video-consistent track ids (no-op for dummy / image-only backends)
    inst_results = instance_net.track(images, inst_results)

    manifest: Dict = {"config": _config_summary(cfg), "frames": []}
    auto, review = 0, 0
    for path, sem, inst in zip(frames, sem_results, inst_results):
        inst = inst.filter_by_score(cfg.instance_score_thr)

        if cfg.use_boundary_refine:
            snap_masks = [i.mask for i in inst.instances]  # + SAM-everything masks
            sem = SemanticResult(
                label_map=snap_to_sam(sem.label_map, snap_masks, cfg.snap_min_mask_area),
                confidence=sem.confidence, margin=sem.margin,
            )

        pan = merge_panoptic(sem, inst, cfg)
        q = score_frame(sem, cfg)

        stem = path.stem
        save_semantic_png(sem.label_map, cfg.output_dir / "semantic" / f"{stem}.png")
        save_panoptic(pan, stem, cfg.output_dir / ("review" if q.route_to_human
                                                   else "auto"))
        manifest["frames"].append({
            "stem": stem,
            "n_instances": len(inst.instances),
            "n_segments": len(pan.segments_info),
            "image_score": round(q.image_score, 4),
            "low_conf_frac": round(q.low_conf_frac, 4),
            "route": "human" if q.route_to_human else "auto",
            "reason": q.reason,
        })
        auto += int(not q.route_to_human)
        review += int(q.route_to_human)

    manifest["summary"] = {"total": len(frames), "auto": auto, "human_review": review}
    out = cfg.output_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    logger.info("done: %d frames -> %d auto, %d for review | %s",
                len(frames), auto, review, out)
    return manifest


def _config_summary(cfg: PipelineConfig) -> Dict:
    return {
        "semantic_backend": cfg.semantic_backend,
        "instance_backend": cfg.instance_backend,
        "split": cfg.split,
        "keyframe_stride": cfg.keyframe_stride,
        "instance_score_thr": cfg.instance_score_thr,
        "route_threshold": cfg.route_threshold,
    }
