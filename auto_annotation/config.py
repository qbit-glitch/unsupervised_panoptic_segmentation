#!/usr/bin/env python3
"""Immutable configuration for the monocular-video auto-annotation pipeline.

All hyperparameters live here (no hardcoded constants in stage code). The config
is frozen so it can be hashed, logged, and saved alongside outputs for repro.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

__all__ = ["PipelineConfig"]


@dataclass(frozen=True)
class PipelineConfig:
    """Single source of truth for the whole pipeline.

    Paths default to the repo's auto_annotation/ workspace; point data_root at
    your own dashcam footage. device defaults to mps (M4 Pro) but falls back to
    cpu if unavailable (resolved at runtime, not here).
    """

    # --- I/O ---
    video_path: Path = Path("auto_annotation/data/raw_video.mp4")
    output_dir: Path = Path("auto_annotation/outputs")
    frames_dir: Path = Path("auto_annotation/outputs/frames")

    # --- frame sampling ---
    keyframe_stride: int = 5          # keep every Nth frame
    max_frames: int = 0               # 0 = no cap

    # --- backends (registry keys; "dummy" runs end-to-end with no weights) ---
    semantic_backend: str = "dummy"   # "insid3" | "fcclip" | "dummy"
    instance_backend: str = "dummy"   # "sam3" | "grounded_sam2" | "dummy"
    device: str = "mps"               # "mps" | "cuda" | "cpu"

    # --- INSID3 in-context exemplars ---
    # Directory of support examples: exemplars/<class_name>/*.png (image) with a
    # paired *_mask.png. One folder per concept you want propagated.
    exemplar_dir: Path = Path("auto_annotation/data/exemplars")
    insid3_repo: Path = Path("external/INSID3")     # cloned github.com/visinf/INSID3
    insid3_weights: Path = Path("weights/dinov3")    # frozen DINOv3 backbone

    # --- SAM 3 / instance prompts ---
    sam3_repo: Path = Path("external/sam3")
    sam3_weights: Path = Path("weights/sam3.pt")
    instance_score_thr: float = 0.30
    # text prompts for promptable concept segmentation (edit for your taxonomy)
    instance_prompts: Tuple[str, ...] = (
        "auto rickshaw", "car", "truck", "bus", "motorcycle",
        "bicycle", "person", "rider", "animal", "cart",
    )

    # --- boundary refinement (SAM-snap) ---
    use_boundary_refine: bool = True
    snap_min_mask_area: int = 50      # ignore SAM masks smaller than this (px)

    # --- panoptic merge ---
    label_divisor: int = 1000
    stuff_overlap_is_thing: bool = True  # instances paint over stuff on overlap

    # --- QA / human-review routing ---
    low_conf_quantile: float = 0.10      # cleanlab-style soft-min quantile
    route_threshold: float = 0.60        # frame label-quality below -> human review
    low_conf_pixel_frac_thr: float = 0.20  # frac low-conf pixels above -> review

    # --- run control ---
    split: str = "train"              # "train" -> auto, "val"/"test" -> force review
    seed: int = 42
    log_level: str = "INFO"

    def resolved(self, project_root: Path) -> "PipelineConfig":
        """Return a copy with all relative paths anchored to project_root."""

        def anchor(p: Path) -> Path:
            return p if p.is_absolute() else (project_root / p)

        return PipelineConfig(
            video_path=anchor(self.video_path),
            output_dir=anchor(self.output_dir),
            frames_dir=anchor(self.frames_dir),
            keyframe_stride=self.keyframe_stride,
            max_frames=self.max_frames,
            semantic_backend=self.semantic_backend,
            instance_backend=self.instance_backend,
            device=self.device,
            exemplar_dir=anchor(self.exemplar_dir),
            insid3_repo=anchor(self.insid3_repo),
            insid3_weights=anchor(self.insid3_weights),
            sam3_repo=anchor(self.sam3_repo),
            sam3_weights=anchor(self.sam3_weights),
            instance_score_thr=self.instance_score_thr,
            instance_prompts=self.instance_prompts,
            use_boundary_refine=self.use_boundary_refine,
            snap_min_mask_area=self.snap_min_mask_area,
            label_divisor=self.label_divisor,
            stuff_overlap_is_thing=self.stuff_overlap_is_thing,
            low_conf_quantile=self.low_conf_quantile,
            route_threshold=self.route_threshold,
            low_conf_pixel_frac_thr=self.low_conf_pixel_frac_thr,
            split=self.split,
            seed=self.seed,
            log_level=self.log_level,
        )
