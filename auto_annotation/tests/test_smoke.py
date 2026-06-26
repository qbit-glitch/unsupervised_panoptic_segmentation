#!/usr/bin/env python3
"""End-to-end smoke test for the auto-annotation scaffold (dummy backends).

Run: .venv_cups_cpu/bin/python -m pytest auto_annotation/tests/test_smoke.py -v
"""

from pathlib import Path

import cv2
import numpy as np

from auto_annotation.config import PipelineConfig
from auto_annotation.pipeline import run
from auto_annotation.schemas import InstanceMask, InstanceResult, SemanticResult
from auto_annotation.stages.boundary_refine import snap_to_sam
from auto_annotation.stages.panoptic_merge import merge_panoptic
from auto_annotation.stages.quality import score_frame
from auto_annotation.taxonomy import THING_IDS, VOID_ID, is_thing, name_to_id


def _make_video(path: Path, n: int = 6, h: int = 64, w: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
    assert vw.isOpened(), "VideoWriter failed to open (codec issue)"
    for i in range(n):
        frame = np.full((h, w, 3), i * 10 % 255, dtype=np.uint8)
        vw.write(frame)
    vw.release()


def test_taxonomy_thing_stuff_disjoint():
    assert name_to_id("auto rickshaw") in THING_IDS
    assert is_thing(name_to_id("car"))
    assert not is_thing(name_to_id("road"))


def test_merge_encoding_and_void():
    h, w = 32, 48
    sem = SemanticResult(
        label_map=np.full((h, w), name_to_id("road"), dtype=np.int32),
        confidence=np.full((h, w), 0.9, dtype=np.float32),
    )
    # a thing region in the semantic map with NO instance -> should become void
    sem.label_map[:8, :8] = name_to_id("car")
    m = np.zeros((h, w), dtype=bool)
    m[16:24, 20:30] = True
    inst = InstanceResult([InstanceMask(mask=m, class_id=name_to_id("car"),
                                        score=0.9, track_id=0)])
    cfg = PipelineConfig()
    pan = merge_panoptic(sem, inst, cfg)

    # the uncovered car region is void
    assert (pan.pan_map[:8, :8] // cfg.label_divisor == VOID_ID).all()
    # the instance is encoded class*div + 1
    car = name_to_id("car")
    assert pan.pan_map[18, 25] == car * cfg.label_divisor + 1
    # one thing segment recorded with its score
    things = [s for s in pan.segments_info if s["isthing"]]
    assert len(things) == 1 and things[0]["score"] == 0.9


def test_snap_relabels_to_majority():
    lab = np.full((16, 16), name_to_id("road"), dtype=np.int32)
    lab[:, 8:] = name_to_id("sidewalk")
    mask = np.zeros((16, 16), dtype=bool)
    mask[:, 6:10] = True  # straddles the road/sidewalk seam, mostly... road side bigger
    out = snap_to_sam(lab, [mask], min_area=1)
    # inside the mask everything is now a single label (snapped)
    assert len(np.unique(out[mask])) == 1


def test_quality_routing_val_always_review():
    sem = SemanticResult(label_map=np.zeros((8, 8), np.int32),
                         confidence=np.full((8, 8), 0.99, np.float32))
    q = score_frame(sem, PipelineConfig(split="val"))
    assert q.route_to_human and q.reason == "val/test-always-verified"


def test_full_pipeline_dummy(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video)
    cfg = PipelineConfig(
        video_path=video,
        output_dir=tmp_path / "out",
        frames_dir=tmp_path / "out" / "frames",
        semantic_backend="dummy",
        instance_backend="dummy",
        keyframe_stride=2,
        split="train",
    )
    manifest = run(cfg)
    assert manifest["summary"]["total"] >= 1
    assert (tmp_path / "out" / "manifest.json").exists()
    # every frame is routed somewhere and has segments
    assert all(f["n_segments"] >= 1 for f in manifest["frames"])
