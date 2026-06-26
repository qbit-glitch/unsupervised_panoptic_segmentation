#!/usr/bin/env python3
"""Instance stage: SAM 3 promptable concept segmentation (+ dummy fallback).

SAM 3 (Meta/FAIR, arXiv:2511.16719) detects+segments+tracks from text noun-phrases
and/or image exemplars. We prompt it with cfg.instance_prompts (one per thing class)
and keep masks above cfg.instance_score_thr. Its native video tracking provides the
stable track_ids needed for video-panoptic consistency.

Alternative real backend (grounded_sam2): Grounding DINO -> SAM 2
(github.com/IDEA-Research/Grounded-SAM-2) — same interface, swap the loader.
"""

import logging

import numpy as np

from ..schemas import InstanceMask, InstanceResult
from ..taxonomy import THING_IDS, name_to_id
from . import InstanceAnnotator, register_instance

logger = logging.getLogger(__name__)

__all__ = ["Sam3InstanceAnnotator", "DummyInstanceAnnotator"]


@register_instance("sam3")
class Sam3InstanceAnnotator(InstanceAnnotator):
    """Promptable concept instances via SAM 3."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # map each text prompt to a taxonomy class id once
        self._prompt_to_id = {p: name_to_id(p) for p in cfg.instance_prompts}
        self._model = self._load_model()
        logger.info("SAM3 ready: %d prompts, thr=%.2f",
                    len(self._prompt_to_id), cfg.instance_score_thr)

    def _load_model(self):
        # SAM 3 runs on CUDA natively, and on CPU via a compat shim (stub triton,
        # coerce cuda->cpu, force fp32, no-op pin_memory). Verified on Mac CPU:
        # text 'car' on Cityscapes -> 5 masks, scores ~0.95, ~3s/img. CUDA is still
        # preferred for dataset-scale throughput. Weights auto-download from
        # facebook/sam3 (sam3.pt, ~3.45GB) with an accepted-license HF token.
        import sys
        import torch
        use_cuda = self.cfg.device == "cuda" and torch.cuda.is_available()
        if not use_cuda:
            from ..backends.sam3_compat import enable_cpu_sam3
            enable_cpu_sam3()  # MUST run before importing sam3
        sys.path.insert(0, str(self.cfg.sam3_repo))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        dev = "cuda" if use_cuda else "cpu"
        model = build_sam3_image_model(device=dev)
        self._proc = Sam3Processor(model, resolution=1008, device=dev)
        return model

    def predict(self, image: np.ndarray) -> InstanceResult:
        from PIL import Image
        state = self._proc.set_image(Image.fromarray(image))  # encode once, reuse
        insts = []
        for prompt, class_id in self._prompt_to_id.items():
            out = self._proc.set_text_prompt(state=state, prompt=prompt)
            masks, scores = out.get("masks"), out.get("scores")
            if masks is None:
                continue
            m = masks.detach().cpu().numpy()
            if m.ndim == 4:
                m = m[:, 0]
            for i in range(m.shape[0]):
                sc = float(scores[i]) if scores is not None else 1.0
                if sc >= self.cfg.instance_score_thr:
                    insts.append(InstanceMask(m[i] > 0.5, class_id, sc,
                                              track_id=len(insts)))
        return InstanceResult(insts)

    def track(self, images, results):
        # Per-frame ids only (image model). For temporally-consistent ids across a
        # clip, use build_sam3_video_predictor instead (CUDA strongly recommended).
        return results


@register_instance("dummy")
class DummyInstanceAnnotator(InstanceAnnotator):
    """Deterministic placeholder: two 'car' boxes + one 'auto rickshaw' box."""

    def predict(self, image: np.ndarray) -> InstanceResult:
        h, w = image.shape[:2]
        insts = []
        boxes = [
            (name_to_id("car"), 0.92, (int(0.55 * h), int(0.45 * w),
                                       int(0.80 * h), int(0.62 * w))),
            (name_to_id("car"), 0.88, (int(0.58 * h), int(0.64 * w),
                                       int(0.82 * h), int(0.80 * w))),
            (name_to_id("auto rickshaw"), 0.71, (int(0.60 * h), int(0.20 * w),
                                                 int(0.85 * h), int(0.40 * w))),
        ]
        for tid, (cid, score, (y0, x0, y1, x1)) in enumerate(boxes):
            m = np.zeros((h, w), dtype=bool)
            m[y0:y1, x0:x1] = True
            assert cid in THING_IDS
            insts.append(InstanceMask(mask=m, class_id=cid, score=score, track_id=tid))
        return InstanceResult(insts)

    def track(self, images, results):
        # dummy ids are already stable across frames (same boxes)
        return results
