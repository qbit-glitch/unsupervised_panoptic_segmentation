#!/usr/bin/env python3
"""Semantic stage: INSID3 in-context segmentation (+ dummy fallback).

INSID3 (Cuttano et al., CVPR 2026 Oral, arXiv:2603.28480, github.com/visinf/INSID3)
is training-free: given one annotated EXEMPLAR of a concept, it segments that concept
in new images from frozen DINOv3 features. We run it once per concept folder under
cfg.exemplar_dir and stack the per-concept masks into a dense label map (later
boundary-refined with SAM-snap).

The real wrapper is a thin seam (see Insid3SemanticAnnotator.predict TODO) — the
deterministic dummy lets you exercise merge/QA/IO with no weights.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..schemas import SemanticResult
from ..taxonomy import CLASSES, STUFF_IDS, name_to_id
from . import SemanticAnnotator, register_semantic

logger = logging.getLogger(__name__)

__all__ = ["Insid3SemanticAnnotator", "DummySemanticAnnotator"]


@register_semantic("insid3")
class Insid3SemanticAnnotator(SemanticAnnotator):
    """In-context semantic annotator backed by frozen DINOv3 + INSID3.

    Exemplar layout (one folder per concept, name must match taxonomy):
        exemplars/road/0001.png + 0001_mask.png
        exemplars/auto rickshaw/0001.png + 0001_mask.png
        ...
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._concepts = self._discover_exemplars(cfg.exemplar_dir)
        self._model = self._load_model()
        logger.info("INSID3 ready: %d concepts, device=%s",
                    len(self._concepts), cfg.device)

    @staticmethod
    def _discover_exemplars(root: Path) -> List[Tuple[int, List[Path]]]:
        concepts: List[Tuple[int, List[Path]]] = []
        if not root.exists():
            logger.warning("exemplar_dir %s missing; INSID3 will label nothing", root)
            return concepts
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            imgs = sorted(p for p in sub.glob("*.png") if "_mask" not in p.stem)
            if imgs:
                concepts.append((name_to_id(sub.name), imgs))
        return concepts

    def _load_model(self):
        # Build INSID3 with a frozen DINOv3 loaded via HF transformers (the gated
        # .pth is unavailable; HF model.safetensors + a thin adapter is equivalent).
        # Validated on Cityscapes: 1-shot road IoU=0.896. (auto_annotation/scripts/)
        import sys
        sys.path.insert(0, str(self.cfg.insid3_repo))
        from models.insid3 import INSID3
        from ..backends.dinov3_hf import DinoV3HFEncoder

        dev = self.cfg.device if self.cfg.device != "mps" else "cpu"  # INSID3 SVD ∉ MPS
        encoder = DinoV3HFEncoder(model_size="small", device=dev)
        model = INSID3(encoder=encoder, image_size=768, svd_components=500,
                       tau=0.6, merge_threshold=0.2, mask_refiner="bilinear",
                       resize_to_orig_size=True, device=dev)
        for p in model.parameters():
            p.requires_grad = False
        return model

    def predict(self, image: np.ndarray) -> SemanticResult:
        from PIL import Image
        rgb = Image.fromarray(image)
        h, w = image.shape[:2]
        label = np.full((h, w), 255, dtype=np.int32)  # 255 = void
        conf = np.full((h, w), 0.4, dtype=np.float32)
        # paint each concept's in-context mask; later concepts (things) override
        for class_id, exemplars in self._concepts:
            self._model.reset_state()
            for img_path in exemplars:
                mask_path = img_path.with_name(f"{img_path.stem}_mask.png")
                if not mask_path.exists():
                    continue
                self._model.set_reference(Image.open(img_path).convert("RGB"),
                                          Image.open(mask_path))
            if self._model._ref_images is None:
                continue
            self._model.set_target(rgb)
            pred = self._model.segment().cpu().numpy().astype(bool)
            if pred.shape != (h, w):
                pred = np.array(Image.fromarray(pred).resize((w, h), Image.NEAREST))
            label[pred] = class_id
            conf[pred] = 0.85
        return SemanticResult(label_map=label, confidence=conf)


@register_semantic("dummy")
class DummySemanticAnnotator(SemanticAnnotator):
    """Deterministic placeholder: sky on top third, building mid, road below.

    Produces a plausible HxW stuff layout + a confidence ramp so the merge, QA, and
    routing stages can be run and unit-tested without any model weights.
    """

    def predict(self, image: np.ndarray) -> SemanticResult:
        h, w = image.shape[:2]
        label = np.full((h, w), name_to_id("road"), dtype=np.int32)
        label[: h // 3] = name_to_id("sky")
        label[h // 3: 2 * h // 3] = name_to_id("building")
        # a vegetation strip on the right margin
        label[:, int(0.9 * w):] = name_to_id("vegetation")
        # confidence: high in region centers, low near horizontal seams (boundaries)
        conf = np.full((h, w), 0.9, dtype=np.float32)
        for seam in (h // 3, 2 * h // 3):
            lo, hi = max(0, seam - 8), min(h, seam + 8)
            conf[lo:hi] = 0.35
        assert set(np.unique(label)).issubset(STUFF_IDS | set(CLASSES))
        return SemanticResult(label_map=label, confidence=conf,
                              margin=(conf - 0.3).clip(0, 1))
