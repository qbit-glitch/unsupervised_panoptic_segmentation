#!/usr/bin/env python3
"""CLIP region labeler — open-vocab semantics on crisp masks (FC-CLIP's mechanism).

Given crisp class-agnostic masks (from Mask2Former segments + SAM 3 instances, or any
source), classify each region with frozen CLIP over an OPEN vocabulary. You get crisp
boundaries (from the masks) + open-vocab, domain-robust labels (from CLIP) + a
per-region confidence that doubles as an agreement/QA signal.

Load CLIP BEFORE constructing Sam3Runner (SAM 3's CPU shims corrupt the next
transformers model built after them — same gotcha as Mask2Former).
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from .demo_panoptic import IDX2NAME, NAME2IDX, VOID

logger = logging.getLogger(__name__)

__all__ = ["ClipRegionLabeler", "regions_from_semantic", "regions_from_instances"]

# short prompt templates averaged per class (standard CLIP zero-shot trick)
_TEMPLATES = ("a photo of a {}.", "a street photo containing {}.", "{} in a road scene.")


class ClipRegionLabeler:
    def __init__(self, vocab_names: List[str], device: str = "cpu",
                 model_id: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).eval().to(device)
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.vocab = list(vocab_names)
        self.idx = [NAME2IDX[n] for n in self.vocab]
        self.text_feats = self._encode_text(self.vocab)
        logger.info("CLIP region labeler ready (%s, %d classes)", model_id, len(self.vocab))

    @staticmethod
    def _as_tensor(out) -> torch.Tensor:
        """transformers 5.x may return a ModelOutput from get_*_features; coerce."""
        if torch.is_tensor(out):
            return out
        for attr in ("text_embeds", "image_embeds", "pooler_output"):
            if hasattr(out, attr) and torch.is_tensor(getattr(out, attr)):
                return getattr(out, attr)
        return out.last_hidden_state[:, 0]

    @torch.no_grad()
    def _encode_text(self, names: List[str]) -> torch.Tensor:
        prompts = [t.format(n) for n in names for t in _TEMPLATES]
        toks = self.proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
        f = self._as_tensor(self.model.get_text_features(**toks))
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.reshape(len(names), len(_TEMPLATES), -1).mean(1)   # avg templates
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def classify_regions(self, rgb: np.ndarray, masks: List[np.ndarray],
                         pad: int = 8, batch: int = 32) -> List[Tuple[int, float]]:
        """Return [(class_idx, confidence)] per mask via CLIP zero-shot."""
        crops = []
        for m in masks:
            ys, xs = np.where(m)
            if ys.size == 0:
                crops.append(Image.new("RGB", (32, 32), (128, 128, 128))); continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
            y1, x1 = min(rgb.shape[0], y1 + pad), min(rgb.shape[1], x1 + pad)
            patch = rgb[y0:y1, x0:x1].copy()
            mm = m[y0:y1, x0:x1]
            patch[~mm] = (0.5 * patch[~mm] + 64).astype(np.uint8)  # dim the background
            crops.append(Image.fromarray(patch))
        out = []
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            inp = self.proc(images=chunk, return_tensors="pt").to(self.device)
            f = self._as_tensor(self.model.get_image_features(**inp))
            f = f / f.norm(dim=-1, keepdim=True)
            logits = (100.0 * f @ self.text_feats.T).softmax(dim=-1)
            conf, arg = logits.max(dim=-1)
            for a, c in zip(arg.tolist(), conf.tolist()):
                out.append((self.idx[a], float(c)))
        return out

    def dense_map(self, rgb: np.ndarray, masks: List[np.ndarray]
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Paint each mask with its CLIP label (largest first). Returns (sem, conf)."""
        labels = self.classify_regions(rgb, masks)
        order = sorted(range(len(masks)), key=lambda k: -int(masks[k].sum()))
        sem = np.full(rgb.shape[:2], VOID, dtype=np.int32)
        conf = np.zeros(rgb.shape[:2], dtype=np.float32)
        for k in order:
            cid, cf = labels[k]
            sem[masks[k]] = cid
            conf[masks[k]] = cf
        return sem, conf


def regions_from_semantic(sem: np.ndarray, min_area: int = 400) -> List[np.ndarray]:
    """Connected components of a semantic map → class-agnostic region masks."""
    import cv2
    regions = []
    for cid in np.unique(sem):
        if cid == VOID:
            continue
        n, lab = cv2.connectedComponents((sem == cid).astype(np.uint8))
        for k in range(1, n):
            comp = lab == k
            if comp.sum() >= min_area:
                regions.append(comp)
    return regions


def regions_from_instances(insts: List[dict], min_area: int = 200) -> List[np.ndarray]:
    return [i["mask"] for i in insts if i["mask"].sum() >= min_area]
