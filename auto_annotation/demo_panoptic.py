#!/usr/bin/env python3
"""Self-contained Cityscapes panoptic demo: INSID3 semantics + SAM 3 instances.

Pipeline on a single RGB image (all CPU):
  semantic  = INSID3 in-context segmentation, one concept at a time (stuff classes),
              reference image+mask drawn from gtFine of OTHER images.
  instances = SAM 3 promptable concept segmentation (thing classes, text prompts).
  panoptic  = merge (instances paint over stuff; COCO id = class*divisor + inst).

Used by scripts/run_panoptic_demo.py and the notebook
notebooks/auto_annotation_panoptic_demo.ipynb. Everything here is inference-only.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

# ----------------------------- Cityscapes palette -----------------------------
# (name, gtFine labelId, is_thing, RGB) — the standard 19 train classes.
_CS = [
    ("road", 7, False, (128, 64, 128)), ("sidewalk", 8, False, (244, 35, 232)),
    ("building", 11, False, (222, 184, 135)), ("wall", 12, False, (102, 102, 156)),  # building tan (was 70,70,70 dark-gray => invisible on overlay)
    ("fence", 13, False, (190, 153, 153)), ("pole", 17, False, (153, 153, 153)),
    ("traffic light", 19, False, (250, 170, 30)), ("traffic sign", 20, False, (220, 220, 0)),
    ("vegetation", 21, False, (107, 142, 35)), ("terrain", 22, False, (152, 251, 152)),
    ("sky", 23, False, (70, 130, 180)), ("person", 24, True, (220, 20, 60)),
    ("rider", 25, True, (255, 0, 0)), ("car", 26, True, (0, 0, 142)),
    ("truck", 27, True, (0, 0, 70)), ("bus", 28, True, (0, 60, 100)),
    ("train", 31, True, (0, 80, 100)), ("motorcycle", 32, True, (0, 0, 230)),
    ("bicycle", 33, True, (119, 11, 32)),
    # --- Indian/unstructured-traffic things (labelId -1: no Cityscapes GT) ---
    ("auto rickshaw", -1, True, (255, 140, 0)), ("cow", -1, True, (160, 82, 45)),
    ("dog", -1, True, (255, 215, 0)), ("cart", -1, True, (75, 0, 130)),
]
NAME2IDX = {n: i for i, (n, *_ ) in enumerate(_CS)}
IDX2NAME = {i: n for n, i in NAME2IDX.items()}
IDX2COLOR = {i: c for i, (_, _, _, c) in enumerate(_CS)}
IDX2LABELID = {i: lid for i, (_, lid, _, _) in enumerate(_CS)}
THING_IDX = {i for i, (_, _, t, _) in enumerate(_CS) if t}
VOID = 255

# stuff painted background->foreground (later wins on overlap)
STUFF_ORDER = ["sky", "building", "wall", "fence", "vegetation", "terrain",
               "pole", "traffic light", "traffic sign", "sidewalk", "road"]
# SAM3 paint order: large background first, road/sidewalk, then thin classes ON TOP
STUFF_ORDER_SAM3 = ["sky", "vegetation", "terrain", "building", "wall", "fence",
                    "road", "sidewalk", "pole", "traffic sign", "traffic light"]
THING_PROMPTS = ["car", "truck", "bus", "train", "person", "rider",
                 "bicycle", "motorcycle"]


@dataclass
class DemoConfig:
    cityscapes_root: Path = Path("/Volumes/code_files/datasets/cityscapes")
    split_city: str = "val/frankfurt"
    model_size: str = "small"          # DINOv3 size: small | base | large
    image_size: int = 768              # INSID3 input resolution
    sam3_resolution: int = 1008        # SAM3 canonical resolution (RoPE table)
    device: str = "cpu"
    instance_score_thr: float = 0.5
    stuff_score_thr: float = 0.30      # SAM3 stuff masks score lower than things
    label_divisor: int = 1000
    stuff_classes: List[str] = field(default_factory=lambda: list(STUFF_ORDER))
    stuff_order_sam3: List[str] = field(default_factory=lambda: list(STUFF_ORDER_SAM3))
    thing_prompts: List[str] = field(default_factory=lambda: list(THING_PROMPTS))


# ------------------------------- model loading --------------------------------
def load_insid3(cfg: DemoConfig):
    sys.path.insert(0, str(ROOT / "external" / "INSID3"))
    from models.insid3 import INSID3
    from auto_annotation.backends.dinov3_hf import DinoV3HFEncoder
    enc = DinoV3HFEncoder(model_size=cfg.model_size, device=cfg.device)
    model = INSID3(encoder=enc, image_size=cfg.image_size, svd_components=500,
                   tau=0.6, merge_threshold=0.2, mask_refiner="bilinear",
                   resize_to_orig_size=True, device=cfg.device)
    for p in model.parameters():
        p.requires_grad = False
    logger.info("INSID3 ready (DINOv3-%s, %dpx)", cfg.model_size, cfg.image_size)
    return model


class Sam3Runner:
    """Thin SAM 3 image wrapper: encode once, prompt many concepts."""

    def __init__(self, cfg: DemoConfig):
        from auto_annotation.backends.sam3_compat import enable_cpu_sam3
        if cfg.device != "cuda":
            enable_cpu_sam3()
        sys.path.insert(0, str(ROOT / "external" / "sam3"))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        dev = "cuda" if cfg.device == "cuda" else "cpu"
        self.model = build_sam3_image_model(device=dev)
        # SAM3's forward hard-casts some activations to bf16 (sam3_image.py); on Ampere+ it runs
        # under a bf16 autocast so the fp32 weights get cast to match. Our processor path had NO
        # autocast -> bf16 act vs fp32 weight mismatch. Fix = standard AMP: KEEP weights fp32 and
        # wrap the call in bf16 autocast (predict). Do NOT convert weights to bf16 — the decoder
        # FFN runs autocast-DISABLED on fp32 LayerNorm output, so it needs fp32 weights to match.
        self._bf16 = (dev == "cuda")
        self.proc = Sam3Processor(self.model, resolution=cfg.sam3_resolution, device=dev)
        self.thr = cfg.instance_score_thr
        logger.info("SAM3 ready (res=%d, thr=%.2f, bf16=%s)", cfg.sam3_resolution, self.thr, self._bf16)

    @torch.no_grad()
    def predict(self, pil_img: Image.Image, prompts: List[str], thr=None) -> List[dict]:
        from contextlib import nullcontext
        thr = self.thr if thr is None else thr
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if self._bf16 else nullcontext()
        with amp:
            state = self.proc.set_image(pil_img)
            dets = []
            for p in prompts:
                out = self.proc.set_text_prompt(state=state, prompt=p)
                masks, scores = out.get("masks"), out.get("scores")
                if masks is None:
                    continue
                m = masks.detach().float().cpu().numpy()   # bf16 -> float (numpy has no bf16)
                if m.ndim == 4:
                    m = m[:, 0]
                for i in range(m.shape[0]):
                    sc = float(scores[i]) if scores is not None else 1.0
                    if sc >= thr:
                        dets.append({"mask": m[i] > 0.5, "prompt": p, "score": sc})
        return dets


# --------------------------------- data utils ---------------------------------
# Mapillary-Vistas (65 cls) -> our Cityscapes-19 STUFF names (things/ambiguous -> None).
MAPILLARY_TO_CS = {
    13: "road", 7: "road", 14: "road", 8: "road", 24: "road", 23: "road", 10: "road",
    43: "road", 41: "road", 36: "road",
    15: "sidewalk", 2: "sidewalk", 9: "sidewalk", 11: "sidewalk",
    17: "building", 16: "building", 18: "building", 35: "building", 32: "building",
    6: "wall", 3: "fence", 4: "fence", 5: "fence",
    45: "pole", 47: "pole", 46: "pole", 44: "pole",
    48: "traffic light", 50: "traffic sign", 49: "traffic sign",
    30: "vegetation", 29: "terrain", 26: "terrain", 25: "terrain", 28: "terrain",
    31: "terrain", 27: "sky",
}


class Mask2FormerStuffRunner:
    """Closed-set dense stuff segmenter (Mask2Former, Mapillary). Maps to CS-19.

    Dense (every pixel labelled) + crisp boundaries, but carries the train-domain gap:
    strong on structured roads, weaker on unstructured Indian scenes (re-test there).
    """

    REPO = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

    def __init__(self, cfg: DemoConfig, long_side: int = 1024):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.device = cfg.device
        self.proc = AutoImageProcessor.from_pretrained(self.REPO)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.REPO).eval().to(self.device)
        self.long_side = long_side
        self.lut = np.full(65, VOID, dtype=np.int32)
        for mid, name in MAPILLARY_TO_CS.items():
            self.lut[mid] = NAME2IDX[name]
        logger.info("Mask2Former-Mapillary ready (swin-L)")

    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> np.ndarray:
        W, H = pil_img.size
        small = pil_img
        if max(W, H) > self.long_side:
            s = self.long_side / max(W, H)
            small = pil_img.resize((int(W * s), int(H * s)), Image.BILINEAR)
        inp = self.proc(images=small, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        mp = self.proc.post_process_semantic_segmentation(
            out, target_sizes=[(H, W)])[0].cpu().numpy().astype(np.int32)
        return self.lut[np.clip(mp, 0, 64)]   # Mapillary id -> CS stuff idx (else VOID)


def gt_labelids(cfg: DemoConfig, img_path: Path) -> np.ndarray:
    stem = img_path.name.replace("_leftImg8bit.png", "")
    gp = cfg.cityscapes_root / "gtFine" / cfg.split_city / f"{stem}_gtFine_labelIds.png"
    return np.array(Image.open(gp))


def build_stuff_refs(cfg: DemoConfig, pool: List[Path]
                     ) -> Dict[str, Tuple[Image.Image, Image.Image]]:
    refs = {}
    for name in cfg.stuff_classes:
        lid = IDX2LABELID[NAME2IDX[name]]
        best, best_px = None, 0
        for p in pool:
            m = gt_labelids(cfg, p) == lid
            if m.sum() > best_px:
                best, best_px = (p, m), int(m.sum())
        if best and best_px > 2000:
            refs[name] = (Image.open(best[0]).convert("RGB"),
                          Image.fromarray((best[1] * 255).astype(np.uint8)))
    logger.info("stuff refs built for %d/%d classes", len(refs), len(cfg.stuff_classes))
    return refs


# --------------------------------- stages -------------------------------------
def run_semantic(insid3, refs, pil_img: Image.Image, cfg: DemoConfig) -> np.ndarray:
    W, H = pil_img.size
    sem = np.full((H, W), VOID, dtype=np.int32)
    for name in cfg.stuff_classes:           # painted in order; later wins
        if name not in refs:
            continue
        insid3.set_reference(*refs[name])
        insid3.set_target(pil_img)
        pred = insid3.segment().cpu().numpy().astype(bool)
        if pred.shape != (H, W):
            pred = np.array(Image.fromarray(pred).resize((W, H), Image.NEAREST))
        sem[pred] = NAME2IDX[name]
    return sem


def run_semantic_sam3(sam3, pil_img: Image.Image, cfg: DemoConfig) -> np.ndarray:
    """Dense stuff map from SAM 3 text concepts (crisp boundaries). Big background
    classes painted first, thin classes (pole/sign) on top."""
    W, H = pil_img.size
    dets = sam3.predict(pil_img, cfg.stuff_order_sam3, thr=cfg.stuff_score_thr)
    by_name: Dict[str, List[np.ndarray]] = {}
    for d in dets:
        by_name.setdefault(d["prompt"], []).append(d["mask"])
    sem = np.full((H, W), VOID, dtype=np.int32)
    for name in cfg.stuff_order_sam3:           # paint order = list order
        masks = by_name.get(name)
        if masks:
            sem[np.any(masks, axis=0)] = NAME2IDX[name]
    return sem


def run_instances(dets: List[dict]) -> List[dict]:
    insts = []
    for d in dets:
        insts.append({"mask": d["mask"], "class_idx": NAME2IDX[d["prompt"]],
                      "score": d["score"]})
    return insts


def merge_panoptic(sem: np.ndarray, insts: List[dict], cfg: DemoConfig
                   ) -> Tuple[np.ndarray, List[dict]]:
    div = cfg.label_divisor
    pan = np.where(sem == VOID, VOID * div, sem * div).astype(np.int32)
    per_class = {}
    for inst in sorted(insts, key=lambda x: x["score"]):  # best score painted last
        c = inst["class_idx"]
        per_class[c] = per_class.get(c, 0) + 1
        pan[inst["mask"]] = c * div + per_class[c]
    segments = []
    for pid in np.unique(pan):
        cid = int(pid) // div
        if cid == VOID:
            continue
        segments.append({"id": int(pid), "class_idx": cid,
                         "isthing": cid in THING_IDX,
                         "area": int((pan == pid).sum())})
    return pan, segments


# ------------------------------- convenience ----------------------------------
def process_image(insid3, sam3, refs, pil_img: Image.Image, cfg: DemoConfig,
                  m2f=None, stuff_source: str = "mask2former") -> dict:
    """Run all stages. `stuff_source` (mask2former|sam3|insid3) picks the stuff map
    the PANOPTIC is built from; the others are still computed for comparison if
    their model is provided. Variant 2 default = mask2former stuff + SAM3 things."""
    sem_i = run_semantic(insid3, refs, pil_img, cfg) if insid3 is not None else None
    sem_s = run_semantic_sam3(sam3, pil_img, cfg)
    sem_m = m2f.predict(pil_img) if m2f is not None else None
    insts = run_instances(sam3.predict(pil_img, cfg.thing_prompts))
    chosen = {"mask2former": sem_m, "sam3": sem_s, "insid3": sem_i}[stuff_source]
    if chosen is None:
        chosen = sem_s
    pan, segs = merge_panoptic(chosen, insts, cfg)
    return {"sem_insid3": sem_i, "sem_sam3": sem_s, "sem_m2f": sem_m,
            "insts": insts, "pan": pan, "segs": segs, "stuff_source": stuff_source}


def semantic_miou(sem: np.ndarray, cfg: DemoConfig, img_path: Path) -> float:
    """Stuff mIoU vs gtFine over the configured stuff classes (sanity metric)."""
    gt = gt_labelids(cfg, img_path)
    per = []
    for name in cfg.stuff_classes:
        idx = NAME2IDX[name]
        p, g = (sem == idx), (gt == IDX2LABELID[idx])
        u = (p | g).sum()
        if u:
            per.append((p & g).sum() / u)
    return float(np.mean(per)) if per else 0.0
