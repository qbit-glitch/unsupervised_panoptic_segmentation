#!/usr/bin/env python3
"""Self-contained Jaipur panoptic auto-labeller for fics-lab (Blackwell, fp32 SAM3).

Reads frames from a folder; writes per frame to <out>/:
  {stem}_pan.png   uint16 panoptic id = class_idx*1000 + instance  (void -> 65535)
  {stem}_seg.json  segments_info
  {stem}_color.png RGB visualization: stuff = Cityscapes class color, each THING
                   INSTANCE a distinct color (so adjacent instances are separable)
Stack = stuff ensemble {M2F-Mapillary, M2F-Cityscapes, EoMT} + CRF + SAM3 things (tiled).
SAM3 repo is sed-patched to fp32 -> no bf16/fp32 dtype issues. Resumable (skip-done).
"""
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("autolabel")

# Cityscapes-19 (trainId order == idx) + Indian things appended, with display colors
_CS = [("road",0,(128,64,128)),("sidewalk",0,(244,35,232)),("building",0,(70,70,70)),
       ("wall",0,(102,102,156)),("fence",0,(190,153,153)),("pole",0,(153,153,153)),
       ("traffic light",0,(250,170,30)),("traffic sign",0,(220,220,0)),
       ("vegetation",0,(107,142,35)),("terrain",0,(152,251,152)),("sky",0,(70,130,180)),
       ("person",1,(220,20,60)),("rider",1,(255,0,0)),("car",1,(0,0,142)),
       ("truck",1,(0,0,70)),("bus",1,(0,60,100)),("train",1,(0,80,100)),
       ("motorcycle",1,(0,0,230)),("bicycle",1,(119,11,32)),("auto rickshaw",1,(255,140,0)),
       ("cow",1,(160,82,45)),("dog",1,(255,215,0)),("cart",1,(75,0,130))]
NAME2IDX = {n: i for i, (n, _, _) in enumerate(_CS)}
IDX2COLOR = {i: c for i, (_, _, c) in enumerate(_CS)}
THING_IDX = {i for i, (_, t, _) in enumerate(_CS) if t}
VOID, DIV = 255, 1000
STUFF_ORDER = ["sky","building","wall","fence","vegetation","terrain","pole",
               "traffic light","traffic sign","sidewalk","road"]
STUFF_IDX = {NAME2IDX[n] for n in STUFF_ORDER}
THING_PROMPTS = ["car","truck","bus","motorcycle","bicycle","person","rider",
                 "auto rickshaw","cow","dog","cart"]
MAPILLARY_TO_CS = {13:"road",7:"road",14:"road",8:"road",24:"road",23:"road",10:"road",
    43:"road",41:"road",36:"road",15:"sidewalk",2:"sidewalk",9:"sidewalk",11:"sidewalk",
    17:"building",16:"building",18:"building",35:"building",32:"building",6:"wall",
    3:"fence",4:"fence",5:"fence",45:"pole",47:"pole",46:"pole",44:"pole",
    48:"traffic light",50:"traffic sign",49:"traffic sign",30:"vegetation",29:"terrain",
    26:"terrain",25:"terrain",28:"terrain",31:"terrain",27:"sky"}


class M2FMapillary:
    REPO = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    def __init__(self, device, long_side=1024):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.device, self.long_side = device, long_side
        self.proc = AutoImageProcessor.from_pretrained(self.REPO)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.REPO).eval().to(device)
        self.lut = np.full(65, VOID, np.int32)
        for mid, name in MAPILLARY_TO_CS.items():
            self.lut[mid] = NAME2IDX[name]
        log.info("M2F-Mapillary ready")
    @torch.no_grad()
    def predict(self, pil):
        W, H = pil.size; s = pil
        if max(W, H) > self.long_side:
            r = self.long_side / max(W, H); s = pil.resize((int(W*r), int(H*r)), Image.BILINEAR)
        inp = self.proc(images=s, return_tensors="pt").to(self.device)
        mp = self.proc.post_process_semantic_segmentation(
            self.model(**inp), target_sizes=[(H, W)])[0].cpu().numpy().astype(np.int32)
        return self.lut[np.clip(mp, 0, 64)]


class _CSSem:
    def __init__(self, repo, device, auto_cls):
        from transformers import AutoImageProcessor
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = auto_cls.from_pretrained(repo).eval().to(device)
        log.info("%s ready", repo.split("/")[-1])
    @torch.no_grad()
    def predict(self, pil):
        W, H = pil.size
        inp = self.proc(images=pil, return_tensors="pt").to(self.device)
        return self.proc.post_process_semantic_segmentation(
            self.model(**inp), target_sizes=[(H, W)])[0].cpu().numpy().astype(np.int32)


def ensemble_stuff(maps):
    stack = np.stack(maps, 0); res = maps[0].reshape(-1).copy()
    flat = stack.reshape(stack.shape[0], -1)
    for lab in np.unique(stack):
        if lab in STUFF_IDX:
            res[(flat == lab).sum(0) >= 2] = lab
    return res.reshape(maps[0].shape)


def crf_refine(rgb, sem, n_iter=5):
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels
    except Exception:
        return sem
    h, w = sem.shape; work = sem.copy(); work[work == VOID] = 19
    d = dcrf.DenseCRF2D(w, h, 20)
    d.setUnaryEnergy(unary_from_labels(work.astype(np.int32), 20, gt_prob=0.7, zero_unsure=False))
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=13, rgbim=np.ascontiguousarray(rgb), compat=8)
    out = np.argmax(np.array(d.inference(n_iter)), 0).reshape(h, w).astype(np.int32)
    out[out == 19] = VOID
    return out


class Sam3Runner:
    def __init__(self, device, repo_dir, resolution=1008, thr=0.5):
        sys.path.insert(0, repo_dir)
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.model = build_sam3_image_model(device=device)   # fp32 (repo sed-patched)
        self.proc = Sam3Processor(self.model, resolution=resolution, device=device)
        self.thr = thr
        log.info("SAM3 ready (fp32)")
    @torch.no_grad()
    def predict(self, pil, prompts, thr=None):
        thr = self.thr if thr is None else thr
        state = self.proc.set_image(pil); dets = []
        for p in prompts:
            out = self.proc.set_text_prompt(state=state, prompt=p)
            masks, scores = out.get("masks"), out.get("scores")
            if masks is None:
                continue
            m = masks.detach().float().cpu().numpy()
            if m.ndim == 4:
                m = m[:, 0]
            for i in range(m.shape[0]):
                sc = float(scores[i]) if scores is not None else 1.0
                if sc >= thr:
                    dets.append({"mask": m[i] > 0.5, "prompt": p, "score": sc})
        return dets


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def sam3_tiled(sam3, pil, prompts, thr, tiles=2, overlap=0.2):
    W, H = pil.size
    dets = [{"mask": d["mask"], "class_idx": NAME2IDX[d["prompt"]], "score": d["score"]}
            for d in sam3.predict(pil, prompts)]
    tw, th = int(W/tiles*(1+overlap)), int(H/tiles*(1+overlap))
    for iy in range(tiles):
        for ix in range(tiles):
            x0, y0 = int(ix*W/tiles), int(iy*H/tiles)
            x1, y1 = min(W, x0+tw), min(H, y0+th)
            for d in sam3.predict(pil.crop((x0, y0, x1, y1)), prompts):
                full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = d["mask"]
                dets.append({"mask": full, "class_idx": NAME2IDX[d["prompt"]], "score": d["score"]*0.95})
    dets.sort(key=lambda d: -d["score"]); kept = []
    for d in dets:
        if d["mask"].sum() >= 200 and all(
                _iou(d["mask"], k["mask"]) < 0.5 for k in kept if k["class_idx"] == d["class_idx"]):
            kept.append(d)
    return kept


def merge_panoptic(sem, insts):
    pan = np.where(sem == VOID, VOID*DIV, sem*DIV).astype(np.int32)
    per = {}
    for inst in sorted(insts, key=lambda x: x["score"]):
        c = inst["class_idx"]; per[c] = per.get(c, 0) + 1
        pan[inst["mask"]] = c*DIV + per[c]
    segs = []
    for pid in np.unique(pan):
        cid = int(pid)//DIV
        if cid != VOID:
            segs.append({"id": int(pid), "class_idx": cid,
                         "isthing": cid in THING_IDX, "area": int((pan == pid).sum())})
    return pan, segs


def colorize(pan):
    """RGB viz: stuff = fixed class color, each THING INSTANCE a distinct color."""
    h, w = pan.shape
    out = np.zeros((h, w, 3), np.uint8)
    for pid in np.unique(pan):
        cid = int(pid) // DIV
        if cid == VOID:
            continue                                  # void -> black
        m = pan == pid
        if cid in THING_IDX:                          # distinct, reproducible per-instance color
            out[m] = np.random.default_rng(int(pid)).integers(40, 256, 3)
        else:
            out[m] = IDX2COLOR.get(cid, (127, 127, 127))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sam3_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tiles", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    k, n = (int(x) for x in a.shard.split("/"))

    fs = sorted(Path(a.frames).glob("*.jpg")) + sorted(Path(a.frames).glob("*.png"))
    items = [(str(f), f.stem) for f in fs]
    items = [it for j, it in enumerate(items) if j % n == k]
    todo = [it for it in items if not (out / f"{it[1]}_pan.png").exists()]
    if a.limit:
        todo = todo[:a.limit]
    log.info("shard %d/%d: %d frames, %d to label", k, n, len(items), len(todo))
    if not todo:
        return

    from transformers import Mask2FormerForUniversalSegmentation, AutoModelForUniversalSegmentation
    m2f_map = M2FMapillary(a.device)
    m2f_cs = _CSSem("facebook/mask2former-swin-large-cityscapes-semantic", a.device,
                    Mask2FormerForUniversalSegmentation)
    eomt = _CSSem("tue-mps/cityscapes_semantic_eomt_large_1024", a.device,
                  AutoModelForUniversalSegmentation)
    sam3 = Sam3Runner(a.device, a.sam3_dir)                      # built LAST

    for i, (src, stem) in enumerate(todo):
        t0 = time.time()
        pil = Image.open(src).convert("RGB"); rgb = np.array(pil)
        sem = crf_refine(rgb, ensemble_stuff([m2f_map.predict(pil), m2f_cs.predict(pil), eomt.predict(pil)]))
        insts = sam3_tiled(sam3, pil, THING_PROMPTS, sam3.thr, tiles=a.tiles) if a.tiles >= 2 else \
            [{"mask": d["mask"], "class_idx": NAME2IDX[d["prompt"]], "score": d["score"]}
             for d in sam3.predict(pil, THING_PROMPTS)]
        pan, segs = merge_panoptic(sem, insts)
        Image.fromarray(np.where(pan // DIV == VOID, 65535, pan).astype(np.uint16), mode="I;16").save(out / f"{stem}_pan.png")
        Image.fromarray(colorize(pan)).save(out / f"{stem}_color.png")
        (out / f"{stem}_seg.json").write_text(json.dumps({"segments": segs}))
        n_things = sum(s["isthing"] for s in segs)
        log.info("[%d/%d] %s | %d segs (%d inst) | %.1fs", i+1, len(todo), stem[:26],
                 len(segs), n_things, time.time()-t0)
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
