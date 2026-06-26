#!/usr/bin/env python3
"""Fine-tune EoMT (DINOv3 ViT-L panoptic) on the Jaipur auto-labelled pseudo-labels.

Start from tue-mps/eomt-dinov3-coco-panoptic-large-640 (133-class COCO panoptic),
reinit the class head for our 23 classes, FREEZE the DINOv3 backbone (embeddings +
ViT layers) and train only queries + mask/class heads + upscale block.

Targets come straight from our labels: the EomtImageProcessor turns each frame's
panoptic _pan.png (segment-id per pixel) + an id->class map into (mask_labels,
class_labels). Void (65535) -> ignore_index.

No GT exists for Jaipur, so "eval" = periodic colored prediction dumps on held-out
frames + train loss. Resumable (loads latest step_*.pt). ponytail: backbone frozen
keeps it light + robust to noisy pseudo-labels; unfreeze last ViT layers if it underfits.
"""
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

REPO = "tue-mps/eomt-dinov3-coco-panoptic-large-640"
NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
         "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
         "truck", "bus", "train", "motorcycle", "bicycle", "auto rickshaw", "cow",
         "dog", "cart"]
NUM = len(NAMES)            # 23
VOID = 65535
IGNORE = 255               # EomtImageProcessor ignore_index
DIV = 1000
THING = set(range(11, 23))
# distinct color per class (stuff) + per instance (things) for viz
import colorsys
_STUFF_COLOR = {i: tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / NUM, 0.5, 0.9)) for i in range(NUM)}


class JaipurDS(Dataset):
    def __init__(self, frames_dir, labels_dir, proc, stems):
        self.fr, self.lb, self.proc, self.stems = frames_dir, labels_dir, proc, stems

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        s = self.stems[i]
        img = Image.open(os.path.join(self.fr, s + ".jpg")).convert("RGB")
        pan = np.array(Image.open(os.path.join(self.lb, s + "_pan.png"))).astype(np.int64)
        # EomtImageProcessor convention: 0 = background/ignore; it shifts seg-1 and returns
        # class-1, so seg ids must be 1-based and instance_id_to_semantic_id 1-based class.
        ids = [int(v) for v in np.unique(pan) if v != VOID]
        seg = np.zeros_like(pan)                                   # void -> 0 (ignored)
        id2cls = {}
        for k, orig in enumerate(ids, start=1):
            seg[pan == orig] = k
            id2cls[k] = (orig // DIV) + 1                          # 1-based class (proc returns -1)
        enc = self.proc.preprocess(images=[img], segmentation_maps=[seg],
                                   instance_id_to_semantic_id=id2cls, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"][0],
                "mask_labels": enc["mask_labels"][0],
                "class_labels": enc["class_labels"][0]}


def collate(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "mask_labels": [b["mask_labels"] for b in batch],
            "class_labels": [b["class_labels"] for b in batch]}


def build_model(unfreeze_last=0):
    from transformers import AutoModelForUniversalSegmentation
    id2label = {i: n for i, n in enumerate(NAMES)}
    model = AutoModelForUniversalSegmentation.from_pretrained(
        REPO, num_labels=NUM, id2label=id2label, label2id={n: i for i, n in id2label.items()},
        ignore_mismatched_sizes=True)
    nlayers = 1 + max(int(n.split(".")[1]) for n, _ in model.named_parameters() if n.startswith("layers."))
    keep = set(range(nlayers - unfreeze_last, nlayers))     # last N ViT blocks stay trainable
    frozen = trainable = 0
    for n, p in model.named_parameters():
        is_bb = n.startswith("embeddings") or n.startswith("layers.") or n.startswith("rope_embeddings")
        unfroze = n.startswith("layers.") and int(n.split(".")[1]) in keep
        if is_bb and not unfroze:
            p.requires_grad = False; frozen += p.numel()
        else:
            trainable += p.numel()
    print(f"layers={nlayers}, unfreeze_last={unfreeze_last} | frozen {frozen/1e6:.0f}M | trainable {trainable/1e6:.0f}M", flush=True)
    return model


def colorize(pan, segs_info):
    """pan: (H,W) seg ids from post_process; segs_info: list of {id,label_id}. Per-instance color."""
    h, w = pan.shape
    out = np.zeros((h, w, 3), np.uint8)
    for s in segs_info:
        cid = s["label_id"]; m = pan == s["id"]
        if cid in THING:
            out[m] = np.random.default_rng(s["id"] + 1).integers(40, 256, 3)
        else:
            out[m] = _STUFF_COLOR.get(int(cid), (127, 127, 127))
    return out


@torch.no_grad()
def viz(model, proc, frames_dir, labels_dir, stems, out_dir, step, device):
    model.eval(); os.makedirs(out_dir, exist_ok=True)
    for s in stems:
        img = Image.open(os.path.join(frames_dir, s + ".jpg")).convert("RGB")
        enc = proc(images=[img], return_tensors="pt").to(device)
        out = model(pixel_values=enc["pixel_values"])
        res = proc.post_process_panoptic_segmentation(out, target_sizes=[img.size[::-1]])[0]
        pan = res["segmentation"].cpu().numpy()
        Image.fromarray(colorize(pan, res["segments_info"])).save(
            os.path.join(out_dir, f"{s}_step{step}_pred.png"))
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/mnt/HDD_16TB/datasets/jaipur_dashcam_frames")
    ap.add_argument("--labels", default="/mnt/HDD_16TB/jaipur_work/labels")
    ap.add_argument("--out", default="/mnt/HDD_16TB/jaipur_work/eomt_ft")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1, help="effective batch = bs * grad_accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--viz_every", type=int, default=1000)
    ap.add_argument("--val_n", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap train stems")
    ap.add_argument("--unfreeze_last", type=int, default=0, help="unfreeze last N ViT blocks")
    ap.add_argument("--bb_lr_mult", type=float, default=0.1, help="LR mult for unfrozen backbone")
    ap.add_argument("--init_from", default="", help="load model weights to warm-start (no optimizer)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(REPO)
    proc.ignore_index = 255            # enable 0=background; seg-1 shift + class-1 (see __getitem__)

    stems = sorted(p[:-8].split("/")[-1] for p in glob.glob(os.path.join(a.labels, "*_pan.png")))
    val_stems = stems[:a.val_n]; train_stems = stems[a.val_n:]
    if a.limit:
        train_stems = train_stems[:a.limit]
    print(f"train {len(train_stems)} | val(viz) {len(val_stems)}", flush=True)

    model = build_model(a.unfreeze_last).to(dev)
    if a.init_from:                                          # warm-start weights from a prior phase
        sd = torch.load(a.init_from, map_location=dev)
        model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
        print(f"warm-started weights from {a.init_from}", flush=True)
    # two LR groups: heads at lr, unfrozen ViT blocks at lr*bb_lr_mult
    head_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("layers.")]
    bb_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("layers.")]
    groups = [{"params": head_p, "lr": a.lr}]
    if bb_p:
        groups.append({"params": bb_p, "lr": a.lr * a.bb_lr_mult})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    start = 0
    cks = sorted(glob.glob(os.path.join(a.out, "step_*.pt")), key=lambda p: int(p.split("step_")[1].split(".")[0]))
    if cks:
        sd = torch.load(cks[-1], map_location=dev)
        model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"]); start = sd["step"]
        print(f"resumed from {cks[-1]} @ step {start}", flush=True)

    dl = DataLoader(JaipurDS(a.frames, a.labels, proc, train_stems), batch_size=a.bs,
                    shuffle=True, num_workers=a.workers, collate_fn=collate,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    accum = a.grad_accum                         # effective batch = bs * grad_accum
    print(f"effective batch = {a.bs} x {accum} = {a.bs*accum}", flush=True)
    model.train(); it = iter(dl); t0 = time.time(); run = 0.0
    opt.zero_grad()
    for step in range(start, a.steps):           # step = OPTIMIZER step
        for _ in range(accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            out = model(pixel_values=b["pixel_values"].to(dev),
                        mask_labels=[m.to(dev) for m in b["mask_labels"]],
                        class_labels=[c.to(dev) for c in b["class_labels"]])
            (out.loss / accum).backward()
            run += out.loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 20 == 0:
            dt = (time.time() - t0) / 20; t0 = time.time()
            print(f"step {step}/{a.steps} | loss {run/20 if step else run:.3f} | {dt:.2f}s/it", flush=True)
            run = 0.0
        if step > start and step % a.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
                       os.path.join(a.out, f"step_{step}.pt"))
            if a.viz_every and step % a.viz_every == 0:
                viz(model, proc, a.frames, a.labels, val_stems, os.path.join(a.out, "viz"), step, dev)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": a.steps},
               os.path.join(a.out, "final.pt"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
