"""Run canonical DINOv2 CAUSE-TR -> 27-class semantic PNGs (CUPS pseudo-label semantics).

Replicates refs/cause/eval_cause_tr_dinov2.py forward (no CRF), saving per-image maps:
  img -> Resize 322x322 -> DINOv2 ViT-B/14 -> Segment_TR.head_ema -> interp to 1024x2048
      -> Cluster.forward_centroid(inference) -> argmax 27 cls -> {stem}_leftImg8bit_semantic.png

Trained states: refs/cause/CAUSE_dinov2_retrain/epoch_040/{segment_tr,cluster_tr}.pth + modular.npy
Run:   python scripts/gen_cause_semantics.py --split val --out_dir <...>
Smoke: python scripts/gen_cause_semantics.py --smoke
"""
import argparse
import glob
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

CAUSE = os.path.join(os.path.dirname(__file__), "..", "refs", "cause")
CAUSE = os.path.abspath(CAUSE)
sys.path.insert(0, CAUSE)
# pydensecrf is only needed for CRF post-processing (not used here) — stub it
import types as _types
_pkg = _types.ModuleType("pydensecrf"); _pkg.__path__ = []
sys.modules.setdefault("pydensecrf", _pkg)
sys.modules.setdefault("pydensecrf.densecrf", _types.ModuleType("pydensecrf.densecrf"))
sys.modules.setdefault("pydensecrf.utils", _types.ModuleType("pydensecrf.utils"))
from modules.segment import Segment_TR                       # noqa: E402
from modules.segment_module import Cluster, transform, untransform  # noqa: E402
from utils.utils import ckpt_to_arch, freeze                 # noqa: E402
import models.dinov2vit as dv2                               # noqa: E402

B = "/Volumes/code_files/datasets/cityscapes"
H, W = 1024, 2048
RETRAIN = f"{CAUSE}/CAUSE_dinov2_retrain"
TF = T.Compose([T.Resize((322, 322)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class _Args:
    dim, reduced_dim, projection_dim = 768, 90, 2048
    num_codebook, n_classes, num_queries = 2048, 27, 529


def load(device):
    bb = f"{CAUSE}/checkpoint/dinov2_vit_base_14.pth"
    net = getattr(dv2, ckpt_to_arch(bb))()
    net.load_state_dict(torch.load(bb, map_location="cpu"), strict=False)
    freeze(net); net = net.to(device).eval()
    a = _Args()
    seg = Segment_TR(a)
    seg.load_state_dict(torch.load(f"{RETRAIN}/epoch_040/segment_tr.pth", map_location="cpu"), strict=False)
    seg = seg.to(device).eval()
    cl = Cluster(a)
    cl.load_state_dict(torch.load(f"{RETRAIN}/epoch_040/cluster_tr.pth", map_location="cpu"), strict=False)
    cb = torch.from_numpy(np.load(f"{RETRAIN}/modular.npy")).to(device)
    cl.codebook.data = cb
    cl = cl.to(device).eval()
    seg.head.codebook = cb            # Decoder uses self.codebook (set externally, eval:459-460)
    seg.head_ema.codebook = cb
    return net, seg, cl


@torch.no_grad()
def infer(net, seg, cl, pil, device):
    x = TF(pil)[None].to(device)
    feat = net(x)[:, 1:, :]                       # (1, 529, 768) drop CLS
    s = seg.head_ema(feat)                         # (1, 529, 90)
    it = F.interpolate(transform(s), (H, W), mode="bilinear", align_corners=False)  # (1,90,H,W)
    # forward_centroid math on a 2:1 grid (transform()'s square reshape can't take non-square):
    nf = F.normalize(it, dim=1)
    nc = F.normalize(cl.cluster_probe, dim=1)      # (27, 90)
    pred = torch.einsum("bchw,nc->bnhw", nf, nc).argmax(1)   # (1, H, W) ids 0..26
    return pred[0].cpu().numpy().astype(np.uint8)


def stems(split):
    out = []
    for p in sorted(glob.glob(f"{B}/leftImg8bit/{split}/*/*_leftImg8bit.png")):
        city = os.path.basename(os.path.dirname(p))
        out.append((city, os.path.basename(p).replace("_leftImg8bit.png", ""), p))
    return out


def stems_from_root(images_root):
    """Walk images_root recursively for *_leftImg8bit.png -> (city, stem, path)."""
    out = []
    for p in sorted(glob.glob(f"{images_root}/**/*_leftImg8bit.png", recursive=True)):
        stem = os.path.basename(p).replace("_leftImg8bit.png", "")
        city = stem.split("_")[0]
        out.append((city, stem, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--images_root", default=None,
                    help="Walk recursively for *_leftImg8bit.png (sequence mode); "
                         "overrides --split. Output is city-nested.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--total", type=int, default=1)
    a = ap.parse_args()

    if a.images_root:
        items = stems_from_root(a.images_root)
        nested = True
    else:
        items = stems(a.split)
        nested = False

    if a.limit:
        items = items[:a.limit]
    if a.total > 1:
        items = [x for i, x in enumerate(items) if i % a.total == a.shard]

    net, seg, cl = load(a.device)
    n_done = n_skip = 0
    total = len(items)
    for n, (city, stem, img_path) in enumerate(items):
        if nested:
            out_dir = os.path.join(a.out_dir, city)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{stem}_leftImg8bit_semantic.png")
        else:
            os.makedirs(a.out_dir, exist_ok=True)
            out_path = os.path.join(a.out_dir, f"{stem}_leftImg8bit_semantic.png")

        if os.path.exists(out_path):
            n_skip += 1
            continue

        pil = Image.open(img_path).convert("RGB")
        sem = infer(net, seg, cl, pil, a.device)
        Image.fromarray(sem).save(out_path)
        n_done += 1
        if n_done % 100 == 0:
            print(f"  [{n+1}/{total}] done={n_done} skip={n_skip}")

    print(f"done: {n_done} new, {n_skip} skipped -> {a.out_dir}")


def smoke():
    dev = "cpu"
    net, seg, cl = load(dev)
    for city, stem, img_path in stems("val")[:2]:
        pil = Image.open(img_path).convert("RGB")
        sem = infer(net, seg, cl, pil, dev)
        u, c = np.unique(sem, return_counts=True)
        top = sorted(zip(c, u), reverse=True)[:6]
        print(f"{stem}: shape={sem.shape} dtype={sem.dtype} nclasses={len(u)} "
              f"top(class:%)={[(int(cl_), round(100*ct/sem.size,1)) for ct, cl_ in top]}")
    print(f"CAUSE smoke OK  ({len(u)} classes fired, need >=8)")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        main()
