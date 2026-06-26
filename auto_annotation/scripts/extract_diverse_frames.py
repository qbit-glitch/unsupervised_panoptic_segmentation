#!/usr/bin/env python3
"""Extract diverse frames from a folder of videos.

SINGLE RULE: keep a frame iff it is at least `--min-diff` (default 0.20 = 20%) DIFFERENT
from the previously kept frame. No cap on frame count.

Difference metric (`--metric`, value in [0,1]):
  phash  (default) : Hamming(pHash_cand, pHash_lastkept) / 64  -> fraction of perceptual-
                     hash bits that differ. Robust to lighting/compression/small motion.
  pixels           : fraction of (downscaled grayscale) pixels whose intensity changed by
                     more than a small tolerance -> literal "% of the image changed".
  embed            : 1 - cosine_similarity(CLIP_cand, CLIP_lastkept) -> semantic-content
                     difference (slower; ignores pixel motion, best for true variety).

Decode via PyAV seek-sampling every `--interval` s (efficient on 4K webm). Saves frames
resized to `--save-long-side` (2048) as JPG + manifest.csv (with the measured diff).

Optional quality gates (OFF by default to honor "only condition = 20% diff"):
  --min-entropy >0 drops blank/sky frames; --min-blur >0 drops motion-blurred frames.

Run: PYTHONPATH=<repo> .venv_cups_cpu/bin/python auto_annotation/scripts/extract_diverse_frames.py \
        --videos_dir "/Volumes/.../videos" --out_dir <out> --min-diff 0.20 --metric phash
"""

import argparse
import csv
import hashlib
import logging
import re
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("extract")
VID_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
PIXEL_TOL = 25          # intensity-change tolerance for metric=pixels


# ------------------------------- descriptors ----------------------------------
def phash_bits(gray: np.ndarray, size: int = 32, lo: int = 8) -> np.ndarray:
    g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:lo, :lo]
    return (d > np.median(d[1:, 1:])).flatten()          # 64-bit


def entropy(gray: np.ndarray) -> float:
    h = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = h / (h.sum() + 1e-9); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def blur_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name)[:40].strip("_")


# ------------------------------ difference ([0,1]) ----------------------------
class Differ:
    """Computes a 0..1 difference of a candidate vs the last KEPT frame."""

    def __init__(self, metric: str):
        self.metric = metric
        self.last = None                      # descriptor of last kept frame
        self._clip = None

    def _clip_feat(self, small_rgb):
        import torch
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor
            self._m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
            self._p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip = True
        with torch.no_grad():
            inp = self._p(images=Image.fromarray(small_rgb), return_tensors="pt")
            f = self._m.get_image_features(**inp)[0]
            return (f / f.norm()).cpu().numpy()

    def descriptor(self, small_rgb, gray):
        if self.metric == "phash":
            return phash_bits(gray)
        if self.metric == "pixels":
            return gray.astype(np.int16)
        return self._clip_feat(small_rgb)     # embed

    def diff(self, desc) -> float:
        """Return difference in [0,1] vs last kept; 1.0 if no last (always keep first)."""
        if self.last is None:
            return 1.0
        if self.metric == "phash":
            return float(np.count_nonzero(desc != self.last)) / desc.size
        if self.metric == "pixels":
            return float(np.mean(np.abs(desc - self.last) > PIXEL_TOL))
        return float(1.0 - np.dot(desc, self.last))        # embed cosine distance

    def commit(self, desc):
        self.last = desc


# ------------------------------- decoding -------------------------------------
def iter_frames(path: Path, interval: float):
    """Yield (time_s, rgb) ~every `interval` s.

    interval >= 0.5 s : SEEK sampling (fast; decodes only at sampled timestamps).
    interval <  0.5 s : SEQUENTIAL decode (correct for fine spacing; seek would land on
                        the same keyframe repeatedly and under-sample). Slower (decodes
                        ~every frame) but the only way to honour sub-second intervals.
    """
    container = av.open(str(path))
    vs = container.streams.video[0]; vs.thread_type = "AUTO"
    tb = float(vs.time_base)
    dur = float(vs.duration * tb) if vs.duration else (
        float(container.duration) / 1e6 if container.duration else None)
    if interval >= 0.5 and dur and dur > 0:                  # SEEK path
        t = 0.0
        while t < dur:
            try:
                container.seek(int(t / tb), stream=vs, any_frame=False, backward=True)
                yield t, next(container.decode(vs)).to_ndarray(format="rgb24")
            except Exception:
                pass
            t += interval
        container.close(); return
    last = -1e9                                              # SEQUENTIAL path
    for fr in container.decode(vs):
        t = float(fr.pts * tb) if fr.pts is not None else last + 0.04
        if t - last >= interval - 1e-6:
            last = t; yield t, fr.to_ndarray(format="rgb24")
    container.close()


def process_video(path: Path, out_dir: Path, args, writer, vslug: str) -> int:
    differ = Differ(args.metric)
    kept = 0
    try:                                            # iter_frames is lazy: errors (no video
        for t, rgb in iter_frames(path, args.interval):   # stream, e.g. audio-only .f251) fire
            small = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)  # here, not above
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            if args.min_entropy > 0 and entropy(gray) < args.min_entropy:
                continue
            if args.min_blur > 0 and blur_var(gray) < args.min_blur:
                continue
            desc = differ.descriptor(small, gray)
            d = differ.diff(desc)
            if d < args.min_diff:                       # < 20% different -> skip
                continue
            differ.commit(desc)
            H, W = rgb.shape[:2]; s = args.save_long_side / max(H, W)
            out = rgb if s >= 1.0 else cv2.resize(rgb, (int(W * s), int(H * s)),
                                                  interpolation=cv2.INTER_AREA)
            name = f"{vslug}_{kept:06d}.jpg"
            Image.fromarray(out).save(out_dir / name, quality=95)
            writer.writerow([path.name, kept, round(t, 2), round(d, 3), name])
            kept += 1
    except Exception as e:                          # skip unreadable/streamless video, keep going
        log.warning("skip %s: %s", path.name[:50], str(e)[:80])
    log.info("%-42s -> kept %d", path.name[:42], kept)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--out_dir", default="auto_annotation/outputs/extracted_frames")
    ap.add_argument("--interval", type=float, default=1.0, help="seek-sample seconds")
    ap.add_argument("--min-diff", type=float, default=0.20, help="keep if >= this fraction (0-1) different")
    ap.add_argument("--metric", choices=["phash", "pixels", "embed"], default="phash")
    ap.add_argument("--min-entropy", type=float, default=0.0, help=">0 to drop blank frames")
    ap.add_argument("--min-blur", type=float, default=0.0, help=">0 to drop blurry frames")
    ap.add_argument("--save-long-side", type=int, default=2048)
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--recursive", action="store_true", help="find videos in nested subdirs")
    ap.add_argument("--shard", default="0/1", help="k/n: this worker takes every n-th video at offset k")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.videos_dir)
    finder = root.rglob("*") if args.recursive else root.iterdir()
    vids = sorted(p for p in finder if p.suffix.lower() in VID_EXT)
    if args.limit_videos:
        vids = vids[:args.limit_videos]
    k, n = (int(x) for x in args.shard.split("/"))
    vids = vids[k::n]                                          # this worker's shard
    # unique per-video key (path-based, hashed) so nested/duplicate names never collide
    def vkey(p: Path) -> str:
        rel = p.relative_to(root).with_suffix("")
        return slug(str(rel))[:44] + "_" + hashlib.md5(str(rel).encode()).hexdigest()[:6]
    log.info("shard %d/%d: %d videos -> %s | rule: keep if >= %.0f%% different (metric=%s, interval=%.2fs)",
             k, n, len(vids), out, 100 * args.min_diff, args.metric, args.interval)
    man = open(out / (f"manifest_{k}.csv" if n > 1 else "manifest.csv"), "w", newline="")
    w = csv.writer(man); w.writerow(["video", "kept_idx", "src_time_s", "diff_vs_prev", "out_name"])
    total = sum(process_video(v, out, args, w, vkey(v)) for v in vids)
    man.close()
    log.info("TOTAL kept (shard %d/%d): %d -> %s", k, n, total, out)


if __name__ == "__main__":
    main()
