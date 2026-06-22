"""Fuse k=27 semantic pseudo-labels + pre-gated instance masks → CUPS panoptic pseudo-labels.

Two-pass algorithm:
  Pass 1 (all frames): accumulate pixel + instance-overlap distributions → thing-set via psi
  Pass 2 (per frame): apply thing-class guard, relabel, write CUPS-format PNG pair

Output per frame:
  {out_dir}/{stem}_leftImg8bit_semantic.png   uint8 0..26
  {out_dir}/{stem}_leftImg8bit_instance.png   uint16 0=bg 1..N
  {out_dir}/pseudo_classes_split_1.pt         torch dict with two exact distribution keys

Semantic dir formats supported:
  flat:  {sem_dir}/{stem}_leftImg8bit_semantic.png  (pure_cause_tr_27_full layout)
  flat:  {sem_dir}/{stem}_leftImg8bit.png           (pure_cause_tr_27_val layout)
  nested:{sem_dir}/{city}/{stem}.png                (pseudo_semantic_raw_dinov3_k27 layout)

Instance dir formats supported:
  nested:{inst_dir}/{city}/{stem}.png               (train instances layout)
  flat:  {inst_dir}/{stem}_leftImg8bit_instance.png (val cups_pseudo_labels layout)

All output at working resolution H×W = 512×1024. Input resized with NEAREST if needed.

Example (smoke):
  .venv/bin/python scripts/fuse_panoptic_pseudolabels.py \\
    --instance_dir /Volumes/code_files_2/mbps_instances_seq/instances \\
    --semantic_dir /Volumes/code_files_2/cityscapes_sequences/cups_official_root/cups_notebook_27class_caches/pure_cause_tr_27_full \\
    --out_dir /tmp/fuse_smoke --limit 20
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

N_CLS = 27
H, W = 512, 1024   # working resolution


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_img(path: str, method=Image.NEAREST) -> np.ndarray:
    img = np.array(Image.open(path))
    if img.shape[:2] != (H, W):
        img = np.array(Image.fromarray(img.astype(np.int32)).resize((W, H), method))
    return img


def load_sem(path: str) -> np.ndarray:
    return _load_img(path, Image.NEAREST).astype(np.uint8)


def load_inst(path: str) -> np.ndarray:
    return _load_img(path, Image.NEAREST).astype(np.uint16)


def save_sem(path: str, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_inst(path: str, arr: np.ndarray) -> None:
    # I;16 mode avoids Pillow 13 deprecation of I (32-bit) PNG
    img = Image.frombuffer('I;16', (arr.shape[1], arr.shape[0]),
                           arr.astype(np.uint16).tobytes(), 'raw', 'I;16', 0, 1)
    img.save(path)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _stem_from_semfile(fname: str) -> str:
    for suf in ("_leftImg8bit_semantic.png", "_leftImg8bit.png", ".png"):
        if fname.endswith(suf):
            return fname[: -len(suf)]
    return fname


def _city_from_stem(stem: str) -> str:
    return stem.split("_")[0]


def _find_sem_files(sem_dir: str):
    """Yield (stem, path) for all semantic PNGs in sem_dir (flat or nested)."""
    for p in sorted(glob.glob(f"{sem_dir}/**/*.png", recursive=True) +
                    glob.glob(f"{sem_dir}/*.png")):
        stem = _stem_from_semfile(os.path.basename(p))
        if stem:
            yield stem, p


def _find_inst_path(inst_dir: str, city: str, stem: str):
    """Return instance PNG path if it exists (tries city-nested then flat-suffixed)."""
    for cand in (
        f"{inst_dir}/{city}/{stem}.png",
        f"{inst_dir}/{stem}_leftImg8bit_instance.png",
    ):
        if os.path.exists(cand):
            return cand
    return None


def find_pairs(sem_dir: str, inst_dir: str):
    """Return list of (stem, sem_path, inst_path_or_None) sorted by stem."""
    pairs = []
    seen = set()
    for stem, sp in _find_sem_files(sem_dir):
        if stem in seen:
            continue
        seen.add(stem)
        city = _city_from_stem(stem)
        ip = _find_inst_path(inst_dir, city, stem)
        pairs.append((stem, sp, ip))   # ip may be None → zero instances
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Pass 1: distributions
# ---------------------------------------------------------------------------

def accumulate(pairs, verbose=True):
    dist_all = np.zeros(N_CLS, np.int64)
    dist_inst = np.zeros(N_CLS, np.int64)
    for i, (stem, sp, ip) in enumerate(pairs):
        sem = load_sem(sp)
        dist_all += np.bincount(sem.ravel(), minlength=N_CLS)
        if ip is not None:
            inst = load_inst(ip)
            if inst.any():
                dist_inst += np.bincount(sem[inst > 0].ravel(), minlength=N_CLS)
        if verbose and (i + 1) % 500 == 0:
            print(f"  pass1 {i+1}/{len(pairs)}")
    return dist_all, dist_inst


def compute_thing_set(dist_all: np.ndarray, dist_inst: np.ndarray, psi: float):
    frac = dist_inst / (dist_all + 1e-6)
    return {int(c) for c in range(N_CLS) if frac[c] > psi}


# ---------------------------------------------------------------------------
# Pass 2: per-frame fusion
# ---------------------------------------------------------------------------

def fuse_frame(sem: np.ndarray, inst_in: np.ndarray,
               thing_set: set, guard: str, thing_frac: float, uncovered: str):
    """Return (sem_out, inst_out, n_kept, n_dropped)."""
    sem_out = sem.copy()
    inst_out = np.zeros((H, W), np.uint16)
    n_kept = n_dropped = 0

    ids = [int(i) for i in np.unique(inst_in) if i != 0]
    if not ids:
        if uncovered == "void":
            sem_out[np.isin(sem_out, list(thing_set))] = 255
        return sem_out, inst_out, 0, 0

    areas = {i: int((inst_in == i).sum()) for i in ids}
    nid = 0

    for i in sorted(ids, key=lambda x: -areas[x]):
        mask = inst_in == i
        sem_px = sem[mask].astype(np.int64)

        if guard == "off":
            voted = int(np.bincount(sem_px, minlength=N_CLS).argmax())
            keep = True
        elif guard == "hard":
            voted = int(np.bincount(sem_px, minlength=N_CLS).argmax())
            keep = voted in thing_set
        else:  # soft
            thing_mask = np.array([c in thing_set for c in sem_px])
            tf = thing_mask.sum() / max(len(sem_px), 1)
            if tf < thing_frac:
                keep = False
                voted = -1
            else:
                keep = True
                counts = np.bincount(sem_px[thing_mask], minlength=N_CLS).astype(float)
                for c in range(N_CLS):
                    if c not in thing_set:
                        counts[c] = 0.0
                voted = int(counts.argmax())

        if keep:
            nid += 1
            free = mask & (inst_out == 0)
            inst_out[free] = nid
            sem_out[free] = voted
            n_kept += 1
        else:
            n_dropped += 1

    if uncovered == "void":
        uncov = (inst_out == 0) & np.isin(sem_out, list(thing_set))
        sem_out[uncov] = 255

    return sem_out, inst_out, n_kept, n_dropped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_dir", required=True)
    ap.add_argument("--semantic_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thing_guard", choices=["soft", "hard", "off"], default="soft")
    ap.add_argument("--thing_frac", type=float, default=0.4)
    ap.add_argument("--uncovered", choices=["keep", "void"], default="keep")
    ap.add_argument("--psi", type=float, default=0.08)
    ap.add_argument("--limit", type=int, default=0, help="smoke: process first N stems only")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("Discovering pairs...")
    pairs = find_pairs(a.semantic_dir, a.instance_dir)
    n_with_inst = sum(1 for _, _, ip in pairs if ip is not None)
    print(f"  {len(pairs)} semantic stems, {n_with_inst} with instances, "
          f"{len(pairs)-n_with_inst} no-instance (inst=0)")
    if a.limit:
        pairs = pairs[:a.limit]
        print(f"  Smoke: capped at {a.limit}")

    print(f"\nPass 1: accumulating distributions ({len(pairs)} frames)...")
    dist_all, dist_inst = accumulate(pairs)
    thing_set = compute_thing_set(dist_all, dist_inst, a.psi)
    frac = dist_inst / (dist_all + 1e-6)
    print(f"  psi={a.psi}: {len(thing_set)} thing-clusters: {sorted(thing_set)}")
    print(f"  {N_CLS-len(thing_set)} stuff-clusters: {sorted(set(range(N_CLS))-thing_set)}")
    for c in sorted(thing_set):
        print(f"    cluster {c:2d}: instance-frac={frac[c]:.3f}  "
              f"all_px={dist_all[c]:,}  inst_px={dist_inst[c]:,}")

    print(f"\nPass 2: fusing {len(pairs)} frames...")
    total_kept = total_dropped = 0
    for n, (stem, sp, ip) in enumerate(pairs):
        sem = load_sem(sp)
        inst_in = load_inst(ip) if ip is not None else np.zeros((H, W), np.uint16)
        sem_out, inst_out, nk, nd = fuse_frame(
            sem, inst_in, thing_set, a.thing_guard, a.thing_frac, a.uncovered)
        save_sem(f"{a.out_dir}/{stem}_leftImg8bit_semantic.png", sem_out)
        save_inst(f"{a.out_dir}/{stem}_leftImg8bit_instance.png", inst_out)
        total_kept += nk
        total_dropped += nd
        if n < 5 or (n + 1) % 200 == 0:
            print(f"  [{n+1}/{len(pairs)}] {stem}: kept={nk} dropped={nd}")

    total_inst = total_kept + total_dropped
    print(f"\n{'='*60}")
    print(f"Instance guard summary  (guard={a.thing_guard}, thing_frac={a.thing_frac})")
    print(f"  Total instances (before guard): {total_inst}")
    pct_k = 100 * total_kept / max(total_inst, 1)
    pct_d = 100 * total_dropped / max(total_inst, 1)
    print(f"  Kept:    {total_kept:6d}  ({pct_k:.1f}%)")
    print(f"  Dropped: {total_dropped:6d}  ({pct_d:.1f}%)")
    print(f"{'='*60}")

    torch.save(
        {"distribution all pixels": torch.from_numpy(dist_all),
         "distribution inside object proposals": torch.from_numpy(dist_inst)},
        f"{a.out_dir}/pseudo_classes_split_1.pt",
    )

    # Self-check first output
    s0 = stem0 = pairs[0][0]
    sc = np.array(Image.open(f"{a.out_dir}/{s0}_leftImg8bit_semantic.png"))
    ic = np.array(Image.open(f"{a.out_dir}/{s0}_leftImg8bit_instance.png"))
    valid_sem = sc[sc != 255]
    assert sc.dtype == np.uint8, f"semantic dtype {sc.dtype}"
    assert ic.dtype == np.uint16, f"instance dtype {ic.dtype}"  # PIL round-trip: I → uint16 on load
    assert int(valid_sem.max()) <= 26 if len(valid_sem) else True, f"sem oor: {valid_sem.max()}"
    d = torch.load(f"{a.out_dir}/pseudo_classes_split_1.pt", weights_only=True)
    assert "distribution all pixels" in d and "distribution inside object proposals" in d
    print(f"\nSelf-check OK: sem=uint8 max={sc.max()}, inst=uint16 max={ic.max()}, .pt keys valid")
    print(f"Done: {len(pairs)} frames → {a.out_dir}")


if __name__ == "__main__":
    main()
