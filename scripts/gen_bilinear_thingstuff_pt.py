#!/usr/bin/env python3
"""Generate pseudo_classes_split_1.pt for the bilinear label dir, mirroring AnyUp's exact structure.

The CUPS Stage-2 trainer (refs/cups/train_eomt.py -> PseudoLabelDataset) loads this .pt from
ROOT_PSEUDO to derive the thing/stuff split at runtime (THING_STUFF_THRESHOLD=0.08). AnyUp's dir
has it; bilinear's does not. This reproduces it from the bilinear (semantic, instance) pairs so the
A/B arms differ ONLY in label content, not in whether training can start.

ponytail: mirror the reference .pt's exact keys/dtype instead of guessing the schema.
"""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ANY = ROOT / "cups_pseudo_labels_causetr_anyup_train"
BIL = ROOT / "cups_pseudo_labels_causetr_depthedge_train_full"
NUM_CLASSES = 27
THR = 0.08  # THING_STUFF_THRESHOLD default


def load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(p, map_location="cpu", weights_only=False)


def semantic_max(d, n=25):
    """Global max semantic value over a sample — must be < NUM_CLASSES."""
    stems = sorted(d.glob("*_leftImg8bit_semantic.png"))[:n]
    return max(int(np.array(Image.open(s)).max()) for s in stems), len(stems)


def build_split(d):
    all_pix = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    inside = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    inst_files = sorted(d.glob("*_leftImg8bit_instance.png"))
    for k, ip in enumerate(inst_files):
        sp = ip.parent / ip.name.replace("_instance.png", "_semantic.png")
        if not sp.exists():
            continue
        sem = torch.from_numpy(np.array(Image.open(sp)).astype(np.int64)).flatten()
        inst = torch.from_numpy(np.array(Image.open(ip)).astype(np.int64)).flatten()
        all_pix += torch.bincount(sem, minlength=NUM_CLASSES)[:NUM_CLASSES]
        m = inst != 0
        if m.any():
            inside += torch.bincount(sem[m], minlength=NUM_CLASSES)[:NUM_CLASSES]
        if (k + 1) % 500 == 0:
            print(f"  {d.name}: {k+1}/{len(inst_files)}", flush=True)
    return all_pix, inside, len(inst_files)


def things_at(all_pix, inside, thr=THR):
    ratio = inside.double() / all_pix.double().clamp(min=1)
    return sorted(int(c) for c in torch.where(ratio > thr)[0].tolist()), ratio


def main():
    # 1. Inspect reference .pt
    ref_p = ANY / "pseudo_classes_split_1.pt"
    ref = load_pt(ref_p)
    print(f"=== reference {ref_p.name} ===")
    keys = list(ref.keys())
    for kk in keys:
        v = ref[kk]
        print(f"  '{kk}': shape={tuple(v.shape)} dtype={v.dtype} sum={int(v.sum())}")
    assert len(keys) == 2, f"unexpected key count: {keys}"
    ALL_KEY = "distribution all pixels"
    INS_KEY = "distribution inside object proposals"
    assert ALL_KEY in ref and INS_KEY in ref, f"key mismatch: {keys}"

    # 2. Verify 27-class in BOTH dirs
    amax, an = semantic_max(ANY); bmax, bn = semantic_max(BIL)
    print(f"\nsemantic max: anyup={amax} (n={an}) bilinear={bmax} (n={bn}); NUM_CLASSES={NUM_CLASSES}")
    assert amax < NUM_CLASSES and bmax < NUM_CLASSES, "semantic values exceed NUM_CLASSES — wrong class space!"

    # 3. Reference thing/stuff split (from stored anyup .pt)
    a_things, _ = things_at(ref[ALL_KEY][:NUM_CLASSES], ref[INS_KEY][:NUM_CLASSES])
    print(f"\nAnyUp stored split: {len(a_things)} things @thr{THR}: {a_things}")

    # 4. Build bilinear split
    print(f"\nbuilding bilinear split over all frames...")
    b_all, b_ins, nb = build_split(BIL)
    b_things, _ = things_at(b_all, b_ins)
    print(f"\nbilinear sums: all_pixels={int(b_all.sum())} (ref anyup={int(ref[ALL_KEY].sum())}) "
          f"inside={int(b_ins.sum())} (ref anyup={int(ref[INS_KEY].sum())})")
    out = {ALL_KEY: b_all.float(), INS_KEY: b_ins.float()}   # mirror reference float32
    out_p = BIL / "pseudo_classes_split_1.pt"
    torch.save(out, out_p)
    print(f"wrote {out_p}  (n={nb} frames, dtype={out[ALL_KEY].dtype})")
    print(f"Bilinear split: {len(b_things)} things @thr{THR}: {b_things}")

    # 5. Confound check: do the two splits agree?
    same = set(a_things) == set(b_things)
    print(f"\nthing-set identical across arms: {same}")
    if not same:
        print(f"  anyup-only things: {sorted(set(a_things)-set(b_things))}")
        print(f"  bilinear-only things: {sorted(set(b_things)-set(a_things))}")
    json.dump({"num_classes": NUM_CLASSES, "thr": THR, "n_bilinear": nb,
               "anyup_things": a_things, "bilinear_things": b_things,
               "thing_set_identical": same},
              open(ROOT / "analysis-output/anyup_classagnostic/stage2_thingstuff_check.json", "w"), indent=2)
    print("wrote analysis-output/anyup_classagnostic/stage2_thingstuff_check.json")


if __name__ == "__main__":
    main()
