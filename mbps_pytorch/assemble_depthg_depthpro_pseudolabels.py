"""Assemble a Stage-2-ready CUPS-format pseudo-label cache from two pieces:

  - Semantic PNGs  : our retrained DepthG-DepthPro-monocular outputs
                     (cups_pseudo_labels_depthg_depthpro_monocular/train/<city>/<id>_semantic.png)
  - Instance PNGs+: the existing DCFA+SIMCF DepthPro instance cache
                     (cups_pseudo_labels_dcfa_simcf_v3depthpro/<id>_instance.png + <id>.pt)

The output mirrors the flat CUPS layout that gen_pseudo_labels.py produces, so it drops directly
into the Stage-2 trainer config (`DATA.ROOT_PSEUDO=<out>`).

Strategy: symlinks (no copies) so the 18-city assembly is instant and reversible.

Run:
  .venv_cups_cpu/bin/python mbps_pytorch/assemble_depthg_depthpro_pseudolabels.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_SEMANTIC_ROOT = Path("/Volumes/code_files/datasets/cityscapes/cups_pseudo_labels_depthg_depthpro_monocular/train")
DEFAULT_INSTANCE_ROOT = Path("/Volumes/code_files/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_v3depthpro")
DEFAULT_OUT = Path("/Volumes/code_files/datasets/cityscapes/cups_pseudo_labels_depthg_depthpro_monocular_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--semantic_root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    p.add_argument("--instance_root", type=Path, default=DEFAULT_INSTANCE_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--require_all", action="store_true",
                   help="error if any semantic is missing an instance partner")
    return p.parse_args()


def link(src: Path, dst: Path) -> None:
    """idempotent symlink: skip if already correct, replace if wrong, create otherwise."""
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sem_paths = sorted(args.semantic_root.rglob("*_leftImg8bit_semantic.png"))
    print(f"[assemble] {len(sem_paths)} semantic PNGs under {args.semantic_root}")
    if not sem_paths:
        sys.exit(f"no semantic PNGs found under {args.semantic_root}")

    linked = 0
    missing_inst = 0
    missing_pt = 0
    for sem in sem_paths:
        # frame stem in our cache: <id>_leftImg8bit_semantic.png
        # CUPS-format flat name: <id>_leftImg8bit_semantic.png and <id>_leftImg8bit_instance.png
        flat_sem = args.out / sem.name
        link(sem, flat_sem)
        base = sem.name[: -len("_semantic.png")]  # <id>_leftImg8bit
        inst_src = args.instance_root / f"{base}_instance.png"
        pt_src = args.instance_root / f"{base}.pt"
        if inst_src.exists():
            link(inst_src, args.out / inst_src.name)
        else:
            missing_inst += 1
            if args.require_all:
                sys.exit(f"missing instance partner for {sem.name}: {inst_src}")
        if pt_src.exists():
            link(pt_src, args.out / pt_src.name)
        else:
            missing_pt += 1
        linked += 1

    print(f"[assemble] linked {linked} semantic into {args.out}")
    print(f"[assemble] missing instance partner: {missing_inst}")
    print(f"[assemble] missing .pt partner:      {missing_pt}")
    # surface a sample for sanity
    sample_sem = next(args.out.glob("*_semantic.png"), None)
    sample_inst = next(args.out.glob("*_instance.png"), None)
    sample_pt = next(args.out.glob("*.pt"), None)
    print(f"[assemble] sample semantic : {sample_sem}")
    print(f"[assemble] sample instance : {sample_inst}")
    print(f"[assemble] sample .pt      : {sample_pt}")


if __name__ == "__main__":
    main()
