"""Build CAUSE-TR K=27 semantic labels for all 89,250 sequence frames.

Same methodology as build_cause_tr_27_full_2975.py:
  - CAUSE-TR ClusterLookup (learned cosine-similarity prototypes, k=27)
  - sliding_window_features at 644x1288 for dense codes
  - nearest-upsample to 1024x2048
  - No AnyUp (not used by fuse_panoptic_pseudolabels.py)

Output (city-nested):
  {OUT_ROOT}/{city}/{stem}.png  (uint8, values 0-26)

Sharding:
  python scripts/build_cause_tr_27_seq.py --shard 0 --total 4   # shard 0 of 4
  python scripts/build_cause_tr_27_seq.py --shard 1 --total 4   # shard 1 of 4
  ...

Idempotent: skips frames whose output already exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.generate_depth_overclustered_semantics import (
    load_cause_models,
    sliding_window_features,
)

INST_ROOT     = Path('/Volumes/code_files_2/mbps_instances_seq/instances')
RGB_ROOT      = Path('/Volumes/code_files_2/cityscapes_sequences/cups_official_root/Cityscapes/leftImg8bit_sequence/train')
CAUSE_CKPT_DIR = PROJECT_ROOT / 'refs' / 'cause'
OUT_ROOT      = Path('/Volumes/code_files_2/mbps_instances_seq/cause_tr_27_seq')

TARGET_H, TARGET_W = 644, 1288
FULL_H,   FULL_W   = 1024, 2048

IMG_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@torch.inference_mode()
def cluster_probe_27(codes_2d: torch.Tensor, segment) -> torch.Tensor:
    """(1,90,ph,pw) -> (1,ph,pw) long cluster IDs."""
    return segment.linear.f(codes_2d).argmax(dim=1)


def _save_label(pred_t: torch.Tensor, out_path: Path) -> None:
    arr = F.interpolate(
        pred_t.unsqueeze(1).float(), size=(FULL_H, FULL_W), mode='nearest'
    ).long().squeeze().cpu().numpy().astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--total', type=int, default=1)
    a = ap.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'device: {device}  |  shard {a.shard}/{a.total}')

    print('loading CAUSE-TR ...')
    net, segment, cause_args = load_cause_models(str(CAUSE_CKPT_DIR), device)
    CROP = cause_args.crop_size
    PATCH = cause_args.patch_size

    # Enumerate all stems from instance dir
    all_stems = sorted(
        (city.name, inst.stem)
        for city in sorted(INST_ROOT.iterdir()) if city.is_dir()
        for inst in sorted(city.glob('*.png'))
    )
    # Apply shard
    stems = [s for i, s in enumerate(all_stems) if i % a.total == a.shard]
    print(f'shard {a.shard}/{a.total}: {len(stems)} frames (total {len(all_stems)})')

    n_done = n_already = n_missing = n_fail = 0
    for city, stem in tqdm(stems, desc=f'shard{a.shard}'):
        out_dir = OUT_ROOT / city
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{stem}.png'

        if out_path.exists():
            n_already += 1
            continue

        rgb_path = RGB_ROOT / city / f'{stem}_leftImg8bit.png'
        if not rgb_path.exists():
            n_missing += 1
            continue

        try:
            rgb = Image.open(rgb_path).convert('RGB').resize((TARGET_W, TARGET_H), Image.BILINEAR)
            img_t = IMG_NORM(transforms.ToTensor()(rgb)).unsqueeze(0).to(device)

            feat_90 = sliding_window_features(net, segment, img_t, crop_size=CROP).unsqueeze(0)
            ph, pw = TARGET_H // PATCH, TARGET_W // PATCH
            codes_lr = F.adaptive_avg_pool2d(feat_90, (ph, pw))  # (1, 90, 46, 92)

            _save_label(cluster_probe_27(codes_lr, segment), out_path)
            n_done += 1
        except Exception as e:
            n_fail += 1
            tqdm.write(f'FAIL {city}/{stem}: {e}')

    print(f'\nshard {a.shard}: done={n_done}  already={n_already}  missing_rgb={n_missing}  failed={n_fail}')
    print(f'output: {OUT_ROOT}')


if __name__ == '__main__':
    main()
