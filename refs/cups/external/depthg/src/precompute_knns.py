"""Build the NN cache required by data.py:1193 (STEGO/DepthG positive-pair mining).

For each training image, save the indices of its K nearest neighbours in DINO CLS-token feature
space. The cache file path matches the format expected by `ContrastiveSegDataset.__init__`:

    {data_dir}/nns/nns_{model_type}_{dataset}_{split}_{crop_type}_{res}.npz   key="nns" shape (N, K+1)

Row i is [i, nn1, nn2, ..., nnK]. The self-index in column 0 is what `data.py:1217`'s
`torch.randint(low=1, high=num_neighbors+1)` skips over.

Run on santosh (mirrors the training launcher):

    bash scripts/santosh_depthg_precompute_knns.sh
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.cityscapes import Cityscapes
from tqdm import tqdm

# Make the depthg/src imports resolvable. precompute_knns lives in src/, so we add
# the depthg/ parent to sys.path so `from src.modules import DinoFeaturizer` works.
DEPTHG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEPTHG_ROOT))
from src.modules import DinoFeaturizer  # noqa: E402


class _CfgShim:
    """Minimal cfg-like object DinoFeaturizer expects (attribute access, not dict)."""

    def __init__(
        self,
        model_type: str = "vit_base",
        dino_patch_size: int = 8,
        dino_feat_type: str = "feat",
        dim: int = 100,
        projection_type: str = "nonlinear",
        dropout: bool = True,
        pretrained_weights=None,
    ):
        self.model_type = model_type
        self.dino_patch_size = dino_patch_size
        self.dino_feat_type = dino_feat_type
        self.dim = dim
        self.projection_type = projection_type
        self.dropout = dropout
        self.pretrained_weights = pretrained_weights


class _CityscapesImageOnly(Dataset):
    """Plain-image loader for the NN precompute pass."""

    def __init__(self, root: str, split: str, res: int):
        self.inner = Cityscapes(root, split, mode="fine", target_type="semantic")
        self.tf = T.Compose([T.Resize((res, res)), T.ToTensor()])

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path, _ = self.inner.images[idx], self.inner.targets[idx]
        img = Image.open(img_path).convert("RGB")
        return self.tf(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", default=os.environ.get("DEPTHG_CITYSCAPES_ROOT", "/home/santosh/datasets/cityscapes"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_name", default="cityscapes")
    parser.add_argument("--crop_type", default="None", help="must match cfg.crop_type as string (Hydra null -> 'None')")
    parser.add_argument("--res", type=int, default=224)
    parser.add_argument("--model_type", default="vit_base")
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--K", type=int, default=7, help="num_neighbors")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.cityscapes_root) / "nns"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"nns_{args.model_type}_{args.dataset_name}_{args.split}_{args.crop_type}_{args.res}.npz"
    if out_path.exists():
        print(f"[precompute_knns] {out_path} already exists; skipping. Delete to regenerate.")
        return

    print(f"[precompute_knns] root={args.cityscapes_root} split={args.split} res={args.res} K={args.K}")
    ds = _CityscapesImageOnly(args.cityscapes_root, args.split, args.res)
    print(f"[precompute_knns] dataset size = {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    cfg = _CfgShim(model_type=args.model_type, dino_patch_size=args.patch_size)
    feat = DinoFeaturizer(dim=cfg.dim, cfg=cfg).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat.to(device)

    cls_feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="DINO CLS features"):
            img = batch.to(device, non_blocking=True)
            cls = feat(img, return_class_feat=True)  # (B, 768, 1, 1)
            cls = cls.flatten(1)
            cls = F.normalize(cls, dim=1)
            cls_feats.append(cls.cpu())
    cls = torch.cat(cls_feats, dim=0)
    print(f"[precompute_knns] CLS features: {tuple(cls.shape)}")

    # cosine-sim then top-K
    sim = (cls @ cls.t())
    sim.fill_diagonal_(-1.0)
    topk = torch.topk(sim, k=args.K, dim=1).indices.numpy().astype(np.int32)
    self_idx = np.arange(len(ds), dtype=np.int32)[:, None]
    nns = np.concatenate([self_idx, topk], axis=1)
    np.savez(str(out_path), nns=nns)
    print(f"[precompute_knns] wrote {out_path}  shape={nns.shape}")


if __name__ == "__main__":
    main()
