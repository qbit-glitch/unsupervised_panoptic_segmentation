#!/usr/bin/env python3
"""Evaluate vanilla or fusion-adapted codes under each baseline paper's protocol.

Side A (CAUSE): mirrors refs/cause/eval_cause_tr_dinov2.py — DINOv2 backbone ->
Segment_TR.head_ema -> [optional adapter on the (B, N, 90) tokens] -> interpolate
-> FROZEN cluster_tr probe (forward_centroid) -> NiceTool Hungarian, same
no-CRF pass + CRF/flip pass. Conditioning for adapted runs comes from the
cached *val* DepthG codes, sliced at the protocol's deterministic center crop
(Resize(322) shortest side + CenterCrop(322) => x in [0.25, 0.75] for 2:1
Cityscapes frames).

Side B (DepthG): mirrors refs/depthg/src/eval_segmentation.py — frozen
LitUnsupervisedSegmenter, val ContrastiveSegDataset (resize 320 + center crop),
flip-averaged net code -> [optional adapter on the (B, 1600, d_g) tokens] ->
interpolate -> FROZEN linear/cluster probes -> UnsupervisedMetrics, CRF on by
default. Conditioning from cached val CAUSE z, same center-crop slice.

Vanilla fidelity rule: `--vanilla` runs MUST reproduce the official numbers
(ledger R1/R2/R3) before any adapted row counts.

Usage:
    .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side A --vanilla
    .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side B --vanilla \
        --depthg_ckpt refs/depthg/saved_models/cityscapes_vitb.ckpt
    .venv_cups_cpu/bin/python mbps_pytorch/eval_fusion_adapter.py --side A \
        --adapter_ckpt results/fusion_adapter/A2_strat_w16/best.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_ROOT_DEFAULT = "/Volumes/code_files/datasets"
CACHE_ROOT_DEFAULT = "/Volumes/code_files/datasets/cityscapes/fusion_feature_cache"
MONO_CKPT = PROJECT_ROOT / "checkpoints" / "depthg_depthpro_monocular" / "epoch6_step1680.ckpt"
N_CLASSES = 27


def crop_region_tokens(grid_hwc: torch.Tensor, out_hw, x_range=(0.25, 0.75),
                       y_range=(0.0, 1.0)) -> torch.Tensor:
    """Sample a normalized sub-region of a (h, w, C) grid -> (out_h*out_w, C)."""
    c = grid_hwc.permute(2, 0, 1).unsqueeze(0)
    ys = torch.linspace(y_range[0], y_range[1], out_hw[0])
    xs = torch.linspace(x_range[0], x_range[1], out_hw[1])
    gy, gx = torch.meshgrid(ys * 2 - 1, xs * 2 - 1, indexing="ij")
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    out = F.grid_sample(c, grid, mode="bilinear", align_corners=False)
    return out.squeeze(0).permute(1, 2, 0).reshape(-1, grid_hwc.shape[-1])


def load_fusion_adapter(ckpt_path: str, device: torch.device):
    from mbps_pytorch.models.semantic.cross_model_adapter import CrossModelAdapter
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["adapter_config"]
    adapter = CrossModelAdapter(
        code_dim=cfg["code_dim"], cond_dim=cfg["cond_dim"],
        proj_width=cfg["proj_width"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(device)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    logger.info("adapter loaded: %s (side=%s, teacher=%s, epoch=%s)",
                ckpt_path, cfg["side"], cfg["teacher_mode"], ckpt.get("epoch"))
    return adapter, cfg


def build_cond_index(cache_root: str, subdir: str, suffix: str) -> Dict[str, Path]:
    """stem -> cached val feature path."""
    root = Path(cache_root) / subdir / "val"
    index = {}
    for f in root.glob(f"*/*{suffix}"):
        stem = f.name.replace(suffix, "").replace("_leftImg8bit", "")
        index[stem] = f
    return index


def normalize_stem(name: str) -> str:
    for ext in (".png", ".jpg", ".npy"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name.replace("_leftImg8bit", "")


class ConfusionMapper:
    """27x27 confusion accumulated in cluster space; Hungarian map for dumps."""

    def __init__(self, n: int = N_CLASSES) -> None:
        self.n = n
        self.hist = torch.zeros(n, n, dtype=torch.long)

    def update(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        pred = pred.reshape(-1).cpu()
        label = label.reshape(-1).cpu()
        mask = (label >= 0) & (label < self.n) & (pred >= 0) & (pred < self.n)
        self.hist += torch.bincount(
            self.n * pred[mask] + label[mask], minlength=self.n ** 2
        ).reshape(self.n, self.n)

    def mapping(self) -> np.ndarray:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(self.hist.numpy(), maximize=True)
        m = np.zeros(self.n, dtype=np.int64)
        m[row] = col
        return m


def dump_pred(dump_dir: Path, stem: str, mapped_pred: np.ndarray) -> None:
    from PIL import Image
    dump_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mapped_pred.astype(np.uint8)).save(dump_dir / f"{stem}_pred.png")


# ---------------------------------------------------------------------------
# Side A — CAUSE protocol
# ---------------------------------------------------------------------------


def run_side_a(args: argparse.Namespace, device: torch.device) -> None:
    cause_dir = PROJECT_ROOT / "refs" / "cause"
    spec = importlib.util.spec_from_file_location(
        "cause_eval", str(cause_dir / "eval_cause_tr_dinov2.py"))
    cause_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cause_eval)  # inserts refs/cause into sys.path

    from loader.dataloader import ContrastiveSegDataset  # noqa: E402
    from utils.utils import get_cococity_transform  # noqa: E402

    ca = SimpleNamespace(
        dataset="cityscapes", ckpt=str(cause_dir / "checkpoint" / "dinov2_vit_base_14.pth"),
        num_codebook=2048, reduced_dim=90, projection_dim=2048, dim=768,
        train_resolution=322, test_resolution=322, num_queries=23 * 23,
        n_classes=N_CLASSES, distributed=False, gpu="0",
        load_segment=False, load_cluster=False,
    )
    net = cause_eval.load_backbone(ca.ckpt, device)
    # load_segment_tr/load_cluster_tr join paths from the script's CAUSE_ROOT and
    # parse the *basename* of ca.ckpt — absolute path is fine.
    segment = cause_eval.load_segment_tr(ca, device)
    cluster, cb = cause_eval.load_cluster_tr(ca, device)
    segment.head.codebook = cb
    segment.head_ema.codebook = cb
    segment.eval()

    dataset = ContrastiveSegDataset(
        pytorch_data_dir=args.data_dir, dataset_name="cityscapes", crop_type=None,
        image_set="val",
        transform=get_cococity_transform(ca.test_resolution, False),
        target_transform=get_cococity_transform(ca.test_resolution, True),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)
    logger.info("side A val set: %d images", len(dataset))

    # CityscapesSeg wraps torchvision Cityscapes (leftImg8bit layout, sorted
    # city dirs then sorted files) — replicate its ordering for cond pairing.
    val_files = sorted(
        p.name for p in (Path(args.data_dir) / "cityscapes" / "leftImg8bit" / "val").glob("*/*.png"))
    assert len(val_files) == len(dataset), (len(val_files), len(dataset))

    adapter = None
    dcfa = None
    cond_index: Dict[str, Path] = {}
    if args.adapter_ckpt:
        adapter, acfg = load_fusion_adapter(args.adapter_ckpt, device)
        assert acfg["side"] == "A", "side A eval needs a side-A adapter"
        cond_index = build_cond_index(args.cache_root, args.val_g_subdir, "_g.npy")
        if not cond_index:
            raise RuntimeError(f"no cached val g under {args.cache_root}/{args.val_g_subdir}/val")
    elif args.dcfa_ckpt:
        from mbps_pytorch.models.semantic.depth_adapter import (
            DepthAdapter, sinusoidal_depth_encode)
        ck = torch.load(args.dcfa_ckpt, map_location=device, weights_only=False)
        dcfa = DepthAdapter(**{k: v for k, v in ck["adapter_kwargs"].items()
                               if k in ("code_dim", "depth_dim", "hidden_dim", "num_layers")}).to(device)
        dcfa.load_state_dict(ck["state_dict"])
        dcfa.eval()
        self_encode = sinusoidal_depth_encode
        logger.info("DCFA adapter loaded: %s", args.dcfa_ckpt)

    def cond_for(batch_inds: torch.Tensor, out_hw) -> Optional[torch.Tensor]:
        if adapter is None:
            return None
        toks = []
        for ind in batch_inds.tolist():
            stem = normalize_stem(val_files[ind])
            g = torch.from_numpy(np.load(cond_index[stem]).astype(np.float32))
            toks.append(crop_region_tokens(g, out_hw))
        return torch.stack(toks).to(device)

    def depth_cond_for(batch_inds: torch.Tensor, out_hw) -> Optional[torch.Tensor]:
        if dcfa is None:
            return None
        toks = []
        droot = Path(args.data_dir) / "cityscapes" / "depth_depthpro" / "val"
        for ind in batch_inds.tolist():
            stem = normalize_stem(val_files[ind])
            city = stem.split("_")[0]
            dp = droot / city / f"{stem}_leftImg8bit.npy"
            if not dp.is_file():
                dp = droot / city / f"{stem}.npy"
            d = torch.from_numpy(np.load(dp).astype(np.float32)).unsqueeze(-1)
            t = crop_region_tokens(d, out_hw).squeeze(-1)
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)
            toks.append(t)
        return self_encode(torch.stack(toks).to(device))

    def inject(seg_2d: torch.Tensor, batch_inds: torch.Tensor) -> torch.Tensor:
        """seg_2d: (B, 90, h, w) -> adapted (B, 90, h, w)."""
        if adapter is None and dcfa is None:
            return seg_2d
        b, c, h, w = seg_2d.shape
        toks = seg_2d.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if adapter is not None:
            toks = adapter(toks, cond_for(batch_inds, (h, w)))
        else:
            toks = dcfa(toks, depth_cond_for(batch_inds, (h, w)))
        return toks.reshape(b, h, w, c).permute(0, 3, 1, 2)

    transform = cause_eval.transform
    untransform = cause_eval.untransform

    def eval_pass(use_crf: bool) -> Dict[str, float]:
        nice = cause_eval.NiceTool(N_CLASSES, device)
        mapper = ConfusionMapper()
        stored = []  # (stem, raw_pred uint8) for dumps
        pool = None
        if use_crf:
            from multiprocessing import Pool
            pool = Pool(args.num_pool_workers)
        metric_dict: Dict[str, float] = {}
        for bi, batch in enumerate(loader):
            if args.limit and bi >= args.limit:
                break
            img = batch["img"].to(device)
            label = batch["label"].to(device)
            inds = batch["ind"]
            with torch.no_grad():
                feat = net(img)[:, 1:, :]
                if use_crf:
                    feat_flip = net(img.flip(dims=[3]))[:, 1:, :]
                    seg = transform(segment.head_ema(feat))
                    seg_f = transform(segment.head_ema(feat_flip))
                    seg = (seg + seg_f.flip(dims=[3])) / 2
                else:
                    seg = transform(segment.head_ema(feat))
                seg = inject(seg, inds)
                interp = F.interpolate(seg, label.shape[-2:], mode="bilinear",
                                       align_corners=False)
                if use_crf:
                    logits = cluster.forward_centroid(untransform(interp), crf=True)
                    preds = cause_eval.do_crf(pool, img, logits).argmax(1).to(device)
                else:
                    preds = cluster.forward_centroid(untransform(interp), inference=True)
                metric_dict, desc = nice.eval(preds, label)
            mapper.update(preds, label)
            if args.dump_preds:
                for k, ind in enumerate(inds.tolist()):
                    stored.append((normalize_stem(val_files[ind]),
                                   preds[k].cpu().numpy().astype(np.uint8)))
            if bi % 10 == 0:
                logger.info("[A][%s] batch %d/%d %s",
                            "CRF" if use_crf else "noCRF", bi, len(loader), desc)
        if pool is not None:
            pool.close()
        if args.dump_preds and stored:
            m = mapper.mapping()
            dump_dir = Path(args.dump_preds)
            for stem, raw in stored:
                dump_pred(dump_dir, stem, m[raw])
            with open(dump_dir / "mapping_and_metrics.json", "w") as fh:
                json.dump({"mapping": m.tolist(), "metrics": metric_dict,
                           "crf": use_crf}, fh, indent=2)
            logger.info("dumped %d mapped preds -> %s", len(stored), dump_dir)
        return metric_dict

    res_nocrf = eval_pass(use_crf=False)
    logger.info("[A] FINAL noCRF: %s", res_nocrf)
    if not args.skip_crf:
        res_crf = eval_pass(use_crf=True)
        logger.info("[A] FINAL CRF: %s", res_crf)


# ---------------------------------------------------------------------------
# Side B — DepthG protocol
# ---------------------------------------------------------------------------


def run_side_b(args: argparse.Namespace, device: torch.device) -> None:
    dg_src = PROJECT_ROOT / "refs" / "depthg" / "src"
    sys.path.insert(0, str(dg_src))
    from train_segmentation import LitUnsupervisedSegmenter  # noqa: E402
    from data import ContrastiveSegDataset  # noqa: E402
    from utils import get_transform  # noqa: E402
    from crf import dense_crf  # noqa: E402

    model = LitUnsupervisedSegmenter.load_from_checkpoint(
        str(args.depthg_ckpt), map_location=device, weights_only=False)
    model.eval().to(device)
    res = 320

    dataset = ContrastiveSegDataset(
        data_dir=args.data_dir, dataset_name="cityscapes", crop_type=None,
        image_set="val", transform=get_transform(res, False, "center"),
        target_transform=get_transform(res, True, "center"), cfg=model.cfg,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)
    logger.info("side B val set: %d images (ckpt=%s)", len(dataset), args.depthg_ckpt)

    val_files = sorted(
        str(p) for p in (Path(args.data_dir) / "cityscapes" / "leftImg8bit" / "val").glob("*/*.png"))
    assert len(val_files) == len(dataset), (len(val_files), len(dataset))

    adapter = None
    cond_index: Dict[str, Path] = {}
    if args.adapter_ckpt:
        adapter, acfg = load_fusion_adapter(args.adapter_ckpt, device)
        assert acfg["side"] == "B", "side B eval needs a side-B adapter"
        cond_index = build_cond_index(args.cache_root, "cause_z", "_codes.npy")
        if not cond_index:
            raise RuntimeError(f"no cached val z under {args.cache_root}/cause_z/val")

    def inject(code_2d: torch.Tensor, batch_inds: torch.Tensor) -> torch.Tensor:
        if adapter is None:
            return code_2d
        b, c, h, w = code_2d.shape
        toks = code_2d.permute(0, 2, 3, 1).reshape(b, h * w, c)
        conds = []
        for ind in batch_inds.tolist():
            stem = normalize_stem(Path(val_files[ind]).name)
            z = torch.from_numpy(np.load(cond_index[stem]).astype(np.float32))
            conds.append(crop_region_tokens(z, (h, w)))
        toks = adapter(toks, torch.stack(conds).to(device))
        return toks.reshape(b, h, w, c).permute(0, 3, 1, 2)

    # frozen probes + the model's own test metrics ("final/..." keys)
    mapper = ConfusionMapper()
    stored = []
    for bi, batch in enumerate(loader):
        if args.limit and bi >= args.limit:
            break
        img = batch["img"].to(device)
        label = batch["label"].to(device)
        inds = batch["ind"]
        with torch.no_grad():
            _, code1 = model.net(img)
            _, code2 = model.net(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2
            code = inject(code, inds)
            code = F.interpolate(code, label.shape[-2:], mode="bilinear",
                                 align_corners=False)
            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)
            cluster_probs = model.cluster_probe(code, 2, log_probs=True)
            if not args.skip_crf:
                linear_preds = torch.cat([
                    torch.from_numpy(dense_crf(i.cpu(), p.cpu())).unsqueeze(0)
                    for i, p in zip(img, linear_probs)]).argmax(1).to(device)
                cluster_preds = torch.cat([
                    torch.from_numpy(dense_crf(i.cpu(), p.cpu())).unsqueeze(0)
                    for i, p in zip(img, cluster_probs)]).argmax(1).to(device)
            else:
                linear_preds = linear_probs.argmax(1)
                cluster_preds = cluster_probs.argmax(1)
            model.test_linear_metrics.update(linear_preds, label)
            model.test_cluster_metrics.update(cluster_preds, label)
        mapper.update(cluster_preds, label)
        if args.dump_preds:
            for k, ind in enumerate(inds.tolist()):
                stored.append((normalize_stem(Path(val_files[ind]).name),
                               cluster_preds[k].cpu().numpy().astype(np.uint8)))
        if bi % 10 == 0:
            logger.info("[B] batch %d/%d", bi, len(loader))

    # UnsupervisedMetrics.compute() already returns percentages
    tb = {**model.test_linear_metrics.compute(), **model.test_cluster_metrics.compute()}
    tb = {k: (v.item() if torch.is_tensor(v) else v) for k, v in tb.items()}
    logger.info("[B] FINAL: %s", {k: round(v, 3) for k, v in tb.items()})
    out = {**tb,
           "model_path": str(args.depthg_ckpt), "num_images": len(dataset),
           "crf_enabled": not args.skip_crf,
           "adapter_ckpt": args.adapter_ckpt or "vanilla"}
    metrics_path = Path(args.dump_preds or "results/fusion_adapter") / "side_b_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("metrics written -> %s", metrics_path)

    if args.dump_preds and stored:
        m = mapper.mapping()
        dump_dir = Path(args.dump_preds)
        for stem, raw in stored:
            dump_pred(dump_dir, stem, m[raw])
        logger.info("dumped %d mapped preds -> %s", len(stored), dump_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", required=True, choices=("A", "B"))
    p.add_argument("--vanilla", action="store_true",
                   help="no adapter — protocol-fidelity mode")
    p.add_argument("--adapter_ckpt", default=None)
    p.add_argument("--dcfa_ckpt", default=None, help="side A attribution row (T10)")
    p.add_argument("--depthg_ckpt", default=str(MONO_CKPT), help="side B substrate")
    p.add_argument("--data_dir", default=DATA_ROOT_DEFAULT)
    p.add_argument("--cache_root", default=CACHE_ROOT_DEFAULT)
    p.add_argument("--val_g_subdir", default="depthg_g_mono")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_pool_workers", type=int, default=4)
    p.add_argument("--skip_crf", action="store_true")
    p.add_argument("--dump_preds", default=None)
    p.add_argument("--limit", type=int, default=0, help="batches cap (smoke)")
    p.add_argument("--device", default=None, choices=("cpu", "mps"))
    args = p.parse_args()

    if not args.vanilla and not args.adapter_ckpt and not args.dcfa_ckpt:
        p.error("pass --vanilla, --adapter_ckpt, or --dcfa_ckpt")
    if args.vanilla:
        args.adapter_ckpt = None
        args.dcfa_ckpt = None

    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    logger.info("side=%s vanilla=%s device=%s", args.side, args.vanilla, device)
    if args.side == "A":
        run_side_a(args, device)
    else:
        run_side_b(args, device)


if __name__ == "__main__":
    main()
