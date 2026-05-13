"""Local eval of Path-C: Stage-3 cluster head + AuxThingAdapter with eval-time fusion.

Produces per-class PQ on Cityscapes val using CUPS-standard PanopticQualitySemanticMatching.
Uses cached P4 features to skip the heavy backbone forward.

Pipeline:
  1. Pass 1: run frozen Stage-3 model and build the CUPS k80 -> CAUSE-27
     global Hungarian assignment.
  2. Pass 2: remap Stage-3 predictions to CAUSE-27, load cached P4 features,
     run AuxThingAdapter, then override high-confidence adapter thing pixels
     in CAUSE-27 space. A threshold sweep is evaluated in one shared pass.
  3. Stream both baseline and fused predictions through torchmetrics PQ.

CLI:
    python scripts/eval_aux_thing_adapter.py \
        --stage3_ckpt checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt \
        --adapter_ckpt checkpoints/aux_thing_adapter_run1/best.pt \
        --cfg refs/cups/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --val_feat_cache /Users/qbit-glitch/Desktop/datasets/cityscapes/p4_cache_stage3_val \
        --confidence_threshold 0.5 \
        --device cpu \
        --out reports/path_c_aux_thing_results.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# SAM3 fine-class index -> Cityscapes CAUSE-27 class ID (raw Cityscapes ID - 7).
# The eval dataset in CUPS uses CAUSE-27, not Cityscapes trainID-19. Keeping this
# mapping local avoids mixing target class IDs into the 80-cluster space.
SAM3_TO_CITYSCAPES_27: Dict[int, int] = {
    0: 17,  # person
    1: 26,  # bicycle
    2: 25,  # motorcycle
    3: 18,  # rider
    4: 13,  # traffic sign
    5: 12,  # traffic light
    6: 20,  # truck
    7: 21,  # bus
    8: 24,  # train
    # 9: guard rail, kept disabled by default because this is a thing adapter.
    10: 22,  # caravan
    11: 23,  # trailer
    12: 19,  # car
    13: 10,  # pole
}

METRIC_KEYS = (
    "pq",
    "sq",
    "rq",
    "pq_per_class",
    "sq_per_class",
    "rq_per_class",
    "pq_things",
    "sq_things",
    "rq_things",
    "pq_stuffs",
    "sq_stuffs",
    "rq_stuffs",
    "miou",
    "acc",
)


def _parse_thresholds(raw: str | None, fallback: float) -> List[float]:
    if raw is None:
        return [float(fallback)]
    thresholds: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))
    if not thresholds:
        raise ValueError("--confidence_thresholds did not contain any numeric thresholds")
    return sorted(set(thresholds))


def _import_cups(project_root: Path):
    """Add CUPS paths to sys.path."""
    cups_root = project_root / "refs" / "cups"
    for p in (project_root, cups_root):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _load_stage3_model(ckpt: Path, cfg: Path, device: str):
    """Load the Stage-3 model in eval mode."""
    import cups
    from cups.augmentation import PhotometricAugmentations, ResolutionJitter
    from cups.data import (
        CITYSCAPES_THING_CLASSES,
        CITYSCAPES_STUFF_CLASSES,
        CITYSCAPES_CLASSNAMES,
    )

    config = cups.get_default_config(experiment_config_file=str(cfg))
    config.defrost()
    config.MODEL.CHECKPOINT = str(ckpt)
    config.SYSTEM.ACCELERATOR = device if not device.startswith("cuda") else "gpu"
    config.freeze()

    model = cups.build_model_self(
        config=config,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        class_names=CITYSCAPES_CLASSNAMES,
        photometric_augmentation=PhotometricAugmentations(),
        freeze_bn=True,
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None, resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def _load_adapter(adapter_ckpt: Path, device: str):
    """Load the AuxThingAdapter from a training-output checkpoint."""
    from cups.model.aux_thing_adapter import AuxThingAdapter

    # weights_only=False: ckpt also stores `args` (a dict with PosixPath fields)
    # which `weights_only=True` rejects. We wrote the file ourselves, so trust it.
    state = torch.load(adapter_ckpt, weights_only=False, map_location=device)
    args = state.get("args", {})
    adapter = AuxThingAdapter(
        in_dim=args.get("in_dim", 256),
        hidden_dim=args.get("hidden_dim", 64),
        num_sam3_classes=args.get("num_sam3_classes", 14),
    ).to(device).eval()
    adapter.load_state_dict(state["adapter_state"])
    print(f"[INFO] Loaded adapter (epoch={state.get('epoch','?')}, "
          f"loss={state.get('best_loss', state.get('loss', '?'))})", flush=True)
    return adapter


def _list_val_images(cityscapes_root: Path) -> List[Path]:
    val_root = cityscapes_root / "leftImg8bit" / "val"
    paths = sorted(val_root.rglob("*_leftImg8bit.png"))
    return paths


def _load_cached_p4(feat_cache: Path, image_id: str) -> Optional[torch.Tensor]:
    """Load cached P4 feature for one image."""
    p = feat_cache / f"{image_id}.pt"
    if not p.exists():
        return None
    return torch.load(p, weights_only=True).float()  # (256, H, W)


def _image_id_from_name(image_name: str) -> str:
    raw_id = image_name.split("/")[-1]
    return raw_id.replace("_leftImg8bit.png", "").replace("_leftImg8bit", "")


def _prepare_target(panoptic_labels: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
    """Normalize CUPS validation target to ``(1, H, W, 2)``."""
    if isinstance(panoptic_labels, (list, tuple)):
        target = panoptic_labels[0]
        if target.ndim == 3:
            target = target.unsqueeze(0)
    else:
        target = panoptic_labels
    if target.ndim != 4:
        raise ValueError(f"unexpected target shape {tuple(target.shape)}")
    return target.long().cpu()


def _metric_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _compute_miou(cost_matrix: torch.Tensor) -> tuple[float, float]:
    cost_matrix = cost_matrix.float()
    tp = torch.diag(cost_matrix)
    fp = cost_matrix.sum(dim=0) - tp
    fn = cost_matrix.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.nan)
    miou = torch.nanmean(iou)
    acc = tp.sum() / cost_matrix.sum().clamp_min(1)
    return _metric_float(miou), _metric_float(acc)


def _extract_streamed_pq(metric, compute_result: torch.Tensor, num_classes: int) -> Dict[str, object]:
    """Convert torchmetrics PanopticQuality per-class output into CUPS-style metrics."""
    per_cont = compute_result.detach().cpu().float()
    if per_cont.ndim != 2 or per_cont.shape[1] != 3:
        raise ValueError(f"unexpected PanopticQuality result shape: {tuple(per_cont.shape)}")

    pq_per_class = torch.zeros(num_classes, dtype=per_cont.dtype)
    sq_per_class = torch.zeros(num_classes, dtype=per_cont.dtype)
    rq_per_class = torch.zeros(num_classes, dtype=per_cont.dtype)
    denom_per_class = torch.zeros(num_classes, dtype=per_cont.dtype)

    iou_sum = metric.iou_sum.detach().cpu().float()
    tp = metric.true_positives.detach().cpu().float()
    fp = metric.false_positives.detach().cpu().float()
    fn = metric.false_negatives.detach().cpu().float()
    denom_cont = tp + 0.5 * fp + 0.5 * fn

    for class_id, continuous_id in metric.cat_id_to_continuous_id.items():
        class_id = int(class_id)
        continuous_id = int(continuous_id)
        if 0 <= class_id < num_classes:
            pq_per_class[class_id] = per_cont[continuous_id, 0]
            sq_per_class[class_id] = per_cont[continuous_id, 1]
            rq_per_class[class_id] = per_cont[continuous_id, 2]
            denom_per_class[class_id] = denom_cont[continuous_id]

    valid = denom_per_class > 0
    return {
        "pq": _metric_float(pq_per_class[valid].mean()) if valid.any() else 0.0,
        "sq": _metric_float(sq_per_class[valid].mean()) if valid.any() else 0.0,
        "rq": _metric_float(rq_per_class[valid].mean()) if valid.any() else 0.0,
        "pq_per_class": pq_per_class,
        "sq_per_class": sq_per_class,
        "rq_per_class": rq_per_class,
        "denom_per_class": denom_per_class,
    }


def _summarize_metric_dict(
    result: Dict[str, object],
    things_set: set[int],
    stuffs_set: set[int],
) -> Dict[str, object]:
    pq_c = result["pq_per_class"]
    sq_c = result["sq_per_class"]
    rq_c = result["rq_per_class"]
    denom = result["denom_per_class"]
    assert isinstance(pq_c, torch.Tensor)
    assert isinstance(sq_c, torch.Tensor)
    assert isinstance(rq_c, torch.Tensor)
    assert isinstance(denom, torch.Tensor)

    thing_ids = torch.tensor(sorted(things_set), dtype=torch.long)
    stuff_ids = torch.tensor(sorted(stuffs_set), dtype=torch.long)
    thing_valid = denom[thing_ids] > 0
    stuff_valid = denom[stuff_ids] > 0

    result["pq_things"] = _metric_float(pq_c[thing_ids][thing_valid].mean()) if thing_valid.any() else 0.0
    result["sq_things"] = _metric_float(sq_c[thing_ids][thing_valid].mean()) if thing_valid.any() else 0.0
    result["rq_things"] = _metric_float(rq_c[thing_ids][thing_valid].mean()) if thing_valid.any() else 0.0
    result["pq_stuffs"] = _metric_float(pq_c[stuff_ids][stuff_valid].mean()) if stuff_valid.any() else 0.0
    result["sq_stuffs"] = _metric_float(sq_c[stuff_ids][stuff_valid].mean()) if stuff_valid.any() else 0.0
    result["rq_stuffs"] = _metric_float(rq_c[stuff_ids][stuff_valid].mean()) if stuff_valid.any() else 0.0
    return result


def _format_scalar_metrics(title: str, result: Dict[str, object]) -> List[str]:
    return [
        f"{title}:",
        f"  PQ={float(result['pq']) * 100:.2f}",
        f"  SQ={float(result['sq']) * 100:.2f}",
        f"  RQ={float(result['rq']) * 100:.2f}",
        f"  PQ_things={float(result['pq_things']) * 100:.2f}",
        f"  PQ_stuff={float(result['pq_stuffs']) * 100:.2f}",
        f"  mIoU={float(result['miou']) * 100:.2f}",
        f"  Acc={float(result['acc']) * 100:.2f}",
    ]


def _per_class_delta_rows(
    baseline: Dict[str, object],
    fused: Dict[str, object],
    class_names: List[str],
) -> List[str]:
    base_pq = baseline["pq_per_class"]
    fused_pq = fused["pq_per_class"]
    assert isinstance(base_pq, torch.Tensor)
    assert isinstance(fused_pq, torch.Tensor)
    rows = ["| cls | class | baseline PQ | fused PQ | delta |", "|---:|---|---:|---:|---:|"]
    for idx, name in enumerate(class_names):
        b = float(base_pq[idx]) * 100.0
        f = float(fused_pq[idx]) * 100.0
        rows.append(f"| {idx} | {name} | {b:.2f} | {f:.2f} | {f - b:+.2f} |")
    return rows


def _build_adapter_c27_prediction(
    adapter_logits_full: torch.Tensor,
    confidence_threshold: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(adapter_logits_full, dim=1)
    conf, c_star = probs.max(dim=1)
    lut = torch.full((adapter_logits_full.shape[1],), 255, dtype=torch.long, device=device)
    for src, dst in SAM3_TO_CITYSCAPES_27.items():
        if 0 <= src < lut.numel():
            lut[src] = int(dst)
    adapter_c27 = lut[c_star]
    active = (conf > confidence_threshold) & (adapter_c27 != 255)
    return adapter_c27, active


def _new_pq_metric(things_set: set[int], stuffs_set: set[int]):
    from torchmetrics.detection import PanopticQuality as PanopticQualityTM

    return PanopticQualityTM(
        things=things_set,
        stuffs=stuffs_set,
        allow_unknown_preds_category=True,
        return_per_class=True,
        return_sq_and_rq=True,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3_ckpt", required=True, type=Path)
    parser.add_argument("--adapter_ckpt", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--cityscapes_root", required=True, type=Path)
    parser.add_argument("--val_feat_cache", required=True, type=Path,
                        help="Directory with cached P4 features for val split.")
    parser.add_argument("--confidence_threshold", default=0.5, type=float,
                        help="Adapter softmax-max threshold tau for fusion.")
    parser.add_argument(
        "--confidence_thresholds",
        default=None,
        type=str,
        help="Comma-separated threshold sweep, e.g. 0.5,0.7,0.8,0.9,0.95,0.98,0.99.",
    )
    parser.add_argument(
        "--fusion_class_scope",
        default="things",
        choices=["things", "all"],
        help="Default 'things' only fuses Cityscapes thing classes; 'all' also lets adapter overwrite stuff classes.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--out", default=None, type=Path,
                        help="Output markdown report (default: print to stdout).")
    parser.add_argument("--max_images", default=None, type=int,
                        help="Cap for fast verification.")
    parser.add_argument(
        "--allow_adapter_create_thing_instances",
        action="store_true",
        help=(
            "Allow adapter thing-class overrides even where CUPS has no instance ID. "
            "Default is conservative: thing-class overrides require an existing "
            "CUPS instance mask."
        ),
    )
    args = parser.parse_args()
    thresholds = _parse_thresholds(args.confidence_thresholds, args.confidence_threshold)

    project_root = Path(__file__).resolve().parents[1]
    _import_cups(project_root)

    print(f"[INFO] Loading Stage-3 model from {args.stage3_ckpt}", flush=True)
    model, config = _load_stage3_model(args.stage3_ckpt, args.cfg, args.device)

    print(f"[INFO] Loading AuxThingAdapter from {args.adapter_ckpt}", flush=True)
    adapter = _load_adapter(args.adapter_ckpt, args.device)

    from cups.metrics.panoptic_quality import PanopticQualitySemanticMatching
    from cups.data import (
        CITYSCAPES_CLASSNAMES, CITYSCAPES_THING_CLASSES, CITYSCAPES_STUFF_CLASSES,
        CityscapesPanopticValidation, collate_function_validation,
    )
    from cups.model.model import prediction_to_standard_format
    from torch.utils.data import DataLoader

    # Build the standard CUPS val data loader
    print(f"[INFO] Building CityscapesPanopticValidation", flush=True)
    val_dataset = CityscapesPanopticValidation(
        root=str(args.cityscapes_root) + "/",
        resize_scale=config.DATA.VAL_SCALE,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        num_classes=config.DATA.NUM_CLASSES,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_function_validation,
        drop_last=False,
        pin_memory=False,
    )

    # Standard CUPS class/prototype definitions.
    n_total = config.DATA.NUM_CLASSES
    things_set = set(int(c) for c in CITYSCAPES_THING_CLASSES)
    stuffs_set = set(int(c) for c in CITYSCAPES_STUFF_CLASSES)
    ours_thing_pseudo = tuple(int(x) for x in model.hparams.thing_pseudo_classes)
    ours_stuff_pseudo = tuple(int(x) for x in model.hparams.stuff_pseudo_classes)
    num_clusters_total = len(ours_thing_pseudo) + len(ours_stuff_pseudo)
    print(f"[INFO] Pseudo-class tuples: things={len(ours_thing_pseudo)}, "
          f"stuff={len(ours_stuff_pseudo)}", flush=True)

    # Metric helper used only for its CUPS matching and cost-matrix implementation.
    pq_helper = PanopticQualitySemanticMatching(
        things=things_set,
        stuffs=stuffs_set,
        num_clusters=num_clusters_total,
        things_prototype=set(ours_thing_pseudo) if config.VALIDATION.ADHERE_THING_STUFF else None,
        stuffs_prototype=set(ours_stuff_pseudo) if config.VALIDATION.ADHERE_THING_STUFF else None,
        cache_device="cpu",
        sync_on_compute=False,
        dist_sync_on_step=False,
    )

    # ── PASS 1: CUPS global Hungarian assignment, no prediction cache ─────
    n_pass1 = 0
    t_start = time.time()
    cost_matrix = torch.zeros(num_clusters_total, n_total, dtype=torch.float32)
    print("\n[INFO] PASS 1/2: building CUPS k80 -> CAUSE-27 assignment", flush=True)
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="eval")):
        if args.max_images is not None and n_pass1 >= args.max_images:
            break
        images, panoptic_labels, image_names = batch  # batch_size=1
        image_dict = images[0]
        image_id = _image_id_from_name(image_names[0])

        try:
            pred = model([image_dict])[0]
        except Exception as e:
            print(f"[WARN] {image_id}: model forward failed: {e}", flush=True)
            continue

        # Convert Detectron2 (panoptic_seg, segments_info) → (H, W, 2) tensor
        try:
            pano_std = prediction_to_standard_format(
                pred["panoptic_seg"],
                stuff_classes=ours_stuff_pseudo,
                thing_classes=ours_thing_pseudo,
            )  # (H, W, 2): [..., 0] = category_id, [..., 1] = instance_id
        except Exception as e:
            print(f"[WARN] {image_id}: prediction_to_standard_format failed: {e}", flush=True)
            continue

        try:
            target = _prepare_target(panoptic_labels)
        except ValueError as e:
            print(f"[WARN] {image_id}: {e}", flush=True)
            continue

        cost_matrix += pq_helper._cost_matrix_update(
            pano_std[..., 0].reshape(-1).cpu(),
            target[..., 0].reshape(-1).cpu(),
            n_total,
            num_clusters_total,
        ).cpu()
        n_pass1 += 1

    if n_pass1 == 0:
        raise RuntimeError("No validation images were evaluated in pass 1.")

    pq_helper.cost_matrix = cost_matrix.to(pq_helper.cost_matrix.device)
    pq_assignments = pq_helper.matching().cpu().long()
    miou_assignments = pq_helper._matching_no_separation().cpu().long()
    print(f"[INFO] Assignment built from {n_pass1} images.", flush=True)

    # ── PASS 2: stream baseline + fused PQ in target CAUSE-27 space ───────
    metric_baseline = _new_pq_metric(things_set, stuffs_set)
    metric_fused = {tau: _new_pq_metric(things_set, stuffs_set) for tau in thresholds}
    cost_miou_baseline = torch.zeros(n_total, n_total, dtype=torch.float32)
    cost_miou_fused = {tau: torch.zeros(n_total, n_total, dtype=torch.float32) for tau in thresholds}
    thing_class_tensor = torch.tensor(sorted(things_set), dtype=torch.long, device=args.device)

    n_evaluated = 0
    n_skipped_no_cache = 0
    n_adapter_pixels = {tau: 0 for tau in thresholds}
    n_adapter_thing_pixels_suppressed = {tau: 0 for tau in thresholds}
    print("\n[INFO] PASS 2/2: evaluating baseline and adapter-fused predictions", flush=True)
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="eval fused")):
        if args.max_images is not None and n_evaluated >= args.max_images:
            break
        images, panoptic_labels, image_names = batch
        image_dict = images[0]
        image_id = _image_id_from_name(image_names[0])

        try:
            pred = model([image_dict])[0]
            pano_std = prediction_to_standard_format(
                pred["panoptic_seg"],
                stuff_classes=ours_stuff_pseudo,
                thing_classes=ours_thing_pseudo,
            ).long().cpu()
            target = _prepare_target(panoptic_labels)
        except Exception as e:
            print(f"[WARN] {image_id}: pass-2 baseline prep failed: {e}", flush=True)
            continue

        baseline_target = PanopticQualitySemanticMatching.map_to_target(
            pano_std.unsqueeze(0), pq_assignments
        ).long().cpu()
        baseline_miou = PanopticQualitySemanticMatching.map_to_target(
            pano_std.unsqueeze(0), miou_assignments
        ).long().cpu()

        metric_baseline.update(baseline_target, target)
        cost_miou_baseline += pq_helper._cost_matrix_update(
            baseline_miou[..., 0].reshape(-1),
            target[..., 0].reshape(-1),
            n_total,
            n_total,
        ).cpu()

        adapter_c27 = None
        adapter_conf = None
        adapter_is_thing = None
        has_cups_instance = baseline_target[0, ..., 1].to(args.device) > 0
        feat = _load_cached_p4(args.val_feat_cache, image_id)
        if feat is None:
            n_skipped_no_cache += 1
        else:
            feat = feat.unsqueeze(0).to(args.device)
            adapter_logits = adapter(feat)
            adapter_logits_full = F.interpolate(
                adapter_logits,
                size=baseline_target.shape[1:3],
                mode="bilinear",
                align_corners=False,
            )
            probs = F.softmax(adapter_logits_full, dim=1)
            adapter_conf, c_star = probs.max(dim=1)
            adapter_conf = adapter_conf[0]
            lut = torch.full((adapter_logits_full.shape[1],), 255, dtype=torch.long, device=args.device)
            for src, dst in SAM3_TO_CITYSCAPES_27.items():
                if 0 <= src < lut.numel():
                    lut[src] = int(dst)
            adapter_c27 = lut[c_star][0]
            adapter_is_thing = torch.isin(adapter_c27, thing_class_tensor)

        for tau in thresholds:
            pano_fused = baseline_target[0].clone()
            if adapter_c27 is not None and adapter_conf is not None and adapter_is_thing is not None:
                adapter_active = (adapter_conf > tau) & (adapter_c27 != 255)
                if args.fusion_class_scope == "things":
                    adapter_active = adapter_active & adapter_is_thing
                suppressed = adapter_active & adapter_is_thing & ~has_cups_instance
                if not args.allow_adapter_create_thing_instances:
                    adapter_active = adapter_active & (~adapter_is_thing | has_cups_instance)
                n_adapter_thing_pixels_suppressed[tau] += int(suppressed.sum().item())
                n_adapter_pixels[tau] += int(adapter_active.sum().item())
                pano_fused[..., 0] = torch.where(
                    adapter_active.cpu(),
                    adapter_c27.cpu(),
                    pano_fused[..., 0],
                )

            fused_batch = pano_fused.unsqueeze(0).long().cpu()
            metric_fused[tau].update(fused_batch, target)
            cost_miou_fused[tau] += pq_helper._cost_matrix_update(
                fused_batch[..., 0].reshape(-1),
                target[..., 0].reshape(-1),
                n_total,
                n_total,
            ).cpu()

        n_evaluated += 1

    elapsed = time.time() - t_start
    print(f"\n[INFO] Eval done in {elapsed/60:.1f} min. "
          f"pass1={n_pass1}, evaluated={n_evaluated}, skipped_no_cache={n_skipped_no_cache}, "
          f"thresholds={thresholds}",
          flush=True)

    if n_evaluated == 0:
        raise RuntimeError("No validation images were evaluated in pass 2.")

    baseline_results = _summarize_metric_dict(
        _extract_streamed_pq(metric_baseline, metric_baseline.compute(), n_total),
        things_set,
        stuffs_set,
    )
    fused_results_by_tau: Dict[float, Dict[str, object]] = {}
    for tau in thresholds:
        fused_results_by_tau[tau] = _summarize_metric_dict(
            _extract_streamed_pq(metric_fused[tau], metric_fused[tau].compute(), n_total),
            things_set,
            stuffs_set,
        )
    baseline_results["miou"], baseline_results["acc"] = _compute_miou(cost_miou_baseline)
    baseline_results["assignments"] = pq_assignments
    for tau, res in fused_results_by_tau.items():
        res["miou"], res["acc"] = _compute_miou(cost_miou_fused[tau])
        res["assignments"] = pq_assignments

    print("\n===== Baseline (Stage-3 cluster head, CUPS Hungarian) =====", flush=True)
    print("\n".join(_format_scalar_metrics("baseline", baseline_results)), flush=True)

    print("\n===== Fused Sweep (Path-C adapter + cluster head) =====", flush=True)
    best_tau = max(
        thresholds,
        key=lambda tau: (
            float(fused_results_by_tau[tau]["pq_things"]),
            float(fused_results_by_tau[tau]["pq"]),
        ),
    )
    for tau in thresholds:
        res = fused_results_by_tau[tau]
        delta_pq = (float(res["pq"]) - float(baseline_results["pq"])) * 100.0
        delta_pqt = (float(res["pq_things"]) - float(baseline_results["pq_things"])) * 100.0
        print(
            f"tau={tau:.3f}: PQ={float(res['pq']) * 100:.2f} ({delta_pq:+.2f}), "
            f"PQ_things={float(res['pq_things']) * 100:.2f} ({delta_pqt:+.2f}), "
            f"active_px={n_adapter_pixels[tau]}, suppressed_px={n_adapter_thing_pixels_suppressed[tau]}",
            flush=True,
        )
    fused_results = fused_results_by_tau[best_tau]
    delta_pq = (float(fused_results["pq"]) - float(baseline_results["pq"])) * 100.0
    delta_pqt = (float(fused_results["pq_things"]) - float(baseline_results["pq_things"])) * 100.0
    print(f"\n[INFO] Best by PQ_things: tau={best_tau:.3f}; Delta PQ={delta_pq:+.2f}, "
          f"Delta PQ_things={delta_pqt:+.2f}", flush=True)

    # Save report
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            f.write("# Path-C AuxThingAdapter Eval Results\n\n")
            f.write("**Protocol:** CUPS k80 global Hungarian assignment, then target-space Path-C fusion.\n")
            f.write(f"**Adapter:** {args.adapter_ckpt}\n")
            f.write(f"**Stage-3 checkpoint:** {args.stage3_ckpt}\n")
            f.write(f"**Fusion class scope:** {args.fusion_class_scope}\n")
            f.write(f"**Thresholds:** {thresholds}\n")
            f.write(f"**Best threshold by PQ_things:** {best_tau:.3f}\n\n")
            f.write("## Summary\n\n")
            f.write("| method | PQ | SQ | RQ | PQ_stuff | PQ_things | mIoU | Acc |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for name, res in (("Stage-3 baseline", baseline_results), (f"Path-C fused tau={best_tau:.3f}", fused_results)):
                f.write(
                    f"| {name} | {float(res['pq']) * 100:.2f} | {float(res['sq']) * 100:.2f} | "
                    f"{float(res['rq']) * 100:.2f} | {float(res['pq_stuffs']) * 100:.2f} | "
                    f"{float(res['pq_things']) * 100:.2f} | {float(res['miou']) * 100:.2f} | "
                    f"{float(res['acc']) * 100:.2f} |\n"
                )
            f.write(
                f"| Delta | {delta_pq:+.2f} |  |  | "
                f"{(float(fused_results['pq_stuffs']) - float(baseline_results['pq_stuffs'])) * 100:+.2f} | "
                f"{delta_pqt:+.2f} | "
                f"{(float(fused_results['miou']) - float(baseline_results['miou'])) * 100:+.2f} | "
                f"{(float(fused_results['acc']) - float(baseline_results['acc'])) * 100:+.2f} |\n\n"
            )
            f.write("## Threshold Sweep\n\n")
            f.write("| tau | PQ | delta PQ | PQ_stuff | delta stuff | PQ_things | delta things | mIoU | active px | suppressed px |\n")
            f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for tau in thresholds:
                res = fused_results_by_tau[tau]
                f.write(
                    f"| {tau:.3f} | {float(res['pq']) * 100:.2f} | "
                    f"{(float(res['pq']) - float(baseline_results['pq'])) * 100:+.2f} | "
                    f"{float(res['pq_stuffs']) * 100:.2f} | "
                    f"{(float(res['pq_stuffs']) - float(baseline_results['pq_stuffs'])) * 100:+.2f} | "
                    f"{float(res['pq_things']) * 100:.2f} | "
                    f"{(float(res['pq_things']) - float(baseline_results['pq_things'])) * 100:+.2f} | "
                    f"{float(res['miou']) * 100:.2f} | {n_adapter_pixels[tau]} | "
                    f"{n_adapter_thing_pixels_suppressed[tau]} |\n"
                )
            f.write("\n")
            f.write("## Eval Counters\n\n")
            f.write(f"- Pass-1 images: {n_pass1}\n")
            f.write(f"- Pass-2 images: {n_evaluated}\n")
            f.write(f"- Missing P4 cache fallback images: {n_skipped_no_cache}\n")
            f.write(f"- Active adapter pixels by threshold: {n_adapter_pixels}\n")
            f.write(f"- Suppressed thing pixels without CUPS instances by threshold: {n_adapter_thing_pixels_suppressed}\n\n")
            f.write("## Per-Class PQ\n\n")
            f.write("\n".join(_per_class_delta_rows(baseline_results, fused_results, CITYSCAPES_CLASSNAMES)))
            f.write("\n\n## CUPS Assignment\n\n")
            f.write("```text\n")
            f.write(str(pq_assignments.tolist()))
            f.write("\n```\n")
        print(f"\n[INFO] Report saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
