#!/usr/bin/env python3
"""Export per-image qualitative comparisons for Ours vs. CUPS.

The supplementary notebook contains the same workflow for interactive use. This
script is the batch runner: it samples images from Cityscapes, KITTI, Mapillary,
and Waymo, runs both panoptic models, and saves one folder per image:

    original.png | ours_prediction.png | ours_overlay.png
    cups_prediction.png | cups_overlay.png | triplet.png | comparison.png
    ours_raw_panoptic.npz | cups_raw_panoptic.npz
    ours_raw_thing_instance_ids.png | cups_raw_thing_instance_ids.png

Use the local CUPS/Detectron2 environment:

    /Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \
        scripts/generate_qualitative_ours_vs_cups.py --datasets cityscapes kitti mapillary waymo
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation")
CUPS_ROOT = PROJECT_ROOT / "refs" / "cups"
DATASETS_ROOT = Path("/Users/qbit-glitch/Desktop/datasets")
OUTPUT_ROOT = PROJECT_ROOT / "notebooks" / "qualitative_results"

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

for path in (PROJECT_ROOT, CUPS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.io  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402


DEFAULT_OURS_CKPT = PROJECT_ROOT / "checkpoints/stage4_pathB_focal_w005_gated_v2/best_pq_step=000575.ckpt"
DEFAULT_CUPS_CKPT = PROJECT_ROOT / "weights/cups.ckpt"
DEFAULT_OURS_CFG = CUPS_ROOT / "configs/val_stage4_fine_object_local.yaml"
FALLBACK_OURS_CFG = CUPS_ROOT / "configs/train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml"
MODEL_SIZE_DIVISOR = 64
NUM_CITYSCAPES_27_CLASSES = 27

# Official semantic assignment shipped in refs/cups/demo.py for weights/cups.ckpt.
# It maps CUPS discovered cluster IDs to Cityscapes-27 class IDs for visualization.
OFFICIAL_CUPS_CITYSCAPES27_ASSIGNMENT: tuple[int, ...] = (
    7, 4, 2, 4, 2, 6, 2, 0, 5, 8, 0, 2, 9, 10, 3, 8, 1, 2, 0, 0, 0, 0, 11, 13, 18, 15, 14
)


@dataclass(frozen=True)
class Sample:
    dataset: str
    path: Path
    image_id: str


@dataclass
class Models:
    ours: torch.nn.Module
    cups: torch.nn.Module
    ours_thing_classes: tuple[int, ...]
    ours_stuff_classes: tuple[int, ...]
    cups_thing_classes: tuple[int, ...]
    cups_stuff_classes: tuple[int, ...]


@dataclass
class SemanticMappings:
    ours: torch.Tensor | None
    cups: torch.Tensor | None
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cityscapes", "kitti", "mapillary", "waymo"],
        choices=["cityscapes", "kitti", "mapillary", "waymo"],
        help="Datasets to export.",
    )
    parser.add_argument("--n-images", type=int, default=20, help="Images per dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument(
        "--max-short-side",
        type=int,
        default=720,
        help="Downscale images whose short side is larger than this value.",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--datasets-root", type=Path, default=DATASETS_ROOT)
    parser.add_argument("--ours-ckpt", type=Path, default=DEFAULT_OURS_CKPT)
    parser.add_argument("--cups-ckpt", type=Path, default=DEFAULT_CUPS_CKPT)
    parser.add_argument("--ours-cfg", type=Path, default=DEFAULT_OURS_CFG)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, or mps. Defaults to CUDA if available.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument(
        "--draw-instance-ids",
        action="store_true",
        help="Draw instance ID text on overlays. Disabled by default for cleaner paper figures.",
    )
    parser.add_argument(
        "--semantic-mapping",
        choices=["cityscapes27", "raw"],
        default="cityscapes27",
        help=(
            "Default cityscapes27 applies the Hungarian assignment used by the PQ metric so both models "
            "are rendered in the same 27-class color space. Use raw only for inspecting individual cluster "
            "behavior; raw is unsafe for cross-model comparison because the cityscapes palette has only 28 "
            "entries and Ours has 80 pseudo-clusters (IDs >=27 collapse to black)."
        ),
    )
    parser.add_argument(
        "--assignment-cache",
        type=Path,
        default=None,
        help="JSON cache for semantic assignments. Defaults inside the output root.",
    )
    parser.add_argument(
        "--assignment-images",
        type=int,
        default=0,
        help="Cityscapes val images used to compute our Hungarian assignment; 0 uses all 500.",
    )
    parser.add_argument(
        "--compute-cups-assignment",
        action="store_true",
        help="Compute CUPS assignment from Cityscapes val instead of using the official demo mapping.",
    )
    parser.add_argument(
        "--fill-void",
        choices=["on", "off"],
        default="on",
        help=(
            "Fill semantic void (combine-logic abstention) with the spatially nearest non-void pixel "
            "before rendering PNGs. Visualization-only; raw_panoptic.npz preserves the original void "
            "mask and the PQ pipeline is unaffected. Default: on."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute folders that already contain triplet.png.")
    parser.add_argument("--dry-run", action="store_true", help="Print sampled images without loading models.")
    return parser.parse_args()


def select_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def safe_image_id(image_id: str) -> str:
    keep = (" ", ".", "_", "-")
    value = "".join(c for c in image_id if c.isalnum() or c in keep).strip().replace(" ", "_")
    return value or "image"


def stable_seed(*parts: str, base: int = 42) -> int:
    value = base
    for part in parts:
        for ch in part:
            value = (value * 33 + ord(ch)) % (2**31 - 1)
    return value


def sample_cityscapes(root: Path, n: int, seed: int) -> list[Sample]:
    val_root = root / "cityscapes/leftImg8bit/val"
    paths = sorted(val_root.glob("*/*_leftImg8bit.png"))
    selected = random.Random(seed).sample(paths, min(n, len(paths)))
    return [Sample("cityscapes", p, p.stem.replace("_leftImg8bit", "")) for p in selected]


def sample_kitti(root: Path, n: int, seed: int) -> list[Sample]:
    val_root = root / "kitti_panoptic/validation/images"
    paths = sorted(val_root.glob("*.png"))
    selected = random.Random(seed).sample(paths, min(n, len(paths)))
    return [Sample("kitti", p, p.stem) for p in selected]


def sample_mapillary(root: Path, n: int, seed: int) -> list[Sample]:
    val_root = root / "mapillary-vistas-v2/validation/images"
    paths = sorted(list(val_root.glob("*.jpg")) + list(val_root.glob("*.png")))
    selected = random.Random(seed).sample(paths, min(n, len(paths)))
    return [Sample("mapillary", p, p.stem) for p in selected]


def _is_waymo_rgb(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return False
    blocked = ("_instance", "_semantic", "_label", "_panoptic", "_depth", "_mask")
    if any(token in name for token in blocked):
        return False
    return "_image" in name or "image" in path.parent.name.lower() or path.suffix.lower() in {".jpg", ".jpeg"}


def sample_waymo(root: Path, n: int, seed: int) -> list[Sample]:
    candidates = [
        Path("/Volumes/code_files/datasets/panoptic_segmentation_datasets/waymo_v2_0_1_preprocessed/validation"),
        root / "waymo_v2_0_1_preprocessed/validation",
        root / "waymo/validation",
        root / "waymo_v2/validation",
        root / "waymo",
        root / "waymo_v2",
        PROJECT_ROOT / ".gcloud_waymo/images",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        paths = sorted(p for p in candidate.rglob("*") if p.is_file() and _is_waymo_rgb(p))
        if not paths:
            continue
        selected = random.Random(seed).sample(paths, min(n, len(paths)))
        samples = []
        for path in selected:
            image_id = f"{path.parent.name}_{path.stem}" if path.parent != candidate else path.stem
            samples.append(Sample("waymo", path, image_id))
        print(f"[INFO] Waymo source: {candidate} ({len(paths)} RGB images)")
        return samples
    print("[WARN] Waymo RGB images not found; skipping Waymo.")
    return []


def sample_dataset(dataset: str, root: Path, n: int, seed: int) -> list[Sample]:
    samplers = {
        "cityscapes": sample_cityscapes,
        "kitti": sample_kitti,
        "mapillary": sample_mapillary,
        "waymo": sample_waymo,
    }
    return samplers[dataset](root, n, seed)


def load_image_tensor(path: Path, max_short_side: int) -> torch.Tensor:
    image = torchvision.io.read_image(str(path)).float() / 255.0
    if image.shape[0] == 4:
        image = image[:3]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    h, w = image.shape[-2:]
    short_side = min(h, w)
    new_h, new_w = h, w
    if short_side > max_short_side:
        scale = max_short_side / short_side
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
    # The DINOv3+FPN path needs stronger alignment than the 16px patch grid:
    # odd FPN stages can otherwise disagree by 4px at overlay time.
    new_h = max(MODEL_SIZE_DIVISOR, (new_h // MODEL_SIZE_DIVISOR) * MODEL_SIZE_DIVISOR)
    new_w = max(MODEL_SIZE_DIVISOR, (new_w // MODEL_SIZE_DIVISOR) * MODEL_SIZE_DIVISOR)
    if (new_h, new_w) != (h, w):
        image = TF.resize(image, [new_h, new_w], antialias=True)
    return image


def tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    return (image.cpu().clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()


def _stable_color(value: int, seed: int) -> tuple[int, int, int]:
    hashed = (value * 1103515245 + seed * 12345 + 0x9E3779B9) & 0xFFFFFFFF
    return (
        40 + ((hashed >> 16) & 0xBF),
        40 + ((hashed >> 8) & 0xBF),
        40 + (hashed & 0xBF),
    )


def colorize_id_map(ids: torch.Tensor | np.ndarray, seed: int, zero_is_black: bool = True) -> np.ndarray:
    id_array = ids.cpu().numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids)
    id_array = id_array.astype(np.int64, copy=False)
    rgb = np.zeros((*id_array.shape, 3), dtype=np.uint8)
    for value in np.unique(id_array):
        int_value = int(value)
        if zero_is_black and int_value == 0:
            continue
        rgb[id_array == int_value] = _stable_color(int_value, seed)
    return rgb


def save_uint16_id_png(ids: torch.Tensor | np.ndarray, out_path: Path) -> None:
    id_array = ids.cpu().numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids)
    if id_array.size == 0:
        id_array = np.zeros((1, 1), dtype=np.uint16)
    if id_array.min() < 0 or id_array.max() > np.iinfo(np.uint16).max:
        np.save(out_path.with_suffix(".npy"), id_array.astype(np.int32, copy=False))
        return
    Image.fromarray(id_array.astype(np.uint16, copy=False)).save(out_path)


def panoptic_summary(panoptic: torch.Tensor, thing_classes: tuple[int, ...]) -> dict[str, object]:
    semantic = panoptic[..., 0].cpu()
    instance = panoptic[..., 1].cpu()
    thing_mask = instance > 0
    semantic_ids = sorted(int(x) for x in torch.unique(semantic).tolist())
    instance_ids = sorted(int(x) for x in torch.unique(instance[thing_mask]).tolist()) if thing_mask.any() else []
    thing_semantic_ids = (
        sorted(int(x) for x in torch.unique(semantic[thing_mask]).tolist()) if thing_mask.any() else []
    )
    return {
        "semantic_ids": semantic_ids,
        "thing_semantic_ids": thing_semantic_ids,
        "thing_instance_ids": instance_ids,
        "num_thing_instances": len(instance_ids),
        "thing_pixel_count": int(thing_mask.sum().item()),
        "thing_classes_config": [int(x) for x in thing_classes],
    }


def save_raw_panoptic_artifacts(
    panoptic: torch.Tensor,
    thing_classes: tuple[int, ...],
    stuff_classes: tuple[int, ...],
    out_dir: Path,
    prefix: str,
    seed: int,
) -> dict[str, object]:
    semantic = panoptic[..., 0].cpu().to(torch.int32)
    instance = panoptic[..., 1].cpu().to(torch.int32)
    thing_mask = instance > 0
    thing_instance_ids = torch.where(thing_mask, instance, torch.zeros_like(instance))

    np.savez_compressed(
        out_dir / f"{prefix}_raw_panoptic.npz",
        semantic_id=semantic.numpy(),
        instance_id=instance.numpy(),
        thing_mask=thing_mask.numpy(),
        thing_classes=np.asarray(thing_classes, dtype=np.int32),
        stuff_classes=np.asarray(stuff_classes, dtype=np.int32),
    )
    save_uint16_id_png(semantic, out_dir / f"{prefix}_raw_semantic_ids.png")
    save_uint16_id_png(thing_instance_ids, out_dir / f"{prefix}_raw_thing_instance_ids.png")
    Image.fromarray(colorize_id_map(semantic, seed=seed, zero_is_black=False)).save(
        out_dir / f"{prefix}_raw_semantic_ids_color.png"
    )
    Image.fromarray(colorize_id_map(thing_instance_ids, seed=seed, zero_is_black=True)).save(
        out_dir / f"{prefix}_raw_thing_instance_ids_color.png"
    )
    Image.fromarray((thing_mask.numpy().astype(np.uint8) * 255)).save(out_dir / f"{prefix}_raw_thing_mask.png")
    return panoptic_summary(panoptic, thing_classes)


def load_models(args: argparse.Namespace, device: str) -> Models:
    require_file(args.ours_ckpt, "our checkpoint")
    require_file(args.cups_ckpt, "CUPS checkpoint")
    ours_cfg = args.ours_cfg if args.ours_cfg.exists() else FALLBACK_OURS_CFG
    require_file(ours_cfg, "our CUPS config")

    from cups.model import panoptic_cascade_mask_r_cnn_from_checkpoint

    cups_model, cups_n_things, cups_n_stuff = panoptic_cascade_mask_r_cnn_from_checkpoint(
        path=str(args.cups_ckpt),
        device=device,
        confidence_threshold=args.confidence_threshold,
    )
    cups_model = cups_model.to(device).eval()
    cups_stuff_classes = tuple(range(cups_n_stuff))
    cups_thing_classes = tuple(range(cups_n_stuff, cups_n_stuff + cups_n_things))
    print(f"[INFO] CUPS loaded: {cups_n_things} things + {cups_n_stuff} stuff clusters")

    import cups
    from cups.augmentation import PhotometricAugmentations, ResolutionJitter
    from cups.data import CITYSCAPES_CLASSNAMES, CITYSCAPES_STUFF_CLASSES, CITYSCAPES_THING_CLASSES
    from pytorch_lightning import seed_everything

    config = cups.get_default_config(experiment_config_file=str(ours_cfg))
    config.defrost()
    config.MODEL.CHECKPOINT = str(args.ours_ckpt)
    config.SYSTEM.ACCELERATOR = device if not device.startswith("cuda") else "gpu"
    config.freeze()
    seed_everything(config.SYSTEM.SEED, verbose=False)

    ours_model = cups.build_model_self(
        config=config,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        class_names=CITYSCAPES_CLASSNAMES,
        photometric_augmentation=PhotometricAugmentations(),
        freeze_bn=True,
        resolution_jitter_augmentation=ResolutionJitter(scales=None, resolutions=config.AUGMENTATION.RESOLUTIONS),
    )
    ours_model = ours_model.to(device).eval()
    ours_thing_classes = tuple(int(x) for x in ours_model.hparams.thing_pseudo_classes)
    ours_stuff_classes = tuple(int(x) for x in ours_model.hparams.stuff_pseudo_classes)
    print(f"[INFO] Ours loaded: {len(ours_thing_classes)} things + {len(ours_stuff_classes)} stuff pseudo-classes")

    return Models(
        ours=ours_model,
        cups=cups_model,
        ours_thing_classes=ours_thing_classes,
        ours_stuff_classes=ours_stuff_classes,
        cups_thing_classes=cups_thing_classes,
        cups_stuff_classes=cups_stuff_classes,
    )


@torch.no_grad()
def predict_panoptic(
    model: torch.nn.Module,
    image: torch.Tensor,
    thing_classes: tuple[int, ...],
    stuff_classes: tuple[int, ...],
    device: str,
) -> torch.Tensor:
    from cups.model import prediction_to_standard_format

    # Match refs/cups/evaluate_cityscapes.py: RGB tensor in [0, 1], CHW.
    model_image = image.contiguous().to(device)
    prediction = model([{"image": model_image}])[0]["panoptic_seg"]
    standard = prediction_to_standard_format(
        prediction,
        stuff_classes=stuff_classes,
        thing_classes=thing_classes,
    )
    return standard.cpu()


def _mapping_cache_path(args: argparse.Namespace) -> Path:
    if args.assignment_cache is not None:
        return args.assignment_cache
    limit_tag = "all" if args.assignment_images <= 0 else str(args.assignment_images)
    ckpt_tag = f"{args.ours_ckpt.parent.name}_{args.ours_ckpt.stem}".replace("=", "")
    return args.output_root / f"_semantic_assignments_cityscapes27_{ckpt_tag}_{limit_tag}.json"


def _valid_cached_mapping(data: dict, models: Models, args: argparse.Namespace) -> bool:
    ours = data.get("ours")
    cups = data.get("cups")
    return (
        data.get("semantic_mapping") == "cityscapes27"
        and data.get("ours_ckpt") == str(args.ours_ckpt)
        and data.get("cups_ckpt") == str(args.cups_ckpt)
        and isinstance(ours, list)
        and isinstance(cups, list)
        and len(ours) == len(models.ours_stuff_classes) + len(models.ours_thing_classes)
        and len(cups) == len(models.cups_stuff_classes) + len(models.cups_thing_classes)
    )


def load_semantic_mappings(args: argparse.Namespace, models: Models, device: str) -> SemanticMappings:
    if args.semantic_mapping == "raw":
        return SemanticMappings(ours=None, cups=None, source="raw")

    cache_path = _mapping_cache_path(args)
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if _valid_cached_mapping(data, models, args):
            print(f"[INFO] Loaded Cityscapes-27 semantic assignments: {cache_path}")
            return SemanticMappings(
                ours=torch.tensor(data["ours"], dtype=torch.long),
                cups=torch.tensor(data["cups"], dtype=torch.long),
                source=f"cache:{cache_path}",
            )
        print(f"[WARN] Ignoring stale semantic assignment cache: {cache_path}")

    print("[INFO] Computing our Cityscapes-27 semantic assignment with the official CUPS protocol...")
    ours = compute_cityscapes27_assignment(
        label="ours",
        model=models.ours,
        thing_classes=models.ours_thing_classes,
        stuff_classes=models.ours_stuff_classes,
        args=args,
        device=device,
    )

    cups_num_clusters = len(models.cups_stuff_classes) + len(models.cups_thing_classes)
    if (
        not args.compute_cups_assignment
        and args.cups_ckpt.name == "cups.ckpt"
        and cups_num_clusters == len(OFFICIAL_CUPS_CITYSCAPES27_ASSIGNMENT)
    ):
        print("[INFO] Using official CUPS Cityscapes-27 assignment from refs/cups/demo.py")
        cups = torch.tensor(OFFICIAL_CUPS_CITYSCAPES27_ASSIGNMENT, dtype=torch.long)
        cups_source = "refs/cups/demo.py"
    else:
        print("[INFO] Computing CUPS Cityscapes-27 semantic assignment with the official CUPS protocol...")
        cups = compute_cityscapes27_assignment(
            label="cups",
            model=models.cups,
            thing_classes=models.cups_thing_classes,
            stuff_classes=models.cups_stuff_classes,
            args=args,
            device=device,
        )
        cups_source = "computed"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "semantic_mapping": "cityscapes27",
        "ours_ckpt": str(args.ours_ckpt),
        "cups_ckpt": str(args.cups_ckpt),
        "ours_cfg": str(args.ours_cfg),
        "assignment_images": args.assignment_images,
        "ours": [int(x) for x in ours.tolist()],
        "cups": [int(x) for x in cups.tolist()],
        "cups_source": cups_source,
    }
    cache_path.write_text(json.dumps(cache_payload, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Saved Cityscapes-27 semantic assignments: {cache_path}")
    return SemanticMappings(ours=ours, cups=cups, source=f"computed:{cache_path}")


@torch.no_grad()
def compute_cityscapes27_assignment(
    label: str,
    model: torch.nn.Module,
    thing_classes: tuple[int, ...],
    stuff_classes: tuple[int, ...],
    args: argparse.Namespace,
    device: str,
) -> torch.Tensor:
    from torch.utils.data import DataLoader

    from cups.data import (
        CITYSCAPES_STUFF_CLASSES,
        CITYSCAPES_THING_CLASSES,
        CityscapesPanopticValidation,
        collate_function_validation,
    )
    from cups.metrics.panoptic_quality import PanopticQualitySemanticMatching
    from cups.model import prediction_to_standard_format

    cityscapes_root = args.datasets_root / "cityscapes"
    validation_dataset = CityscapesPanopticValidation(
        root=str(cityscapes_root),
        crop_resolution=(640, 1280),
        num_classes=NUM_CITYSCAPES_27_CLASSES,
        resize_scale=0.625,
    )
    if args.assignment_images > 0:
        validation_dataset.images = validation_dataset.images[: args.assignment_images]
        validation_dataset.labels = validation_dataset.labels[: args.assignment_images]

    loader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_function_validation,
        drop_last=False,
    )
    num_clusters = len(stuff_classes) + len(thing_classes)
    pq_helper = PanopticQualitySemanticMatching(
        things=CITYSCAPES_THING_CLASSES,
        stuffs=CITYSCAPES_STUFF_CLASSES,
        num_clusters=num_clusters,
        things_prototype=set(thing_classes),
        stuffs_prototype=set(stuff_classes),
        cache_device="cpu",
        sync_on_compute=False,
        dist_sync_on_step=False,
    )
    cost_matrix = torch.zeros(num_clusters, NUM_CITYSCAPES_27_CLASSES, dtype=torch.float32)

    for batch in tqdm(loader, desc=f"{label} assignment"):
        images, panoptic_labels, _image_names = batch
        images_dev = [{"image": item["image"].to(device)} for item in images]
        prediction = model(images_dev)
        panoptic_pred = prediction_to_standard_format(
            prediction[0]["panoptic_seg"],
            stuff_classes=stuff_classes,
            thing_classes=thing_classes,
        )
        cost_matrix += pq_helper._cost_matrix_update(
            panoptic_pred[..., 0].reshape(-1).cpu(),
            panoptic_labels[..., 0].reshape(-1).cpu(),
            NUM_CITYSCAPES_27_CLASSES,
            num_clusters,
        ).cpu()
        del images_dev, prediction, panoptic_pred
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    pq_helper.cost_matrix = cost_matrix.to(pq_helper.cost_matrix.device)
    assignments = pq_helper.matching().cpu().long()
    print(f"[INFO] {label} assignment: {assignments.tolist()}")
    return assignments


def apply_semantic_mapping(panoptic: torch.Tensor, assignment: torch.Tensor | None) -> torch.Tensor:
    if assignment is None:
        return panoptic
    from cups.metrics import PanopticQualitySemanticMatching

    return PanopticQualitySemanticMatching.map_to_target(panoptic, assignment).cpu()


def fill_panoptic_void(panoptic: torch.Tensor, void_id: int = 255) -> torch.Tensor:
    """Fill semantic-channel void pixels with the spatially nearest non-void pixel.

    PanopticFPN's combine logic drops sub-threshold instances and small stuff
    segments to void; the underlying model still classified those pixels but the
    panoptic head abstained. For visualization we fill those gaps via spatial
    nearest-neighbor on the post-Hungarian semantic channel and copy the
    corresponding instance id when the source pixel is a thing. This is a
    visualization-only post-process: the raw `*_raw_panoptic.npz` artifact
    preserves the original void mask, and the PQ evaluation pipeline never
    reads this code path.
    """
    from scipy.ndimage import distance_transform_edt

    semantic = panoptic[..., 0].cpu().numpy()
    instance = panoptic[..., 1].cpu().numpy()

    void_mask = (semantic == void_id)
    if not void_mask.any() or void_mask.all():
        return panoptic

    _, indices = distance_transform_edt(void_mask, return_indices=True)
    nearest_semantic = semantic[tuple(indices)]
    nearest_instance = instance[tuple(indices)]

    out_semantic = np.where(void_mask, nearest_semantic, semantic)
    # Copy instance id only when the nearest neighbor was a thing pixel; stuff
    # filled regions stay at instance=0 so the random per-instance color shift
    # is not applied to fabricated extents.
    out_instance = np.where(void_mask & (nearest_instance > 0), nearest_instance, instance)

    return torch.stack(
        [torch.from_numpy(out_semantic), torch.from_numpy(out_instance)],
        dim=-1,
    ).to(panoptic.dtype)


def render_prediction(
    panoptic: torch.Tensor,
    seed: int,
    instance_blend: float = 0.45,
) -> np.ndarray:
    """Render a panoptic prediction with per-instance distinct colors.

    Each thing instance is assigned a color from a categorical palette (tab20,
    shuffled by seed for diversity within an image) and blended with the
    underlying semantic-class color so semantic identity is still visible.
    Stuff pixels (instance_id == 0) keep the pure semantic-class color.

    Replaces the original CUPS scalar brightness shift, which made adjacent
    same-class instances visually indistinguishable when the per-image
    instance count exceeds ~3 of the same class.
    """
    from cups.visualization import semantic_segmentation_to_rgb
    from matplotlib import colormaps

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    semantic_segmentation = panoptic[..., 0].cpu()
    instance_segmentation = panoptic[..., 1].cpu()
    semantic_rgb = semantic_segmentation_to_rgb(semantic_segmentation, dataset="cityscapes").float()
    max_instance = int(instance_segmentation.max().item()) if instance_segmentation.numel() else 0

    if max_instance == 0:
        return semantic_rgb.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()

    # Categorical palette of 20 visually distinct hues, shuffled per-image so
    # adjacent instance ids do not collide with neighboring tab20 colors.
    base_palette = (np.asarray(colormaps["tab20"].colors) * 255).astype(np.float32)
    shuffled = base_palette[rng.permutation(base_palette.shape[0])]
    inst_colors = np.zeros((max_instance + 1, 3), dtype=np.float32)
    inst_colors[1:] = shuffled[np.arange(max_instance) % shuffled.shape[0]]

    instance_np = instance_segmentation.numpy()
    inst_color_hwc = inst_colors[instance_np]
    inst_color_chw = torch.from_numpy(inst_color_hwc).permute(2, 0, 1)

    thing_mask = (instance_segmentation > 0).unsqueeze(0).float()
    keep_semantic = 1.0 - instance_blend * thing_mask
    use_instance = instance_blend * thing_mask
    blended = semantic_rgb * keep_semantic + inst_color_chw * use_instance

    return blended.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()


def render_overlay(
    panoptic: torch.Tensor,
    image: torch.Tensor,
    alpha: float,
    seed: int,
    draw_instance_ids: bool = False,
) -> np.ndarray:
    from cups.visualization import panoptic_segmentation_overlay_to_rgb

    torch.manual_seed(seed)
    if draw_instance_ids:
        overlay = panoptic_segmentation_overlay_to_rgb(
            panoptic.cpu(),
            image.cpu(),
            alpha=alpha,
            dataset="cityscapes",
            denormalize=False,
        )
    else:
        segmentation_rgb = torch.from_numpy(render_prediction(panoptic, seed)).permute(2, 0, 1).float()
        overlay = alpha * segmentation_rgb + (1.0 - alpha) * (image.cpu() * 255.0)
    if overlay.dtype != torch.uint8:
        overlay = overlay.clamp(0, 255).to(torch.uint8)
    return overlay.permute(1, 2, 0).numpy()


def save_panel_grid(
    panels: list[tuple[np.ndarray, str]],
    out_path: Path,
) -> None:
    images = [Image.fromarray(image).convert("RGB") for image, _ in panels]
    widths, heights = zip(*(image.size for image in images))
    gap = 6
    label_h = 44
    canvas_w = sum(widths) + gap * (len(images) - 1)
    canvas_h = max(heights) + label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_triplet_font(size=24)

    x = 0
    for image, (_, title) in zip(images, panels):
        text_box = draw.textbbox((0, 0), title, font=font)
        text_w = text_box[2] - text_box[0]
        draw.text((x + (image.width - text_w) / 2, 10), title, fill=(20, 20, 20), font=font)
        canvas.paste(image, (x, label_h))
        x += image.width + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_triplet(
    rgb: np.ndarray,
    ours_overlay: np.ndarray,
    cups_overlay: np.ndarray,
    out_path: Path,
) -> None:
    panels: list[tuple[np.ndarray, str]] = [
        (rgb, "Original Image"),
        (ours_overlay, "Ours Inference + Overlay"),
        (cups_overlay, "CUPS Inference + Overlay"),
    ]
    save_panel_grid(panels, out_path)


def save_comparison(
    rgb: np.ndarray,
    ours_prediction: np.ndarray,
    ours_overlay: np.ndarray,
    cups_prediction: np.ndarray,
    cups_overlay: np.ndarray,
    out_path: Path,
) -> None:
    panels: list[tuple[np.ndarray, str]] = [
        (rgb, "Original Image"),
        (ours_prediction, "Ours Raw Prediction"),
        (ours_overlay, "Ours Overlay"),
        (cups_prediction, "CUPS Raw Prediction"),
        (cups_overlay, "CUPS Overlay"),
    ]
    save_panel_grid(panels, out_path)


def _load_triplet_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def process_sample(
    sample: Sample,
    models: Models,
    mappings: SemanticMappings,
    args: argparse.Namespace,
    device: str,
) -> dict[str, str]:
    out_dir = args.output_root / sample.dataset / safe_image_id(sample.image_id)
    triplet_path = out_dir / "triplet.png"
    required_outputs = (
        triplet_path,
        out_dir / "comparison.png",
        out_dir / "ours_raw_panoptic.npz",
        out_dir / "cups_raw_panoptic.npz",
        out_dir / "ours_raw_thing_instance_ids.png",
        out_dir / "cups_raw_thing_instance_ids.png",
    )
    if all(path.exists() for path in required_outputs) and not args.overwrite:
        return {
            "dataset": sample.dataset,
            "image_id": sample.image_id,
            "source_path": str(sample.path),
            "output_dir": str(out_dir),
            "semantic_mapping": args.semantic_mapping,
            "ours_num_thing_instances": "",
            "cups_num_thing_instances": "",
            "status": "skipped_existing",
        }

    image = load_image_tensor(sample.path, args.max_short_side)
    rgb = tensor_to_uint8_hwc(image)

    ours_raw_panoptic = predict_panoptic(
        models.ours,
        image,
        models.ours_thing_classes,
        models.ours_stuff_classes,
        device,
    )
    cups_raw_panoptic = predict_panoptic(
        models.cups,
        image,
        models.cups_thing_classes,
        models.cups_stuff_classes,
        device,
    )
    ours_panoptic = apply_semantic_mapping(ours_raw_panoptic, mappings.ours)
    cups_panoptic = apply_semantic_mapping(cups_raw_panoptic, mappings.cups)

    # Visualization-only void fill: applied AFTER raw artifacts are dumped, so
    # the npz/uint16 PNG outputs preserve the original combine-logic abstention
    # mask. Eval pipeline is unaffected (different code path).
    if args.fill_void == "on":
        ours_panoptic_for_render = fill_panoptic_void(ours_panoptic)
        cups_panoptic_for_render = fill_panoptic_void(cups_panoptic)
    else:
        ours_panoptic_for_render = ours_panoptic
        cups_panoptic_for_render = cups_panoptic

    ours_prediction = render_prediction(
        ours_panoptic_for_render,
        seed=stable_seed(sample.dataset, sample.image_id, "ours", base=args.seed),
    )
    cups_prediction = render_prediction(
        cups_panoptic_for_render,
        seed=stable_seed(sample.dataset, sample.image_id, "cups", base=args.seed),
    )
    ours_overlay = render_overlay(
        ours_panoptic_for_render,
        image,
        alpha=args.overlay_alpha,
        seed=stable_seed(sample.dataset, sample.image_id, "ours", base=args.seed),
        draw_instance_ids=args.draw_instance_ids,
    )
    cups_overlay = render_overlay(
        cups_panoptic_for_render,
        image,
        alpha=args.overlay_alpha,
        seed=stable_seed(sample.dataset, sample.image_id, "cups", base=args.seed),
        draw_instance_ids=args.draw_instance_ids,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_dir / "original.png")
    ours_summary = save_raw_panoptic_artifacts(
        ours_raw_panoptic,
        thing_classes=models.ours_thing_classes,
        stuff_classes=models.ours_stuff_classes,
        out_dir=out_dir,
        prefix="ours",
        seed=stable_seed(sample.dataset, sample.image_id, "ours_raw_ids", base=args.seed),
    )
    cups_summary = save_raw_panoptic_artifacts(
        cups_raw_panoptic,
        thing_classes=models.cups_thing_classes,
        stuff_classes=models.cups_stuff_classes,
        out_dir=out_dir,
        prefix="cups",
        seed=stable_seed(sample.dataset, sample.image_id, "cups_raw_ids", base=args.seed),
    )
    panoptic_metadata = {
        "dataset": sample.dataset,
        "image_id": sample.image_id,
        "source_path": str(sample.path),
        "semantic_mapping_for_overlay": args.semantic_mapping,
        "ours_checkpoint": str(args.ours_ckpt),
        "ours_config": str(args.ours_cfg),
        "cups_checkpoint": str(args.cups_ckpt),
        "standard_format": {
            "channel_0": "raw semantic pseudo-class id",
            "channel_1": "raw thing instance id; 0 means stuff/background/no thing instance",
        },
        "void_fill_for_visualization": args.fill_void,
        "ours": ours_summary,
        "cups": cups_summary,
    }
    (out_dir / "panoptic_metadata.json").write_text(
        json.dumps(panoptic_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    Image.fromarray(ours_prediction).save(out_dir / "ours_prediction.png")
    Image.fromarray(ours_overlay).save(out_dir / "ours_overlay.png")
    Image.fromarray(cups_prediction).save(out_dir / "cups_prediction.png")
    Image.fromarray(cups_overlay).save(out_dir / "cups_overlay.png")
    save_triplet(rgb, ours_overlay, cups_overlay, triplet_path)
    save_comparison(rgb, ours_prediction, ours_overlay, cups_prediction, cups_overlay, out_dir / "comparison.png")
    (out_dir / "source_path.txt").write_text(str(sample.path) + "\n", encoding="utf-8")

    del (
        image,
        rgb,
        ours_raw_panoptic,
        cups_raw_panoptic,
        ours_panoptic,
        cups_panoptic,
        ours_prediction,
        cups_prediction,
        ours_overlay,
        cups_overlay,
    )
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "dataset": sample.dataset,
        "image_id": sample.image_id,
        "source_path": str(sample.path),
        "output_dir": str(out_dir),
        "semantic_mapping": args.semantic_mapping,
        "ours_num_thing_instances": str(ours_summary["num_thing_instances"]),
        "cups_num_thing_instances": str(cups_summary["num_thing_instances"]),
        "status": "ok",
    }


def write_manifest(output_root: Path, rows: Iterable[dict[str, str]]) -> None:
    rows = list(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "image_id",
                "source_path",
                "output_dir",
                "semantic_mapping",
                "ours_num_thing_instances",
                "cups_num_thing_instances",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output root: {args.output_root}")

    samples: list[Sample] = []
    for dataset in args.datasets:
        dataset_samples = sample_dataset(dataset, args.datasets_root, args.n_images, args.seed)
        print(f"[INFO] {dataset}: selected {len(dataset_samples)} images")
        for sample in dataset_samples[:3]:
            print(f"       {sample.image_id}: {sample.path}")
        samples.extend(dataset_samples)

    if args.dry_run:
        write_manifest(
            args.output_root,
            [
                {
                    "dataset": sample.dataset,
                    "image_id": sample.image_id,
                    "source_path": str(sample.path),
                    "output_dir": str(args.output_root / sample.dataset / safe_image_id(sample.image_id)),
                    "semantic_mapping": args.semantic_mapping,
                    "ours_num_thing_instances": "",
                    "cups_num_thing_instances": "",
                    "status": "dry_run",
                }
                for sample in samples
            ],
        )
        return 0

    models = load_models(args, device)
    mappings = load_semantic_mappings(args, models, device)
    rows: list[dict[str, str]] = []
    for sample in tqdm(samples, desc="Qualitative comparisons"):
        try:
            rows.append(process_sample(sample, models, mappings, args, device))
        except Exception as exc:  # keep the batch running so one bad image does not kill the export
            print(f"[ERROR] {sample.dataset}/{sample.image_id}: {type(exc).__name__}: {exc}")
            rows.append(
                {
                    "dataset": sample.dataset,
                    "image_id": sample.image_id,
                    "source_path": str(sample.path),
                    "output_dir": str(args.output_root / sample.dataset / safe_image_id(sample.image_id)),
                    "semantic_mapping": args.semantic_mapping,
                    "ours_num_thing_instances": "",
                    "cups_num_thing_instances": "",
                    "status": f"error:{type(exc).__name__}:{exc}",
                }
            )

    write_manifest(args.output_root, rows)
    ok_count = sum(row["status"] == "ok" for row in rows)
    skipped_count = sum(row["status"] == "skipped_existing" for row in rows)
    error_count = sum(row["status"].startswith("error:") for row in rows)
    print(f"[DONE] ok={ok_count}, skipped={skipped_count}, errors={error_count}")
    print(f"[DONE] manifest: {args.output_root / 'manifest.csv'}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
