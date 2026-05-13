"""
Evaluate official CAUSE-TR with DINOv2 ViT-B/14 on Cityscapes.

MPS-compatible adaptation of test_tr.py.
Reproduces published results: mIoU=29.9%, pAcc=89.8%.

Usage:
    python refs/cause/eval_cause_tr_dinov2.py \
        --data_dir /Users/qbit-glitch/Desktop/datasets \
        --device mps
"""

import argparse
import os
import sys
import math
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# Add CAUSE repo to path so we can import its modules
CAUSE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CAUSE_ROOT)

from modules.segment_module import transform, untransform
from modules.segment import Segment_TR
from modules.segment_module import Cluster
from loader.dataloader import ContrastiveSegDataset
from utils.utils import (
    invTrans,
    create_cityscapes_colormap,
    ckpt_to_name,
    ckpt_to_arch,
    freeze,
    print_argparse,
)

# ---------------------------------------------------------------------------
# Device-aware NiceTool (original hardcodes .cuda())
# ---------------------------------------------------------------------------

class NiceTool:
    """Hungarian-matching mIoU/pAcc evaluator — works on any device."""

    def __init__(self, n_classes: int, device: torch.device) -> None:
        self.n_classes = n_classes
        self.device = device
        self.histogram = torch.zeros(
            (n_classes, n_classes), dtype=torch.long, device=device
        )

    def scores(
        self, label_trues: torch.Tensor, label_preds: torch.Tensor
    ) -> torch.Tensor:
        mask = (
            (label_trues >= 0)
            & (label_trues < self.n_classes)
            & (label_preds >= 0)
            & (label_preds < self.n_classes)
        )
        hist = torch.bincount(
            self.n_classes * label_trues[mask] + label_preds[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes).t().to(self.device)
        return hist

    def eval(
        self, pred: torch.Tensor, label: torch.Tensor
    ) -> tuple:
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        self.histogram += self.scores(label, pred)

        self.assignments = linear_sum_assignment(
            self.histogram.cpu().numpy(), maximize=True
        )
        hist = self.histogram[np.argsort(self.assignments[1]), :]

        tp = torch.diag(hist).float()
        fp = torch.sum(hist, dim=0).float() - tp
        fn = torch.sum(hist, dim=1).float() - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(hist).float()

        metric_dict = OrderedDict(
            {
                "mIoU": iou[~torch.isnan(iou)].mean().item() * 100,
                "mAP": prc[~torch.isnan(prc)].mean().item() * 100,
                "Acc": opc.item() * 100,
            }
        )

        sentence = ""
        for key, value in metric_dict.items():
            sentence += f"[{key}]: {value:.1f}, "
        return metric_dict, sentence

    def reset(self) -> None:
        self.histogram = torch.zeros(
            (self.n_classes, self.n_classes), dtype=torch.long, device=self.device
        )


# ---------------------------------------------------------------------------
# CRF helpers (always runs on CPU)
# ---------------------------------------------------------------------------

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as crf_utils
import torchvision.transforms.functional as VF


def dense_crf(
    image_tensor: torch.Tensor,
    output_logits: torch.Tensor,
    max_iter: int = 10,
) -> np.ndarray:
    POS_W, POS_XY_STD = 3, 1
    Bi_W, Bi_XY_STD, Bi_RGB_STD = 4, 67, 3

    image = np.array(VF.to_pil_image(invTrans(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(
        output_logits.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()
    c, h, w = output_probs.shape

    U = crf_utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(max_iter)
    return np.array(Q).reshape((c, h, w))


def _apply_crf(tup: tuple, max_iter: int = 10) -> np.ndarray:
    return dense_crf(tup[0], tup[1], max_iter=max_iter)


def do_crf(
    pool: Pool,
    img_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    max_iter: int = 10,
) -> torch.Tensor:
    outputs = pool.map(
        partial(_apply_crf, max_iter=max_iter),
        zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()),
    )
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)


# ---------------------------------------------------------------------------
# Model loading (device-aware)
# ---------------------------------------------------------------------------


def load_backbone(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load DINOv2 backbone from checkpoint."""
    name = ckpt_to_name(ckpt_path)
    arch = ckpt_to_arch(ckpt_path)

    if name == "dinov2":
        import models.dinov2vit as model_module
    elif name == "dino" or name == "mae":
        import models.dinomaevit as model_module
    else:
        raise ValueError(f"Unknown backbone: {name}")

    net = getattr(model_module, arch)()
    checkpoint = torch.load(ckpt_path, map_location=device)
    if name == "mae":
        msg = net.load_state_dict(checkpoint["model"], strict=False)
    else:
        msg = net.load_state_dict(checkpoint, strict=False)
    print(f"[Backbone] {ckpt_path} loaded: {msg}")

    net = net.to(device)
    freeze(net)
    return net


def load_segment_tr(args, device: torch.device) -> Segment_TR:
    """Load Segment_TR with pretrained weights."""
    segment = Segment_TR(args).to(device)
    baseline = args.ckpt.split("/")[-1].split(".")[0]
    ckpt_path = os.path.join(
        CAUSE_ROOT,
        "CAUSE",
        args.dataset,
        baseline,
        str(args.num_codebook),
        "segment_tr.pth",
    )
    state = torch.load(ckpt_path, map_location=device)
    msg = segment.load_state_dict(state, strict=False)
    print(f"[Segment] {ckpt_path} loaded: {msg}")
    return segment


def load_cluster_tr(args, device: torch.device) -> Cluster:
    """Load Cluster with pretrained weights + codebook."""
    cluster = Cluster(args).to(device)
    baseline = args.ckpt.split("/")[-1].split(".")[0]
    ckpt_path = os.path.join(
        CAUSE_ROOT,
        "CAUSE",
        args.dataset,
        baseline,
        str(args.num_codebook),
        "cluster_tr.pth",
    )
    state = torch.load(ckpt_path, map_location=device)
    msg = cluster.load_state_dict(state, strict=False)
    print(f"[Cluster] {ckpt_path} loaded: {msg}")

    # Load codebook
    codebook_path = os.path.join(
        CAUSE_ROOT,
        "CAUSE",
        args.dataset,
        "modularity",
        baseline,
        str(args.num_codebook),
        "modular.npy",
    )
    codebook = np.load(codebook_path)
    cb = torch.from_numpy(codebook).to(device)
    cluster.codebook.data = cb
    cluster.codebook.requires_grad = False
    # Also set codebook on segment head decoders
    print(f"[Codebook] {codebook_path} loaded: shape={cb.shape}")
    return cluster, cb


# ---------------------------------------------------------------------------
# Eval without CRF
# ---------------------------------------------------------------------------


def test_without_crf(
    net: torch.nn.Module,
    segment: Segment_TR,
    cluster: Cluster,
    nice: NiceTool,
    test_loader,
    device: torch.device,
) -> None:
    segment.eval()
    print("\n=== Eval WITHOUT CRF ===")
    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    for _, batch in prog_bar:
        img = batch["img"].to(device)
        label = batch["label"].to(device)

        with torch.no_grad():
            feat = net(img)[:, 1:, :]
            seg_feat_ema = segment.head_ema(feat)

            # Interpolate to label resolution
            interp_seg_feat = F.interpolate(
                transform(seg_feat_ema),
                label.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            cluster_preds = cluster.forward_centroid(
                untransform(interp_seg_feat), inference=True
            )

            _, desc = nice.eval(cluster_preds, label)

        prog_bar.set_description(f"[no-CRF] {desc}", refresh=True)

    nice.reset()


# ---------------------------------------------------------------------------
# Eval with CRF + flip augmentation (official protocol)
# ---------------------------------------------------------------------------


def test_with_crf(
    net: torch.nn.Module,
    segment: Segment_TR,
    cluster: Cluster,
    nice: NiceTool,
    test_loader,
    device: torch.device,
    num_pool_workers: int = 4,
) -> None:
    segment.eval()
    print("\n=== Eval WITH CRF + flip augmentation ===")
    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

    with Pool(num_pool_workers) as pool:
        for _, batch in prog_bar:
            img = batch["img"].to(device)
            label = batch["label"].to(device)

            with torch.no_grad():
                feat = net(img)[:, 1:, :]
                feat_flip = net(img.flip(dims=[3]))[:, 1:, :]

            seg_feat = transform(segment.head_ema(feat))
            seg_feat_flip = transform(segment.head_ema(feat_flip))
            seg_feat = untransform(
                (seg_feat + seg_feat_flip.flip(dims=[3])) / 2
            )

            # Interpolate to label resolution
            interp_seg_feat = F.interpolate(
                transform(seg_feat),
                label.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            # Cluster log-softmax for CRF
            cluster_preds = cluster.forward_centroid(
                untransform(interp_seg_feat), crf=True
            )

            # CRF post-processing (on CPU)
            crf_preds = do_crf(pool, img, cluster_preds).argmax(1).to(device)

            _, desc = nice.eval(crf_preds, label)
            prog_bar.set_description(f"[CRF] {desc}", refresh=True)

    nice.reset()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="CAUSE-TR DINOv2 Eval")
    parser.add_argument(
        "--data_dir",
        default="/Users/qbit-glitch/Desktop/datasets",
        type=str,
    )
    parser.add_argument("--dataset", default="cityscapes", type=str)
    parser.add_argument(
        "--ckpt",
        default="checkpoint/dinov2_vit_base_14.pth",
        type=str,
    )
    parser.add_argument("--device", default="mps", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--num_codebook", default=2048, type=int)
    parser.add_argument("--reduced_dim", default=90, type=int)
    parser.add_argument("--projection_dim", default=2048, type=int)
    parser.add_argument("--num_pool_workers", default=4, type=int)
    parser.add_argument("--skip_crf", action="store_true", help="Skip CRF eval")
    parser.add_argument(
        "--retrain_dir",
        default=None,
        type=str,
        help="Path to retrained checkpoint dir (e.g. CAUSE_dinov2_retrain/epoch_040)",
    )

    args = parser.parse_args()

    # Auto-detect dim and resolution from checkpoint name
    if "dinov2" in args.ckpt:
        args.train_resolution = 322
        args.test_resolution = 322
    else:
        args.train_resolution = 320
        args.test_resolution = 320

    if "small" in args.ckpt:
        args.dim = 384
    elif "base" in args.ckpt:
        args.dim = 768
    elif "large" in args.ckpt:
        args.dim = 1024

    patch_size = int(args.ckpt.split("_")[-1].split(".")[0])
    args.num_queries = args.train_resolution ** 2 // patch_size ** 2

    # Cityscapes = 27 classes
    if args.dataset == "cityscapes":
        args.n_classes = 27
    elif args.dataset == "cocostuff27":
        args.n_classes = 27
    elif args.dataset == "pascalvoc":
        args.n_classes = 21

    # For dataloader compatibility
    args.distributed = False
    args.gpu = "0"
    args.load_segment = False
    args.load_cluster = False

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Resolution: {args.train_resolution}, Patches: {args.num_queries}")
    print(f"Dim: {args.dim}, Codebook: {args.num_codebook}")

    # Resolve retrain_dir before chdir
    if args.retrain_dir:
        args.retrain_dir = os.path.abspath(args.retrain_dir)

    # Change to CAUSE root so relative paths work
    original_cwd = os.getcwd()
    os.chdir(CAUSE_ROOT)

    # Load models
    net = load_backbone(args.ckpt, device)

    if args.retrain_dir:
        # Load from retrained checkpoint directory
        retrain_dir = args.retrain_dir
        segment = Segment_TR(args).to(device)
        seg_path = os.path.join(retrain_dir, "segment_tr.pth")
        state = torch.load(seg_path, map_location=device)
        msg = segment.load_state_dict(state, strict=False)
        print(f"[Segment] {seg_path} loaded: {msg}")

        cluster = Cluster(args).to(device)
        cl_path = os.path.join(retrain_dir, "cluster_tr.pth")
        state = torch.load(cl_path, map_location=device)
        msg = cluster.load_state_dict(state, strict=False)
        print(f"[Cluster] {cl_path} loaded: {msg}")

        # Load retrained codebook
        cb_path = os.path.join(os.path.dirname(retrain_dir), "modular.npy")
        codebook = np.load(cb_path)
        cb = torch.from_numpy(codebook).to(device)
        cluster.codebook.data = cb
        cluster.codebook.requires_grad = False
        print(f"[Codebook] {cb_path} loaded: shape={cb.shape}")
    else:
        segment = load_segment_tr(args, device)
        cluster, cb = load_cluster_tr(args, device)

    # Set codebook on segment head decoders
    segment.head.codebook = cb
    segment.head_ema.codebook = cb

    # Load test data only (skip train — needs cropped data we don't have yet)
    from utils.utils import get_cococity_transform, get_pascal_transform

    if args.dataset in ("cityscapes", "cocostuff27"):
        get_transform = get_cococity_transform
    elif args.dataset == "pascalvoc":
        get_transform = get_pascal_transform
    else:
        get_transform = get_cococity_transform

    test_dataset = ContrastiveSegDataset(
        pytorch_data_dir=args.data_dir,
        dataset_name=args.dataset,
        crop_type=None,
        image_set="val",
        transform=get_transform(args.test_resolution, False),
        target_transform=get_transform(args.test_resolution, True),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    print(f"Val set: {len(test_dataset)} images")

    # Evaluation
    nice = NiceTool(args.n_classes, device)

    # Eval without CRF
    test_without_crf(net, segment, cluster, nice, test_loader, device)

    # Eval with CRF + flip
    if not args.skip_crf:
        test_with_crf(
            net,
            segment,
            cluster,
            nice,
            test_loader,
            device,
            num_pool_workers=args.num_pool_workers,
        )

    os.chdir(original_cwd)
    print("\nDone.")


if __name__ == "__main__":
    main()
