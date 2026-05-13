from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_manifest_has_required_files():
    required = [
        "README.md",
        "REPRODUCIBILITY_AUDIT.md",
        "LOCAL_ASSET_PATHS.md",
        "paper/mbps_neurips2026_mbps.tex",
        "paper/mbps_neurips2026_supplementary.tex",
        "mbps_pytorch/train_depth_adapter.py",
        "mbps_pytorch/models/semantic/depth_adapter.py",
        "mbps_pytorch/models/semantic/stego_loss.py",
        "mbps_pytorch/generate_depth_overclustered_semantics.py",
        "mbps_pytorch/convert_to_cups_format.py",
        "repro_scripts/extract_cause_codes.py",
        "repro_scripts/refine_simcf.py",
        "third_party/cups/train.py",
        "third_party/cups/train_self.py",
        "third_party/cups/val.py",
        "third_party/cups/configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml",
        "third_party/cups/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml",
        "third_party/cups/configs/val_stage3_dcfa_simcf_abc_local.yaml",
        "third_party/cause/models/dinov2vit.py",
        "third_party/cause/modules/segment.py",
        "third_party/dinov3/dinov3/hub/backbones.py",
        "paper_artifacts/results/stage3_dcfa_simcf_abc_step3000_eval.json",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert not missing


def test_local_asset_manifest_records_original_machine_paths():
    manifest = (ROOT / "LOCAL_ASSET_PATHS.md").read_text()
    required_refs = [
        "/Users/qbit-glitch/Desktop/datasets/cityscapes",
        "/Users/qbit-glitch/Desktop/datasets/cityscapes/depth_depthpro",
        "/Users/qbit-glitch/Desktop/datasets/cityscapes/cause_codes_90d",
        "/Users/qbit-glitch/Desktop/datasets/cityscapes/dinov3_features",
        "/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc",
        "/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/weights/dinov3_vitb16_official.pth",
        "/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cause/checkpoint/dinov2_vit_base_14.pth",
        "/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt",
        "/Users/qbit-glitch/Desktop/datasets/coco",
        "/Users/qbit-glitch/Desktop/datasets/kitti_panoptic",
        "/Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2",
        "/Users/qbit-glitch/Desktop/datasets/MOTSChallenge",
        "santosh@100.93.203.100",
    ]
    missing_refs = [ref for ref in required_refs if ref not in manifest]
    assert not missing_refs


def test_release_excludes_large_runtime_artifact_dirs():
    forbidden = [
        "third_party/cups/experiments",
        "third_party/cups/results",
        "third_party/cups/logs",
        "third_party/cups/.git",
        "third_party/cups/.ccr",
        "mbps_pytorch/.ccr",
    ]
    present = [path for path in forbidden if (ROOT / path).exists()]
    assert not present

    forbidden_suffixes = {".ckpt", ".pth", ".pkl"}
    copied = [
        p.relative_to(ROOT)
        for p in ROOT.rglob("*")
        if p.is_file() and p.suffix in forbidden_suffixes
    ]
    assert not copied


def test_dcfa_identity_initialization_and_parameter_count():
    from mbps_pytorch.models.semantic.depth_adapter import (
        DepthAdapter,
        sinusoidal_depth_encode,
    )

    model = DepthAdapter(code_dim=90, depth_dim=16, hidden_dim=384, num_layers=2)
    params = sum(p.numel() for p in model.parameters())
    assert params == 225_114

    codes = torch.randn(2, 8, 90)
    depth = torch.rand(2, 8)
    depth_enc = sinusoidal_depth_encode(depth)
    out = model(codes, depth_enc)
    assert out.shape == codes.shape
    assert torch.allclose(out, codes, atol=1e-7)


def test_simcf_step_a_and_b_on_toy_inputs():
    simcf = load_module("release_refine_simcf", ROOT / "repro_scripts/refine_simcf.py")

    semantic = np.zeros((32, 64), dtype=np.uint8)
    semantic[4:8, 4:8] = 1
    instance = np.ones((32, 64), dtype=np.uint16)
    cluster_to_class = np.full(256, 255, dtype=np.uint8)
    cluster_to_class[0] = 0
    cluster_to_class[1] = 1

    changed = simcf.step_a(semantic, instance, cluster_to_class, num_clusters=2)
    assert changed == 16
    assert np.all(semantic[4:8, 4:8] == 0)

    semantic = np.zeros((32, 64), dtype=np.uint8)
    instance = np.zeros((32, 64), dtype=np.uint16)
    instance[:, :32] = 1
    instance[:, 32:] = 2
    features = np.ones((32 * 64, 4), dtype=np.float32)
    merged, n_merges = simcf.step_b(
        semantic,
        instance,
        features,
        cluster_to_class,
        sim_threshold=0.5,
        dilate_px=1,
    )
    assert n_merges == 1
    assert set(np.unique(merged)) == {1}


def test_depth_cc_instance_generation_on_toy_inputs():
    convert = load_module(
        "release_convert_to_cups",
        ROOT / "mbps_pytorch/convert_to_cups_format.py",
    )
    semantic = np.full((16, 16), 3, dtype=np.uint8)
    depth = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    inst = convert.build_instance_map_depth_cc(
        semantic,
        depth,
        thing_ids={3},
        min_area=8,
        grad_threshold=10.0,
        depth_blur_sigma=0.0,
        dilation_iters=0,
    )
    assert inst.shape == semantic.shape
    assert inst.max() == 1


def test_cups_config_loader_parses_paper_configs():
    config_mod = load_module(
        "release_cups_config",
        ROOT / "third_party/cups/cups/config.py",
    )
    stage2 = config_mod.get_default_config(
        experiment_config_file=str(
            ROOT
            / "third_party/cups/configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml"
        ),
        command_line_arguments=[],
    )
    assert stage2.MODEL.BACKBONE_TYPE == "dinov3_vitb"
    assert stage2.TRAINING.STEPS == 8000
    assert "cups_pseudo_labels_dcfa_simcf_abc" in stage2.DATA.ROOT_PSEUDO

    stage3 = config_mod.get_default_config(
        experiment_config_file=str(
            ROOT
            / "third_party/cups/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml"
        ),
        command_line_arguments=[],
    )
    assert stage3.MODEL.BACKBONE_TYPE == "dinov3_vitb"
    assert stage3.SELF_TRAINING.ROUNDS == 3
    assert stage3.SELF_TRAINING.ROUND_STEPS == 4000


def test_final_cityscapes_result_artifact_matches_paper_numbers():
    import json

    path = ROOT / "paper_artifacts/results/stage3_dcfa_simcf_abc_step3000_eval.json"
    data = json.loads(path.read_text())
    metrics = data["metrics"]
    assert round(metrics["PQ"] * 100, 2) == 35.83
    assert round(metrics["PQ_things"] * 100, 2) == 36.26
    assert round(metrics["PQ_stuff"] * 100, 2) == 35.56
    assert round(metrics["mIoU"] * 100, 2) == 44.56
