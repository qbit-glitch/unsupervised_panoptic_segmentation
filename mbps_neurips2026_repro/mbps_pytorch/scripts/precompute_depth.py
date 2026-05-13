"""Pre-compute Depth Maps with ZoeDepth.

Usage:
    python scripts/precompute_depth.py \
        --data_dir /path/to/cityscapes \
        --output_dir /path/to/depth_cache \
        --dataset cityscapes

Pre-computes monocular depth estimates using ZoeDepth and saves
them as numpy files for efficient training-time loading.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def precompute_depth_zoedepth(
    data_dir: str,
    output_dir: str,
    dataset_name: str,
    image_size: tuple = (512, 512),
    batch_size: int = 4,
):
    """Pre-compute depth maps using ZoeDepth.

    Args:
        data_dir: Path to dataset images.
        output_dir: Path to save depth maps.
        dataset_name: Dataset name (cityscapes, coco_stuff27).
        image_size: Processing resolution.
        batch_size: Batch size for ZoeDepth inference.
    """
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        logger.warning(
            "PyTorch not available. Generating random depth maps "
            "for placeholder. Install torch for real depth estimation."
        )

    os.makedirs(output_dir, exist_ok=True)

    # Collect image paths
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(Path(data_dir).rglob(ext))
    image_paths = sorted(image_paths)
    logger.info(f"Found {len(image_paths)} images in {data_dir}")

    if HAS_TORCH:
        # Load ZoeDepth model with compatibility patches for PyTorch 2.10+
        logger.info("Loading ZoeDepth model...")

        # Patch 1: PyTorch 2.10+ saves relative_position_index buffers in
        # Swin checkpoints that ZoeDepth's loader doesn't expect.
        _orig_load = torch.nn.Module.load_state_dict
        torch.nn.Module.load_state_dict = (
            lambda self, state_dict, strict=True, **kw:
            _orig_load(self, state_dict, strict=False, **kw)
        )

        # Patch 2: Newer timm renamed Block.drop_path to drop_path1/drop_path2.
        # MiDaS's BEiT backbone code still references the old name.
        # Patch the cached MiDaS beit.py file directly (most reliable approach).
        _midas_beit = os.path.expanduser(
            "~/.cache/torch/hub/intel-isl_MiDaS_master/midas/backbones/beit.py"
        )
        if os.path.exists(_midas_beit):
            with open(_midas_beit, "r") as f:
                _beit_src = f.read()
            if "self.drop_path(" in _beit_src and "self.drop_path1(" not in _beit_src:
                _beit_src = _beit_src.replace("self.drop_path(", "self.drop_path1(")
                with open(_midas_beit, "w") as f:
                    f.write(_beit_src)
                logger.info("Patched MiDaS beit.py: drop_path -> drop_path1")
                # Clear cached bytecode so the patched source is re-compiled
                _beit_pyc = _midas_beit + "c"
                if os.path.exists(_beit_pyc):
                    os.remove(_beit_pyc)
                import importlib
                _pycache = os.path.join(os.path.dirname(_midas_beit), "__pycache__")
                if os.path.exists(_pycache):
                    import shutil
                    shutil.rmtree(_pycache, ignore_errors=True)

        try:
            model = torch.hub.load(
                "isl-org/ZoeDepth",
                "ZoeD_N",
                pretrained=True,
            )
        finally:
            torch.nn.Module.load_state_dict = _orig_load

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        logger.info("ZoeDepth model loaded successfully.")

        from PIL import Image
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor

        transform = Compose([
            Resize(image_size),
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    for idx, img_path in enumerate(image_paths):
        # Output path mirrors input structure
        rel_path = img_path.relative_to(data_dir)
        depth_path = Path(output_dir) / rel_path.with_suffix(".npy")
        depth_path.parent.mkdir(parents=True, exist_ok=True)

        if depth_path.exists():
            continue

        if HAS_TORCH:
            from PIL import Image

            img = Image.open(img_path).convert("RGB")
            inp = transform(img).unsqueeze(0)
            if torch.cuda.is_available():
                inp = inp.cuda()

            with torch.no_grad():
                depth = model.infer(inp)

            depth_np = depth.squeeze().cpu().numpy()
        else:
            # Placeholder: generate structured random depth
            np.random.seed(idx)
            h, w = image_size
            # Create somewhat realistic depth with gradient
            y_grad = np.linspace(0.3, 1.0, h)[:, None]
            noise = np.random.rand(h, w) * 0.2
            depth_np = (y_grad * np.ones((1, w)) + noise).astype(np.float32)

        np.save(str(depth_path), depth_np)

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(image_paths)} images")

    logger.info(f"Depth maps saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Depth Maps")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cityscapes")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    precompute_depth_zoedepth(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
