#!/usr/bin/env python3
"""Quick smoke test for SAM3 setup — validates token, model loading, and one forward pass.

Usage:
    python scripts/test_sam3_setup.py --hf_token hf_XXXX [--device mps]
"""
import argparse
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--hf_model_id", default="facebook/sam3")
    parser.add_argument("--device", default="mps", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--cityscapes_root", default="/Users/qbit-glitch/Desktop/datasets/cityscapes",
                        help="Path to Cityscapes (for loading a real test image)")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.error("No HuggingFace token found. Pass --hf_token or set HF_TOKEN.")
        sys.exit(1)

    # ── 1. Check transformers version ──────────────────────────────────────
    import transformers
    logger.info("transformers version: %s", transformers.__version__)
    if not hasattr(transformers, "Sam3Model"):
        logger.error("Sam3Model not found in transformers — need transformers>=5.1.0")
        sys.exit(1)
    logger.info("[OK] Sam3Model available in transformers")

    # ── 2. Load model from HuggingFace Hub ────────────────────────────────
    import torch
    from transformers import Sam3Model, Sam3Processor

    logger.info("Loading Sam3Processor from %s ...", args.hf_model_id)
    t0 = time.time()
    try:
        processor = Sam3Processor.from_pretrained(args.hf_model_id, token=token)
    except Exception as e:
        logger.error("Failed to load Sam3Processor: %s", e)
        sys.exit(1)
    logger.info("[OK] Processor loaded in %.1fs", time.time() - t0)

    logger.info("Loading Sam3Model from %s ...", args.hf_model_id)
    t0 = time.time()
    try:
        model = Sam3Model.from_pretrained(
            args.hf_model_id, token=token, torch_dtype=torch.float32
        )
    except Exception as e:
        logger.error("Failed to load Sam3Model: %s", e)
        sys.exit(1)
    logger.info("[OK] Model loaded in %.1fs", time.time() - t0)

    # ── 3. Move to device ─────────────────────────────────────────────────
    logger.info("Moving model to %s ...", args.device)
    try:
        model = model.to(args.device).eval()
    except Exception as e:
        logger.warning("Failed to move to %s, falling back to CPU: %s", args.device, e)
        args.device = "cpu"
        model = model.to("cpu").eval()
    logger.info("[OK] Model on %s", args.device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("SAM3 parameters: %.1fM", n_params)

    # ── 4. Load a test image ──────────────────────────────────────────────
    from pathlib import Path
    import numpy as np
    from PIL import Image

    # Try to find a Cityscapes val image, fall back to a tiny synthetic image
    test_img = None
    cs_root = Path(args.cityscapes_root)
    if cs_root.exists():
        imgs = sorted((cs_root / "leftImg8bit" / "val").rglob("*_leftImg8bit.png"))
        if imgs:
            test_img = np.array(Image.open(imgs[0]).convert("RGB"))
            logger.info("Using Cityscapes image: %s", imgs[0].name)

    if test_img is None:
        logger.info("Cityscapes not found — using 512×1024 synthetic test image")
        test_img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)

    # Resize for speed
    h, w = test_img.shape[:2]
    if h > 512:
        new_w = int(w * 512 / h)
        test_img = np.array(Image.fromarray(test_img).resize((new_w, 512), Image.BILINEAR))
    logger.info("Test image size: %s", test_img.shape)

    # ── 5. Forward pass: pre-compute vision embeddings ────────────────────
    pil_image = Image.fromarray(test_img)
    img_inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = img_inputs["pixel_values"].to(args.device)

    logger.info("Running vision encoder ...")
    t0 = time.time()
    with torch.no_grad():
        vision_embeds = model.get_vision_features(pixel_values=pixel_values)
    logger.info("[OK] Vision encoder in %.2fs", time.time() - t0)

    # ── 6. Text-prompted segmentation for one class ───────────────────────
    test_class = "bicycle"
    logger.info("Running text-prompted segmentation for '%s' ...", test_class)
    text_inputs = processor(text=test_class, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(args.device)
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(
            vision_embeds=vision_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    dt = time.time() - t0

    pred_masks = outputs.pred_masks[0]    # (N_q, H, W)
    pred_logits = outputs.pred_logits     # (1, N_q)
    presence = outputs.presence_logits   # (1, 1)

    n_queries = pred_masks.shape[0]
    if pred_logits is not None and presence is not None:
        scores = (pred_logits[0].sigmoid() * presence[0].sigmoid()).cpu().float().numpy()
    else:
        scores = pred_logits[0].sigmoid().cpu().float().numpy() if pred_logits is not None \
                 else [0.5] * n_queries

    n_above_thresh = int((scores > 0.5).sum())
    logger.info("[OK] Segmentation forward in %.2fs | queries=%d | above-0.5: %d",
                dt, n_queries, n_above_thresh)
    logger.info("Score range: min=%.3f max=%.3f mean=%.3f",
                float(scores.min()), float(scores.max()), float(scores.mean()))

    logger.info("")
    logger.info("=== SAM3 SETUP VERIFIED ===")
    logger.info("You can now run the full mask generation:")
    logger.info("")
    logger.info("  python -u scripts/generate_sam_fine_masks.py \\")
    logger.info("      --cityscapes_root %s \\", args.cityscapes_root)
    logger.info("      --split val \\")
    logger.info("      --backend sam3 \\")
    logger.info("      --hf_token %s \\", token[:8] + "...")
    logger.info("      --device %s \\", args.device)
    logger.info("      --resize_height 512 \\")
    logger.info("      --max_images 15")
    logger.info("")
    logger.info("Then visualize:")
    logger.info("")
    logger.info("  python scripts/visualize_sam_fine_masks.py \\")
    logger.info("      --cityscapes_root %s \\", args.cityscapes_root)
    logger.info("      --masks_dir %s/sam_fine_masks_sam3/val \\",
                args.cityscapes_root)
    logger.info("      --split val --n_images 15 --out_dir /tmp/sam3_vis")


if __name__ == "__main__":
    main()
