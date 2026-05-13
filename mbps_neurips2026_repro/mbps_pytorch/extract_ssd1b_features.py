"""Extract self-attention features from SSD-1B (distilled SDXL) UNet.

Follows the same pattern as SDFeatureExtractor in diffcut_pseudo_semantics.py
but adapted for the SDXL architecture (dual text encoders, added_cond_kwargs).

Usage:
    # Probe mode (verify architecture before full extraction)
    python extract_ssd1b_features.py --coco_root /path/to/coco --device mps --probe

    # Extract all val2017 at timestep 10 (Falcon paper setting)
    nohup python -u extract_ssd1b_features.py --coco_root /path/to/coco --device mps --step 10 \
        > /tmp/ssd1b_extract.log 2>&1 &
"""

import argparse
import gc
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class SSD1BFeatureExtractor:
    """Extract self-attention features from SSD-1B UNet.

    SSD-1B is a distilled SDXL model (segmind/SSD-1B) with 3 down_blocks.
    At 1024×1024 input, the target layer produces 32×32=1024 tokens at 1280 dims.

    Key differences from SD-1.4 (SDFeatureExtractor):
    - Pipeline: StableDiffusionXLPipeline (dual text encoders)
    - Hook: down_blocks[-1] (not [-2]) for 32×32 features at 1024×1024 input
    - UNet forward requires added_cond_kwargs (text_embeds + time_ids)
    """

    def __init__(
        self,
        model_name: str = "segmind/SSD-1B",
        device: str = "mps",
    ) -> None:
        from diffusers import DDIMScheduler, StableDiffusionXLPipeline

        self.device = torch.device(device)
        dtype = torch.float32 if device == "mps" else torch.float16

        logger.info("Loading SSD-1B: %s on %s (dtype=%s)", model_name, device, dtype)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.scheduler.set_timesteps(50)
        self.pipe.enable_attention_slicing()

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self._features: Optional[torch.Tensor] = None
        logger.info("SSD-1B model loaded")

    def _register_hook(self) -> List:
        """Hook into the 32×32 self-attention block.

        SSD-1B UNet at 1024×1024 input:
          down_blocks[0] DownBlock2D:        128→64, 320ch,  NO attention
          down_blocks[1] CrossAttnDownBlock2D: 64→32, 640ch,  attention at 64×64
          down_blocks[2] CrossAttnDownBlock2D: 32→16, 1280ch, attention at 32×32 ← TARGET
        """
        handles = []
        attn_block = (
            self.unet.down_blocks[-1]
            .attentions[-1]
            .transformer_blocks[-1]
            .attn1
        )

        def hook_fn(mod, inp, out):
            self._features = out.detach()

        handles.append(attn_block.register_forward_hook(hook_fn))
        return handles

    @torch.no_grad()
    def extract(
        self,
        image: torch.Tensor,
        step: int = 10,
        img_size: int = 1024,
    ) -> torch.Tensor:
        """Extract SSD-1B self-attention features for a single image.

        Args:
            image: (1, 3, H, W) tensor in [0, 1].
            step: Diffusion timestep for noise injection (paper: 10).
            img_size: Resize image to this size before encoding (paper: 1024).

        Returns:
            Features tensor of shape (1, N, C) where N = (img_size/32)^2.
            For 1024×1024: (1, 1024, 1280).
        """
        image = F.interpolate(
            image, size=(img_size, img_size), mode="bilinear", align_corners=False
        )

        # Encode to latent
        latent = self.vae.encode(2 * image - 1).latent_dist.mean
        latent = latent * self.vae.config.scaling_factor

        # Add noise at timestep
        latent_h, latent_w = img_size // 8, img_size // 8
        rng = torch.Generator(device=self.device).manual_seed(42)
        noise = torch.randn(
            1, 4, latent_h, latent_w,
            generator=rng, device=self.device,
        )
        t = torch.tensor([step], device=self.device)
        noisy_latent = self.pipe.scheduler.add_noise(latent, noise, t)

        # SDXL text encoding (dual encoder: CLIP-L + CLIP-G)
        prompt_result = self.pipe.encode_prompt(
            prompt=[""],
            prompt_2=[""],
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = prompt_result[0]
        pooled_prompt_embeds = prompt_result[2]

        # SDXL time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        add_time_ids = torch.tensor(
            [[float(img_size), float(img_size), 0.0, 0.0,
              float(img_size), float(img_size)]],
            device=self.device,
        )
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        # Forward pass through UNet with hook
        handles = self._register_hook()
        noisy_latent = self.pipe.scheduler.scale_model_input(noisy_latent, t)

        if self.device.type == "mps":
            self.unet(
                noisy_latent, t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                self.unet(
                    noisy_latent, t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                )

        for h in handles:
            h.remove()

        return self._features


def probe_ssd1b(device: str = "mps", img_size: int = 1024, step: int = 10) -> None:
    """Single-image probe to verify architecture before full extraction."""
    logger.info("=== SSD-1B Probe Mode ===")
    logger.info("Device: %s, img_size: %d, step: %d", device, img_size, step)

    # Print UNet structure
    from diffusers import StableDiffusionXLPipeline

    logger.info("Loading model for probe...")
    dtype = torch.float32 if device == "mps" else torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "segmind/SSD-1B", torch_dtype=dtype
    ).to(device)
    pipe.enable_attention_slicing()

    unet = pipe.unet
    logger.info("UNet down_blocks: %d", len(unet.down_blocks))
    for i, block in enumerate(unet.down_blocks):
        block_type = type(block).__name__
        has_attn = hasattr(block, "attentions") and block.attentions is not None
        if has_attn:
            n_attn = len(block.attentions)
            n_tf = len(block.attentions[0].transformer_blocks)
            logger.info(
                "  down_blocks[%d]: %s, attentions=%d, transformer_blocks=%d",
                i, block_type, n_attn, n_tf,
            )
        else:
            logger.info("  down_blocks[%d]: %s, NO attention", i, block_type)

    # Hook ALL attention layers and run one image
    all_features: Dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook_fn(mod, inp, out):
            all_features[name] = out.detach()
        return hook_fn

    handles = []
    for i, block in enumerate(unet.down_blocks):
        if hasattr(block, "attentions") and block.attentions is not None:
            for j, attn in enumerate(block.attentions):
                for k, tf_block in enumerate(attn.transformer_blocks):
                    name = f"down_blocks[{i}].attentions[{j}].transformer_blocks[{k}].attn1"
                    handles.append(
                        tf_block.attn1.register_forward_hook(make_hook(name))
                    )

    # Also hook mid_block
    if hasattr(unet.mid_block, "attentions") and unet.mid_block.attentions is not None:
        for j, attn in enumerate(unet.mid_block.attentions):
            for k, tf_block in enumerate(attn.transformer_blocks):
                name = f"mid_block.attentions[{j}].transformer_blocks[{k}].attn1"
                handles.append(
                    tf_block.attn1.register_forward_hook(make_hook(name))
                )

    # Create a test image
    test_img = torch.rand(1, 3, img_size, img_size, device=device, dtype=dtype)

    # Run extraction
    import time
    logger.info("Running probe extraction...")
    t0 = time.time()

    latent = pipe.vae.encode(2 * test_img - 1).latent_dist.mean
    latent = latent * pipe.vae.config.scaling_factor

    latent_h, latent_w = img_size // 8, img_size // 8
    rng = torch.Generator(device=device).manual_seed(42)
    noise = torch.randn(1, 4, latent_h, latent_w, generator=rng, device=device)
    t_step = torch.tensor([step], device=device)
    noisy_latent = pipe.scheduler.add_noise(latent, noise, t_step)

    prompt_result = pipe.encode_prompt(
        prompt=[""], prompt_2=[""],
        device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    prompt_embeds = prompt_result[0]
    pooled_prompt_embeds = prompt_result[2]

    add_time_ids = torch.tensor(
        [[float(img_size), float(img_size), 0.0, 0.0,
          float(img_size), float(img_size)]],
        device=device,
    )

    noisy_latent = pipe.scheduler.scale_model_input(noisy_latent, t_step)
    unet(
        noisy_latent, t_step,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        },
    )

    elapsed = time.time() - t0

    for h in handles:
        h.remove()

    # Print results
    logger.info("")
    logger.info("=== Probe Results ===")
    logger.info("Extraction time: %.2fs", elapsed)
    logger.info("")
    for name, feat in sorted(all_features.items()):
        n_tokens = feat.shape[1]
        grid = int(math.sqrt(n_tokens))
        grid_ok = grid * grid == n_tokens
        logger.info(
            "  %s: shape=%s, grid=%dx%d%s, min=%.4f, max=%.4f, mean=%.4f",
            name, list(feat.shape), grid, grid,
            " ✓" if grid_ok else " (non-square!)",
            feat.min().item(), feat.max().item(), feat.mean().item(),
        )

    # Check target layer
    target_name = None
    for name, feat in all_features.items():
        if feat.shape[1] == 1024 and feat.shape[2] == 1280:
            target_name = name
            break

    if target_name:
        logger.info("")
        logger.info("✓ TARGET LAYER FOUND: %s → (1, 1024, 1280)", target_name)
    else:
        logger.warning("")
        logger.warning("✗ No layer with shape (1, 1024, 1280) found!")
        logger.warning("Check which layer gives 32×32 features and update _register_hook()")

    if device == "mps":
        mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
        logger.info("MPS memory allocated: %.1f MB", mem_mb)

    del pipe
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def extract_ssd1b_features(
    coco_root: str,
    device: str = "mps",
    img_size: int = 1024,
    step: int = 10,
    n_images: Optional[int] = None,
) -> None:
    """Extract and cache SSD-1B self-attention features for val2017."""
    img_dir = Path(coco_root) / "val2017"
    out_dir = Path(coco_root) / f"ssd1b_features_s{step}" / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg"))
    if n_images:
        img_files = img_files[:n_images]

    # Skip already extracted
    existing = {f.stem for f in out_dir.glob("*.npy")}
    todo = [f for f in img_files if f.stem not in existing]
    logger.info(
        "SSD-1B feature extraction: %d images total, %d already done, %d to extract",
        len(img_files), len(existing), len(todo),
    )

    if not todo:
        logger.info("All features already extracted. Skipping.")
        return

    extractor = SSD1BFeatureExtractor(device=device)

    for i, img_path in enumerate(tqdm(todo, desc="Extracting SSD-1B features")):
        try:
            img = Image.open(img_path).convert("RGB")
            img_t = torch.tensor(
                np.array(img).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)

            feats = extractor.extract(img_t, step=step, img_size=img_size)
            # feats: (1, N, C) — save as (N, C)
            np.save(
                out_dir / f"{img_path.stem}.npy",
                feats[0].cpu().float().numpy(),
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                logger.warning("OOM on %s, clearing cache and retrying...", img_path.stem)
                if device == "mps":
                    torch.mps.empty_cache()
                gc.collect()
                img = Image.open(img_path).convert("RGB")
                img_t = torch.tensor(
                    np.array(img).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device)
                feats = extractor.extract(img_t, step=step, img_size=img_size)
                np.save(
                    out_dir / f"{img_path.stem}.npy",
                    feats[0].cpu().float().numpy(),
                )
            else:
                raise

        # Periodic cache clearing to prevent MPS memory fragmentation
        if device == "mps" and (i + 1) % 100 == 0:
            torch.mps.empty_cache()

    logger.info("SSD-1B features saved to %s (%d files)", out_dir, len(todo))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract SSD-1B self-attention features for COCO val2017"
    )
    p.add_argument("--coco_root", required=True, help="Path to COCO dataset root")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--step", type=int, default=10,
                    help="Diffusion timestep (Falcon paper: 10)")
    p.add_argument("--img_size", type=int, default=1024,
                    help="Input resolution (Falcon paper: 1024)")
    p.add_argument("--n_images", type=int, default=None,
                    help="Limit number of images to extract")
    p.add_argument("--probe", action="store_true",
                    help="Single-image probe mode: verify architecture and print layer shapes")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.probe:
        probe_ssd1b(device=args.device, img_size=args.img_size, step=args.step)
    else:
        extract_ssd1b_features(
            coco_root=args.coco_root,
            device=args.device,
            img_size=args.img_size,
            step=args.step,
            n_images=args.n_images,
        )


if __name__ == "__main__":
    main()
