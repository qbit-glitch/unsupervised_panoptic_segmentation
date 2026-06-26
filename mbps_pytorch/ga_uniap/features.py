import functools

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from mbps_pytorch.ga_uniap.config import Phase0Config

_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


@functools.lru_cache(maxsize=1)
def _load_model(device: str):
    proc = AutoImageProcessor.from_pretrained(_MODEL_NAME)
    proc.size = {"height": 512, "width": 1024}
    proc.crop_size = {"height": 512, "width": 1024}
    proc.do_center_crop = False
    model = AutoModel.from_pretrained(_MODEL_NAME).to(device).eval()
    return proc, model


def extract_grid_features(stem: str, city: str, cfg: Phase0Config) -> np.ndarray:
    """DINOv3 ViT-B/16 patch features -> (grid_h, grid_w, 768) float32.

    512x1024 input with patch 16 -> 32x64 patch grid, matching the merge grid.
    """
    proc, model = _load_model(cfg.device)
    img_path = (cfg.data_root / "leftImg8bit" / cfg.split / city
                / f"{stem}_leftImg8bit.png")
    img = Image.open(img_path).convert("RGB")
    inp = proc(images=img, return_tensors="pt").to(cfg.device)
    with torch.no_grad():
        out = model(**inp).last_hidden_state  # (1, 1+R+P, 768)
    n_reg = getattr(model.config, "num_register_tokens", 4)
    patch = out[:, 1 + n_reg:, :]  # (1, P, 768)
    gh = cfg.work_h // 16  # 32
    gw = cfg.work_w // 16  # 64
    assert patch.shape[1] == gh * gw, f"got {patch.shape[1]} patches, want {gh*gw}"
    grid = patch.reshape(gh, gw, 768).float().cpu().numpy()
    assert (gh, gw) == (cfg.grid_h, cfg.grid_w), "backbone grid != cfg grid"
    return grid.astype(np.float32)
