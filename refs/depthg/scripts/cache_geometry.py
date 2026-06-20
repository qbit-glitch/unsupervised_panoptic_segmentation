"""Precompute GA-DepthG geometry (4-ch [ĥ, n]) from DepthPro depth + Cityscapes camera intrinsics.

Reads  <data_root>/depth_depthpro/<split>/<city>/<stem>.npy   (normalized inverse depth, full FOV)
       <data_root>/camera/<split>/<city>/<stem>_camera.json     (intrinsics, scaled to depth res)
Writes <data_root>/geometry/<split>/<city>/<stem>.npy          (4,H,W) float32 [ĥ, nx, ny, nz]

Ground-plane fit is on the FULL image (compute_geometry), so geometry is spatially consistent with
the depth FOV; the loss interpolates it to code size. Run on fics-lab in the gadepthg env.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geometry_features import compute_geometry  # noqa: E402

FULL_W, FULL_H = 2048, 1024  # Cityscapes intrinsics reference resolution


def load_intrinsics(cam_json: str, H: int, W: int):
    """fx,fy,u0,v0 scaled to the depth (H,W) + camera height; Cityscapes fallbacks if json absent."""
    fx, fy, u0, v0, cam_h = 2262.52, 2265.30, 1096.98, 513.14, 1.22
    try:
        j = json.load(open(cam_json))
        I = j["intrinsic"]
        fx, fy, u0, v0 = I["fx"], I["fy"], I["u0"], I["v0"]
        cam_h = float(j["extrinsic"]["z"])
    except Exception:
        pass
    sx, sy = W / FULL_W, H / FULL_H
    return fx * sx, fy * sy, u0 * sx, v0 * sy, cam_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/mnt/HDD_16TB/datasets/Cityscapes")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.data_root}/depth_depthpro/{a.split}/*/*.npy"))
    if a.limit:
        files = files[:a.limit]
    done = skipped = 0
    for f in files:
        stem, city = Path(f).stem, Path(f).parent.name
        out = Path(a.data_root) / "geometry" / a.split / city / f"{stem}.npy"
        if out.exists():
            skipped += 1
            continue
        depth = np.load(f).astype(np.float32)
        H, W = depth.shape
        fx, fy, u0, v0, cam_h = load_intrinsics(
            f"{a.data_root}/camera/{a.split}/{city}/{stem}_camera.json", H, W)
        g = compute_geometry(depth, fx, fy, u0, v0, cam_h)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, g)
        done += 1
        if done % 200 == 0:
            print(f"  {done} written ({skipped} skipped) / {len(files)}", flush=True)
    print(f"DONE {a.split}: {done} written, {skipped} skipped -> {a.data_root}/geometry/{a.split}")


if __name__ == "__main__":
    main()
