import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

_BUILDER_PATH = Path(__file__).resolve().parents[3] / "scripts" / "build_rare_instance_pool.py"
spec = importlib.util.spec_from_file_location("build_rare_instance_pool", _BUILDER_PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_rare_pool_builder_extracts_synthetic_instances(tmp_path):
    pseudo_dir = tmp_path / "pseudo"
    image_dir = tmp_path / "images"
    depth_dir = tmp_path / "depth"
    pseudo_dir.mkdir()
    image_dir.mkdir()
    depth_dir.mkdir()

    stem = "aachen_000000_000019_leftImg8bit"
    semantic = np.zeros((4, 4), dtype=np.uint8)
    semantic[2:, 2:] = 1
    instance = np.zeros((4, 4), dtype=np.uint16)
    instance[:2, :2] = 1
    instance[2:, 2:] = 2
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:2, :2] = [255, 0, 0]
    image[2:, 2:] = [0, 255, 0]
    depth = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)

    Image.fromarray(semantic).save(pseudo_dir / f"{stem}_semantic.png")
    Image.fromarray(instance).save(pseudo_dir / f"{stem}_instance.png")
    Image.fromarray(image).save(image_dir / f"{stem}.png")
    np.save(depth_dir / "aachen_000000_000019.npy", depth)
    centroids = tmp_path / "kmeans_centroids.npz"
    np.savez(centroids, cluster_to_class=np.array([11, 14], dtype=np.uint8))

    pool = builder.build_pool(
        pseudo_dir=pseudo_dir,
        image_dir=image_dir,
        depth_dir=depth_dir,
        centroids=centroids,
        max_per_class=10,
        workers=1,
        min_area=1,
        min_side=1,
    )

    assert len(pool[11]) == 1
    assert len(pool[14]) == 1
    assert pool[11][0].area == 4
    assert 0.0 <= pool[14][0].src_depth_quantile <= 1.0
