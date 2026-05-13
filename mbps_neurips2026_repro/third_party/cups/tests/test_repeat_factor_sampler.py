import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

_SAMPLER_PATH = Path(__file__).resolve().parents[1] / "cups" / "data" / "repeat_factor_sampler.py"
spec = importlib.util.spec_from_file_location("repeat_factor_sampler", _SAMPLER_PATH)
sampler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler_module)
RepeatFactorTrainingSampler = sampler_module.RepeatFactorTrainingSampler


class ToyDataset:
    def __init__(self, semantic_paths):
        self.semantic_paths = semantic_paths

    def __len__(self):
        return len(self.semantic_paths)


def _write_semantic(path, classes):
    arr = np.array(classes, dtype=np.uint8).reshape(1, -1)
    Image.fromarray(arr).save(path)


def test_repeat_factor_sampler_computes_freq_and_oversamples_rare_images(tmp_path):
    paths = []
    for idx, classes in enumerate(([0], [0], [0, 1], [1])):
        path = tmp_path / f"{idx}_semantic.png"
        _write_semantic(path, classes)
        paths.append(str(path))

    sampler = RepeatFactorTrainingSampler(ToyDataset(paths), threshold_t=1.0, num_samples=400, seed=3)

    assert sampler.class_freq == {0: 0.75, 1: 0.5}
    assert sampler.repeat_factors[2] > sampler.repeat_factors[0]

    indices = list(iter(sampler))
    rare_hits = sum(1 for i in indices if i in (2, 3))
    common_only_hits = sum(1 for i in indices if i in (0, 1))
    assert rare_hits > common_only_hits


def test_repeat_factor_sampler_ddp_sharding_has_no_overlap_for_single_epoch(tmp_path):
    paths = []
    for idx in range(4):
        path = tmp_path / f"{idx}_semantic.png"
        _write_semantic(path, [idx])
        paths.append(str(path))

    shards = []
    for rank in range(4):
        sampler = RepeatFactorTrainingSampler(
            ToyDataset(paths),
            threshold_t=0.001,
            num_samples=4,
            seed=7,
            num_replicas=4,
            rank=rank,
        )
        shards.append(list(iter(sampler)))

    flat = [idx for shard in shards for idx in shard]
    assert len(flat) == 4
    assert len(set(flat)) == 4
