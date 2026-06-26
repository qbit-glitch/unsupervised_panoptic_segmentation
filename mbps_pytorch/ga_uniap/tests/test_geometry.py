import numpy as np

from mbps_pytorch.ga_uniap.config import Phase0Config
from mbps_pytorch.ga_uniap.preflight import list_val_stems


def test_preflight_finds_val_stems():
    cfg = Phase0Config(n_images=10)
    stems = list_val_stems(cfg)
    assert len(stems) >= 10, f"expected >=10 usable val stems, got {len(stems)}"
    stem, city = stems[0]
    assert city in {"frankfurt", "lindau", "munster"}
    assert isinstance(stem, str) and stem.startswith(city)
