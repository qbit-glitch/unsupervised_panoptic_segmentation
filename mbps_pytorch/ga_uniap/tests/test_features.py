import numpy as np

from mbps_pytorch.ga_uniap.config import Phase0Config
from mbps_pytorch.ga_uniap.preflight import list_val_stems
from mbps_pytorch.ga_uniap.features import extract_grid_features


def test_extract_grid_features_shape():
    cfg = Phase0Config(n_images=3)
    stem, city = list_val_stems(cfg)[0]
    feats = extract_grid_features(stem, city, cfg)
    assert feats.shape == (32, 64, 768)
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats))
