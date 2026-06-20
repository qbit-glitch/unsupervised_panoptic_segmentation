import importlib

import numpy as np


def test_build_pred_panoptic_encoding():
    mod = importlib.import_module("mbps_pytorch.eval_cups_protocol")
    sem = np.array([[5, 5, 20, 20],
                    [5, 5, 20, 20],
                    [5, 5, 0, 0],
                    [5, 5, 0, 0]], dtype=np.int32)
    pred = mod.build_pred_panoptic(sem, thing_clusters={20}, min_area=1)
    assert pred.shape == (4, 4, 2)
    assert set(pred[sem == 5][:, 1].tolist()) == {0}      # stuff -> instance 0
    assert pred[sem == 20][:, 1].max() >= 1               # thing -> >=1 instance id
