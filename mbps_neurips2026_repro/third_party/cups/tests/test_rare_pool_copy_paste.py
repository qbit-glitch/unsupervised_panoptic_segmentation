import io
import os
import pickle
import sys

import numpy as np
import pytest
import torch
from PIL import Image

detectron2 = pytest.importorskip("detectron2")
from detectron2.structures import BitMasks, Boxes, Instances

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cups.augmentation_rare_pool import InstanceCrop, RareInstancePoolCopyPaste


def _encode_image(color):
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[:, :] = color
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _encode_mask():
    arr = np.ones((3, 3), dtype=np.uint8) * 255
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def test_rare_pool_copy_paste_adds_pooled_instance(tmp_path):
    pool = {
        11: [
            InstanceCrop(
                train_id=11,
                image_jpeg=_encode_image([255, 0, 0]),
                mask_png=_encode_mask(),
                src_depth_quantile=0.5,
                bbox=(0, 0, 3, 3),
                area=9,
            )
        ]
    }
    pool_path = tmp_path / "pool.pkl"
    with pool_path.open("wb") as f:
        pickle.dump(pool, f, protocol=4)

    instances = Instances(
        image_size=(8, 8),
        gt_masks=BitMasks(torch.zeros(0, 8, 8, dtype=torch.bool)),
        gt_boxes=Boxes(torch.zeros(0, 4)),
        gt_classes=torch.zeros(0, dtype=torch.long),
    )
    sample = {
        "image": torch.zeros(3, 8, 8),
        "sem_seg": torch.ones(8, 8, dtype=torch.long),
        "instances": instances,
        "depth": torch.ones(1, 8, 8) * 0.5,
    }
    aug = RareInstancePoolCopyPaste(
        thing_class=1,
        pool_path=str(pool_path),
        scale_range=(1.0, 1.0),
        pastes_per_image=(1, 1),
        use_depth_placement=True,
        thing_id_to_trainid=(11,),
    )

    out = aug([], [sample])[0]

    assert out["instances"].gt_masks.tensor.shape[0] == 1
    assert out["instances"].gt_classes.tolist() == [0]
    assert (out["sem_seg"] == 0).sum().item() == 9
    assert out["image"].sum().item() > 0
