import numpy as np

from mbps_pytorch.mobile_panoptic_sup import to_coco_panoptic as C


def test_roundtrip_segments():
    pan = np.zeros((4, 4), np.int32)
    pan[:2] = 5 * 1000 + 1          # thing class idx 5, instance 1
    pan[2:] = 130 * 1000           # stuff class idx 130
    rgb, ann = C.pan_to_coco(pan, "x")
    seg_ids = {s["id"] for s in ann["segments_info"]}
    rgb = rgb.astype(np.int64)
    decoded = rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 65536
    assert set(np.unique(decoded)) == seg_ids
    areas = {s["id"]: s["area"] for s in ann["segments_info"]}
    assert sum(areas.values()) == 16


def test_void_excluded():
    pan = np.full((3, 3), 255 * 1000, np.int32)   # all void
    rgb, ann = C.pan_to_coco(pan, "v")
    assert ann["segments_info"] == []
    assert rgb.sum() == 0
