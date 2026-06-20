import numpy as np

from mbps_pytorch.mobile_panoptic_sup.demo import overlay


def test_overlay_shape_and_nonempty():
    img = np.zeros((32, 32, 3), np.uint8)
    pan = np.zeros((32, 32), np.int32)
    pan[:16] = 130 * 1000        # stuff segment
    pan[16:] = 0 * 1000 + 1      # thing instance
    out = overlay(img, pan)
    assert out.shape == img.shape
    assert out.sum() > 0          # colors + text were drawn
