import numpy as np

from mbps_pytorch.ga_uniap.config import Phase0Config
from mbps_pytorch.ga_uniap.evaluate import masks_to_panoptic, score_image


def test_oracle_labeling_assigns_majority_class():
    # one cluster fully over a 'road'(0) region, one over 'car'(13)
    gt_sem = np.zeros((512, 1024), np.uint8)
    gt_sem[:, 512:] = 13  # right half = car (thing)
    masks = np.zeros((2, 32, 64), bool)
    masks[0, :, :32] = True   # left -> road
    masks[1, :, 32:] = True   # right -> car
    pred_sem, pred_inst = masks_to_panoptic(masks, gt_sem)
    assert (pred_sem[:, :512] == 0).all()
    assert len(pred_inst) == 1 and pred_inst[0][1] == 13


def test_perfect_stuff_mask_scores_high_pq():
    cfg = Phase0Config()
    gt_sem = np.zeros((512, 1024), np.uint8)  # all road
    gt_inst = np.zeros((512, 1024), np.int32)
    masks = np.ones((1, 32, 64), bool)        # one cluster covering everything -> road
    tp, fp, fn, iou = score_image(masks, gt_sem, gt_inst, cfg)
    assert tp[0] == 1 and fp[0] == 0 and fn[0] == 0  # road matched
