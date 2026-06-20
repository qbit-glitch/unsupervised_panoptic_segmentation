import numpy as np

from mbps_pytorch.mobile_panoptic_sup import coco_eval


def test_contiguous_maps_have_133_classes():
    catid2idx, idx2name, things, stuff = coco_eval.coco_contiguous_maps()
    assert len(catid2idx) == 133
    assert len(things) == 80 and len(stuff) == 53
    assert max(catid2idx.values()) == 132 and min(catid2idx.values()) == 0
    assert len(idx2name) == 133


def test_gt_as_pred_scores_perfect():
    # Feeding GT as its own prediction must yield PQ ~100 on one val image.
    seg_map, seg2cat = coco_eval.load_coco_gt(139)  # 000000000139
    res = coco_eval.eval_pq([(seg_map, seg2cat)], [(seg_map, seg2cat)])
    assert res["pq"] > 99.0
