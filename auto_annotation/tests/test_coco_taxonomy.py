from auto_annotation import taxonomy_coco as T


def test_coco_taxonomy_133():
    assert len(T.COCO_CLASSES) == 133
    assert len(T.THING_IDXS) == 80 and len(T.STUFF_IDXS) == 53
    assert T.is_thing(T.name_to_idx("person")) is True
    assert T.is_thing(T.name_to_idx("sky-other-merged")) is False
    # idx is contiguous 0..132
    idxs = sorted(T.COCO_CLASSES)
    assert idxs[0] == 0 and idxs[-1] == 132
