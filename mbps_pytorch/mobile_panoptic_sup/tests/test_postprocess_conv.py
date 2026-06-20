import numpy as np

from mbps_pytorch.mobile_panoptic_sup.panoptic_postprocess_conv import semantic_to_panoptic


def test_two_disconnected_things_become_two_instances():
    thing = 0                       # idx 0 = person (a thing)
    sem = np.full((6, 6), 130, np.int64)   # 130 = a stuff class background
    sem[0:2, 0:2] = thing
    sem[4:6, 4:6] = thing           # two disconnected blobs of the same thing class
    pan, seg2cat = semantic_to_panoptic(sem)
    inst_ids = [sid for sid in np.unique(pan) if sid // 1000 == thing]
    assert len(inst_ids) == 2
    assert all(seg2cat[s] == thing for s in inst_ids)


def test_stuff_is_single_segment():
    sem = np.full((4, 4), 130, np.int64)   # all one stuff class
    pan, seg2cat = semantic_to_panoptic(sem)
    assert list(np.unique(pan)) == [130 * 1000]
    assert seg2cat[130 * 1000] == 130
