import importlib

import numpy as np


def test_split_partitions_27_clusters():
    mod = importlib.import_module("mbps_pytorch.classify_stuff_things_freq")
    col = np.arange(100).reshape(10, 10) % 10           # column index 0..9
    sem = [np.where(col < 5, 0, 1) for _ in range(5)]   # cols 0-4 -> cluster 0, 5-9 -> cluster 1
    inst = []
    for _ in range(5):
        m = np.zeros((10, 10), dtype=np.int32)          # cluster-0 region stays background (0)
        m[:, 5:] = (np.arange(50).reshape(10, 5) % 4) + 1  # cluster-1 region: 4 instances
        inst.append(m)
    things, stuff = mod.classify_from_arrays(sem, inst, num_clusters=27, threshold=0.08)
    assert 1 in things and 0 in stuff
    assert len(things) + len(stuff) == 27


def test_cc_mode_instance_free():
    mod = importlib.import_module("mbps_pytorch.classify_stuff_things_freq")
    sems = []
    for _ in range(5):
        s = np.zeros((20, 20), dtype=np.int32)  # cluster 0 = single background blob (stuff)
        s[2:6, 2:6] = 1                          # cluster 1 = two separated blobs (thing)
        s[2:6, 14:18] = 1
        sems.append(s)
    things, stuff = mod.classify_from_semantic_cc(sems, num_clusters=27, threshold=0.08, min_area=4)
    assert 1 in things and 0 in stuff
    assert len(things) + len(stuff) == 27
