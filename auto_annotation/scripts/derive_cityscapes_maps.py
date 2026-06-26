#!/usr/bin/env python3
"""Derive Cityscapes-style semantic + instance maps from the panoptic _pan.png.

No re-inference: _pan.png (uint16 id = trainId*1000 + inst, void=65535) already holds
everything. Per frame writes alongside it:
  {stem}_sem.png   uint8  Cityscapes labelIds (road=7, car=26, ...; 0 = unlabeled/void)
  {stem}_inst.png  uint16 Cityscapes instanceIds: stuff -> labelId; thing -> labelId*1000+inst
Idempotent (skips frames whose _sem.png already exists) -> safe to re-run while labeling.
"""
import argparse, glob, os
import numpy as np
from PIL import Image

# my trainId (idx, see autolabel _CS order) -> Cityscapes labelId.
# 0..18 = the 19 standard classes; 19..22 = Indian things (no official id) -> custom 34..37.
TRAINID2LABELID = {0:7, 1:8, 2:11, 3:12, 4:13, 5:17, 6:19, 7:20, 8:21, 9:22, 10:23,
                   11:24, 12:25, 13:26, 14:27, 15:28, 16:31, 17:32, 18:33,
                   19:34, 20:35, 21:36, 22:37}
THING_TRAINIDS = set(range(11, 23))   # person..cart
VOID_LABELID = 0                      # Cityscapes 'unlabeled'


def derive(pan_path):
    pan = np.array(Image.open(pan_path)).astype(np.int32)
    pan = np.where(pan == 65535, 255000, pan)     # undo void sentinel
    cls, inst = pan // 1000, pan % 1000
    sem = np.zeros(cls.shape, np.uint8)
    instids = np.zeros(cls.shape, np.uint16)
    for tid in np.unique(cls):
        m = cls == tid
        if tid == 255:
            sem[m] = VOID_LABELID; instids[m] = VOID_LABELID; continue
        lid = TRAINID2LABELID[int(tid)]
        sem[m] = lid
        instids[m] = (lid * 1000 + inst[m]).astype(np.uint16) if tid in THING_TRAINIDS else lid
    return sem, instids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--shard", default="0/1")        # k/n: parallelize across processes
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    pans = sorted(glob.glob(os.path.join(a.labels, "*_pan.png")))
    pans = [p for i, p in enumerate(pans) if i % n == k]
    done = 0
    for p in pans:
        base = p[:-len("_pan.png")]
        if os.path.exists(base + "_sem.png"):
            continue
        sem, inst = derive(p)
        Image.fromarray(sem, mode="L").save(base + "_sem.png")
        Image.fromarray(inst, mode="I;16").save(base + "_inst.png")
        done += 1
    print(f"derived sem+inst for {done} new ({len(pans)} pan files total)")


if __name__ == "__main__":
    main()
