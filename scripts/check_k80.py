import numpy as np
from PIL import Image
from scipy import ndimage

root = "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80"
sem = np.array(Image.open(root + "/aachen_000000_000019_leftImg8bit_semantic.png"))
inst = np.array(Image.open(root + "/aachen_000000_000019_leftImg8bit_instance.png"))

thing_ids = [3, 11, 14, 15, 16, 29, 32, 37, 38, 45, 46, 62, 65, 73, 75]
cc_map = np.zeros_like(inst)
iid = 1
for cid in sorted(thing_ids):
    mask = sem == cid
    if mask.sum() == 0:
        continue
    labeled, n = ndimage.label(mask)
    for c in range(1, n + 1):
        if (labeled == c).sum() >= 100:
            cc_map[labeled == c] = iid
            iid += 1

k80_px = int((inst > 0).sum())
cc_px = int((cc_map > 0).sum())
overlap = int(((inst > 0) & (cc_map > 0)).sum())

print(f"k80 inst pixels: {k80_px}")
print(f"CC inst pixels:  {cc_px}")
print(f"Overlap: {overlap}")
print(f"IoU: {overlap / (k80_px + cc_px - overlap) * 100:.1f}%")
print(f"k80 count: {int(inst.max())}, CC count: {int(cc_map.max())}")

k80_sizes = sorted([(inst == i).sum() for i in range(1, int(inst.max()) + 1)])
print(f"k80 min size: {k80_sizes[0]}, max: {k80_sizes[-1]}")
