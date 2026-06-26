#!/usr/bin/env python3
"""Compare raw pseudo-labels vs the fine-tuned EoMT checkpoint's predictions.

For a few frames, builds a vertical composite: original | raw _pan.png class map |
checkpoint prediction — BOTH colorized with the SAME fixed class palette so colours
mean the same class on each panel. Uses the latest (lowest-loss) step_*.pt.
"""
import glob, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/mnt/HDD_16TB/jaipur_work")
from jaipur_eomt_finetune import build_model, VOID, DIV, REPO          # noqa: E402
from transformers import AutoImageProcessor                            # noqa: E402

# fixed Cityscapes-ish palette per class idx (0..22); same for raw + pred
PAL = {0:(128,64,128),1:(244,35,232),2:(70,70,70),3:(102,102,156),4:(190,153,153),
       5:(153,153,153),6:(250,170,30),7:(220,220,0),8:(107,142,35),9:(152,251,152),
       10:(70,130,180),11:(220,20,60),12:(255,0,0),13:(0,0,142),14:(0,0,70),15:(0,60,100),
       16:(0,80,100),17:(0,0,230),18:(119,11,32),19:(255,140,0),20:(160,82,45),
       21:(255,215,0),22:(75,0,130)}
W_OUT = 1024


def cls_color(clsmap):
    out = np.zeros((*clsmap.shape, 3), np.uint8)
    for c in np.unique(clsmap):
        if c != 255:
            out[clsmap == c] = PAL.get(int(c), (127, 127, 127))
    return out


def label(arr, text):
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 230, 26], fill=(0, 0, 0))
    d.text((6, 6), text, fill=(255, 255, 255))
    return np.array(im)


def main():
    ckpt = sorted(glob.glob("/mnt/HDD_16TB/jaipur_work/eomt_ft_unfreeze/step_*.pt"),
                  key=lambda p: int(p.split("step_")[1].split(".")[0]))[-1]
    dev = "cuda"
    proc = AutoImageProcessor.from_pretrained(REPO)
    model = build_model(8).to(dev).eval()
    sd = torch.load(ckpt, map_location=dev)
    model.load_state_dict(sd["model"])
    print(f"loaded {ckpt} (step {sd['step']})", flush=True)

    frames = "/mnt/HDD_16TB/datasets/jaipur_dashcam_frames"
    labels = "/mnt/HDD_16TB/jaipur_work/labels"
    out = "/mnt/HDD_16TB/jaipur_work/compare"; os.makedirs(out, exist_ok=True)
    alls = sorted(p[:-8].split("/")[-1] for p in glob.glob(os.path.join(labels, "*_pan.png")))
    picks = alls[::max(1, len(alls) // 6)][:6]                          # 6 spread across videos

    for s in picks:
        img = Image.open(os.path.join(frames, s + ".jpg")).convert("RGB")
        W, H = img.size; h = int(H * W_OUT / W)
        pan = np.array(Image.open(os.path.join(labels, s + "_pan.png"))).astype(np.int64)
        raw = cls_color(np.where(pan == VOID, 255, pan // DIV).astype(np.int32))
        enc = proc(images=[img], return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(pixel_values=enc["pixel_values"])
        res = proc.post_process_panoptic_segmentation(o, target_sizes=[(H, W)])[0]
        seg = res["segmentation"].cpu().numpy(); pc = np.full((H, W), 255, np.int32)
        for si in res["segments_info"]:
            pc[seg == si["id"]] = si["label_id"]
        pred = cls_color(pc)
        rs = lambda a: np.array(Image.fromarray(a).resize((W_OUT, h), Image.NEAREST))
        comp = np.concatenate([label(rs(np.array(img)), "original"),
                               label(rs(raw), "raw pseudo-label"),
                               label(rs(pred), "checkpoint prediction")], axis=0)
        Image.fromarray(comp).save(os.path.join(out, s + "_compare.png"))
        print("saved", s, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
