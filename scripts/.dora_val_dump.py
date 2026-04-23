import torch, glob, re, os

d = "/home/santosh/cups/experiments/experiments/e2_depthpro_conv_dora_r4_2gpu/Unsupervised Panoptic Segmentation/wseh41f6/checkpoints"
files = sorted(glob.glob(os.path.join(d, "ups_checkpoint_step=*.ckpt")))

rows = []
for fp in files:
    m = re.search(r"step=(\d+)", fp)
    step = int(m.group(1)) if m else -1
    c = torch.load(fp, map_location="cpu", weights_only=False)
    pq_now = None
    for k, v in (c.get("callbacks") or {}).items():
        if isinstance(v, dict) and v.get("monitor") == "pq_val":
            cs = v.get("current_score")
            if cs is not None:
                pq_now = float(cs)
    rows.append((step, pq_now))

print(f"{'step':>6}  {'PQ@val':>8}")
print("-" * 20)
for s, pq in rows:
    pq_str = "None" if pq is None else f"{pq*100:.2f}%"
    print(f"{s:>6}  {pq_str:>8}")

last_fp = os.path.join(d, "last.ckpt")
if os.path.exists(last_fp):
    c = torch.load(last_fp, map_location="cpu", weights_only=False)
    for k, v in (c.get("callbacks") or {}).items():
        if isinstance(v, dict) and v.get("monitor") == "pq_val":
            print()
            print("Monotone improvements (best_k_models):")
            for p, sc in v.get("best_k_models", {}).items():
                m = re.search(r"step=(\d+)", p)
                s = int(m.group(1)) if m else -1
                print(f"  step={s}  PQ={float(sc)*100:.2f}%")
            best_score = v.get("best_model_score")
            best_path = v.get("best_model_path", "")
            if best_score is not None:
                print(f"\nBEST OVERALL: PQ={float(best_score)*100:.2f}% at {os.path.basename(best_path)}")
