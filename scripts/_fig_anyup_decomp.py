"""Figure-02: count decomposition (merge signature) + precision-recall trade, from summary.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/analysis-output/anyup_classagnostic")
r = json.load(open(OUT / "summary.json"))["results"]
b, a = r["bil"], r["any"]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# (a) TP/FP/FN counts — fragmentation would raise FP; merging drops BOTH TP and FP
keys = ["TP", "FP", "FN"]; x = np.arange(3); w = 0.38
ax[0].bar(x - w / 2, [b[k] for k in keys], w, label="bilinear", color="#4C72B0")
ax[0].bar(x + w / 2, [a[k] for k in keys], w, label="anyup", color="#C44E52")
for i, k in enumerate(keys):
    ax[0].annotate(f"{a[k]-b[k]:+d}", (i, max(a[k], b[k])), ha="center", va="bottom", fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels(keys)
ax[0].set_ylabel("segment count (all frames)"); ax[0].legend()
ax[0].set_title("Count decomposition: anyup drops TP AND FP → merging, not fragmenting")

# (b) precision-recall trade — anyup moves up-left (precision up, recall down)
ax[1].scatter([b["recall"]], [b["precision"]], s=90, color="#4C72B0", label="bilinear", zorder=3)
ax[1].scatter([a["recall"]], [a["precision"]], s=90, color="#C44E52", label="anyup", zorder=3)
ax[1].annotate("", xy=(a["recall"], a["precision"]), xytext=(b["recall"], b["precision"]),
               arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
for pt, name in [(b, "bilinear"), (a, "anyup")]:
    ax[1].annotate(name, (pt["recall"], pt["precision"]), textcoords="offset points",
                   xytext=(6, 6), fontsize=9)
ax[1].set_xlabel("recall (%)"); ax[1].set_ylabel("precision (%)")
ax[1].set_title("Precision–recall trade: anyup buys +2.5 precision for −2.8 recall")
ax[1].legend(); ax[1].grid(alpha=0.3)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figures/figure-02-count-decomposition.{ext}", dpi=140)
print("wrote figure-02")
