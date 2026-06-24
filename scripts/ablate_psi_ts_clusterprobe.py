"""ψ_ts PQ ablation: CLUSTERPROBE CAUSE-TR k27 + gated unMORE vs Cityscapes GT.

Global Hungarian (CUPS protocol): vectorised bincount cost matrix, scipy LSA,
then per-image PQ accumulation. ~2 min on Mac CPU for 500 val images × 6 ψ.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

CLUSTERPROBE_VAL = Path("/Volumes/code_files/datasets/cityscapes/fused_causetr_k27_CLUSTERPROBE_val")
GTFINE_VAL       = Path("/Volumes/code_files/datasets/cityscapes/gtFine/val")
# Pred is natively 512×1024; GT is 2048×1024 — resize GT to match pred
PRED_H, PRED_W = 512, 1024
GT_SCALE = 2   # 2048×1024 → 1024×512 (PIL W×H)

CS_RAW_TO_TRAIN = {
    7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9, 23:10,
    24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18
}
STUFF_IDS    = set(range(0, 11))
THING_IDS    = set(range(11, 19))
NUM_CLUSTERS = 27
NUM_CLASSES  = 19
PSI_VALUES   = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]

CLUSTER_RATIOS = {7:0.7342, 6:0.3506, 1:0.1429, 24:0.1391, 9:0.0873, 14:0.0342}
ratio = np.array([CLUSTER_RATIOS.get(c, 0.0) for c in range(NUM_CLUSTERS)])


def load_pred(path):
    return np.array(Image.open(path), dtype=np.int32)


def load_gt(gt_path):
    img = Image.open(gt_path)
    w, h = img.size
    inst = np.array(img.resize((w // GT_SCALE, h // GT_SCALE), Image.NEAREST), dtype=np.int32)
    cls_map = np.full(inst.shape, -1, dtype=np.int32)
    iid_map = np.zeros(inst.shape, dtype=np.int32)
    for raw, tid in CS_RAW_TO_TRAIN.items():
        cls_map[inst == raw] = tid
        m = (inst // 1000 == raw) & (inst >= 1000)
        if m.any():
            cls_map[m] = tid
            iid_map[m] = inst[m] % 1000
    return cls_map, iid_map


# ── Load images once ──────────────────────────────────────────────────────────
triples = []
for sp in sorted(CLUSTERPROBE_VAL.glob("*_semantic.png")):
    stem = sp.name.replace("_leftImg8bit_semantic.png", "").replace("_semantic.png", "")
    ip   = sp.parent / sp.name.replace("_semantic.png", "_instance.png")
    city = stem.split("_")[0]
    gp   = GTFINE_VAL / city / f"{stem}_gtFine_instanceIds.png"
    if ip.exists() and gp.exists():
        triples.append((sp, ip, gp))

print(f"Val triples: {len(triples)}", flush=True)
print("Loading...", flush=True)

sem_arrs, inst_arrs, gt_cls, gt_iid = [], [], [], []
for sp, ip, gp in triples:
    sem_arrs.append(load_pred(sp))
    inst_arrs.append(load_pred(ip))
    c, i = load_gt(gp)
    gt_cls.append(c)
    gt_iid.append(i)
print("Loaded.", flush=True)

# ── Global cost matrix via vectorised bincount ────────────────────────────────
# cost[c, t] = total pixels where cluster c covers GT class t
print("Building cost matrix...", flush=True)
cost = np.zeros((NUM_CLUSTERS, NUM_CLASSES), dtype=np.int64)
for sem, gc in zip(sem_arrs, gt_cls):
    valid = gc >= 0
    s = sem[valid].astype(np.int64)   # cluster ids
    g = gc[valid].astype(np.int64)    # gt class ids
    idx = s * NUM_CLASSES + g         # 1D joint index
    counts = np.bincount(idx, minlength=NUM_CLUSTERS * NUM_CLASSES)
    cost += counts.reshape(NUM_CLUSTERS, NUM_CLASSES)

row_ind, col_ind = linear_sum_assignment(-cost)
assignment = np.full(NUM_CLUSTERS, -1, dtype=np.int32)
for r, c in zip(row_ind, col_ind):
    assignment[r] = c

print("Hungarian assignment (cluster→trainID):", flush=True)
for c in range(NUM_CLUSTERS):
    tid = int(assignment[c])
    if tid >= 0:
        ctype = "thing" if tid in THING_IDS else "stuff"
        pct   = cost[c, tid] / (cost[c].sum() + 1e-9) * 100
        print(f"  c{c:02d} → tid {tid:2d} ({ctype})  overlap={pct:.0f}%")

# ── PQ computation for a given ψ ─────────────────────────────────────────────
def compute_pq(thing_clusters):
    tp_iou = np.zeros(NUM_CLASSES)
    fp     = np.zeros(NUM_CLASSES)
    fn     = np.zeros(NUM_CLASSES)

    for sem, inst, gc, gi in zip(sem_arrs, inst_arrs, gt_cls, gt_iid):
        # pred panoptic id: cluster*1000 for stuff, cluster*1000+inst_id for things
        pred = sem * 1000
        for tc in thing_clusters:
            m = sem == tc
            if m.any():
                pred[m] = tc * 1000 + inst[m]

        # gt panoptic id: tid*1000+iid (0 for stuff)
        gt_pan = np.where(gc >= 0, gc * 1000 + gi, -1)

        matched_pred = set()
        for gt_id in np.unique(gt_pan):
            if gt_id < 0:
                continue
            gt_tid = gt_id // 1000
            if not (0 <= gt_tid < NUM_CLASSES):
                continue
            gm = gt_pan == gt_id
            best_iou, best_pid = 0.0, None
            for pc in np.unique(pred[gm]):
                pc_cluster = pc // 1000
                if assignment[pc_cluster] != gt_tid:
                    continue
                pm  = pred == pc
                iou = float((gm & pm).sum()) / float((gm | pm).sum() + 1e-9)
                if iou > 0.5 and iou > best_iou:
                    best_iou, best_pid = iou, pc
            if best_pid is not None:
                tp_iou[gt_tid] += best_iou
                matched_pred.add(best_pid)
            else:
                fn[gt_tid] += 1
        for pc in np.unique(pred):
            if pc < 0 or pc in matched_pred:
                continue
            pc_cluster = pc // 1000
            tid = int(assignment[pc_cluster])
            if 0 <= tid < NUM_CLASSES:
                fp[tid] += 1

    denom  = tp_iou + 0.5 * fp + 0.5 * fn
    pq_cls = np.where(denom > 0, tp_iou / denom, 0.0)
    return pq_cls


# ── ψ sweep ───────────────────────────────────────────────────────────────────
print("\nψ\t\tPQ\tPQ_stuff\tPQ_things\tn_things", flush=True)
for psi in PSI_VALUES:
    thing_clusters = set(int(c) for c in np.where(ratio > psi)[0])
    pq_cls    = compute_pq(thing_clusters)
    pq_all    = pq_cls.mean() * 100
    pq_stuff  = pq_cls[sorted(STUFF_IDS)].mean() * 100
    pq_things = pq_cls[sorted(THING_IDS)].mean() * 100
    print(f"psi={psi:.2f}\t\t{pq_all:.2f}\t{pq_stuff:.2f}\t\t{pq_things:.2f}\t\t{len(thing_clusters)}  {sorted(thing_clusters)}", flush=True)

print("DONE")
