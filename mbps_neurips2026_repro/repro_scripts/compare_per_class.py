#!/usr/bin/env python3
"""Compare per-class PQ/TP between two evaluate_pseudolabel_quality JSONs.

Usage:
    python scripts/compare_per_class.py NEW.json BASELINE.json [--threshold-pq 0.5]

Reports aggregate PQ delta and per-class TP/PQ deltas, flagging which classes
gained/regressed. Used by the Stage-2 long-tail ablation chain decision gates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


def _load(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new", type=Path, help="Newer eval JSON")
    parser.add_argument("baseline", type=Path, help="Baseline eval JSON")
    parser.add_argument("--threshold-pq", type=float, default=0.5,
                        help="Aggregate PQ delta threshold for promotion (default 0.5)")
    parser.add_argument("--threshold-tp-pct", type=float, default=30.0,
                        help="Per-class TP percent gain to count as 'gain' (default 30%%)")
    args = parser.parse_args()

    new_data = _load(args.new)
    base_data = _load(args.baseline)
    new = new_data["summary"]
    base = base_data["summary"]

    delta_pq = new["PQ"] - base["PQ"]
    delta_st = new["PQ_stuff"] - base["PQ_stuff"]
    delta_th = new["PQ_things"] - base["PQ_things"]
    delta_miou = new["mIoU"] - base["mIoU"]

    print(f"{'Aggregate':<20} {'NEW':>10} {'BASE':>10} {'Delta':>10}")
    print("-" * 52)
    for label, n, b, d in (
        ("PQ", new["PQ"], base["PQ"], delta_pq),
        ("PQ_stuff", new["PQ_stuff"], base["PQ_stuff"], delta_st),
        ("PQ_things", new["PQ_things"], base["PQ_things"], delta_th),
        ("mIoU", new["mIoU"], base["mIoU"], delta_miou),
    ):
        print(f"{label:<20} {n:>10.2f} {b:>10.2f} {d:>+10.2f}")
    print()

    new_pc = new_data["per_class"]
    base_pc = base_data["per_class"]
    print(f"{'Class':<15} {'Type':<6} {'NEW PQ':>8} {'BASE PQ':>8} {'dPQ':>7}  "
          f"{'NEW TP':>7} {'BASE TP':>7} {'dTP%':>7}  Note")
    print("-" * 92)
    gains = 0
    regressions = 0
    for cls, m_new in new_pc.items():
        if cls not in base_pc:
            continue
        m_base = base_pc[cls]
        d_pq = m_new["PQ"] - m_base["PQ"]
        d_tp = m_new["TP"] - m_base["TP"]
        tp_pct = (100.0 * d_tp / max(1, m_base["TP"])) if m_base["TP"] > 0 else (
            float("inf") if d_tp > 0 else 0.0
        )
        note = ""
        if tp_pct >= args.threshold_tp_pct:
            note = "GAIN"
            gains += 1
        elif d_pq <= -args.threshold_pq or tp_pct <= -args.threshold_tp_pct:
            note = "REGRESS"
            regressions += 1
        type_short = m_new["type"][:5]
        print(f"{cls:<15} {type_short:<6} {m_new['PQ']:>8.2f} {m_base['PQ']:>8.2f} {d_pq:>+7.2f}  "
              f"{m_new['TP']:>7d} {m_base['TP']:>7d} {tp_pct:>+7.1f}  {note}")

    print()
    print(f"Summary: {gains} class(es) GAIN >={args.threshold_tp_pct}% TP, "
          f"{regressions} class(es) REGRESS >={args.threshold_pq} PQ")
    if delta_pq >= args.threshold_pq:
        print(f"Decision: PROMOTE (aggregate dPQ={delta_pq:+.2f} >= +{args.threshold_pq})")
        return 0
    if gains >= 3:
        print(f"Decision: PROMOTE ({gains} per-class gains >=3 even though aggregate dPQ={delta_pq:+.2f})")
        return 0
    if delta_pq <= -args.threshold_pq:
        print(f"Decision: ABORT (aggregate dPQ={delta_pq:+.2f} <= -{args.threshold_pq})")
        return 2
    print(f"Decision: NEUTRAL (aggregate dPQ={delta_pq:+.2f}, gains={gains}); inspect per-class table")
    return 1


if __name__ == "__main__":
    sys.exit(main())
