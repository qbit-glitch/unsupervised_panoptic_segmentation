#!/usr/bin/env python3
"""Render a live progress bar from Detectron2 evaluation logs."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


PROGRESS_RE = re.compile(
    r"Inference done (?P<done>\d+)/(?P<total>\d+).*?"
    r"Total: (?P<seconds>[0-9.]+) s/iter\. ETA=(?P<eta>[^\n\r]+)"
)
START_RE = re.compile(r"Start inference on (?P<total>\d+) batches")


def read_tail(path: Path, max_bytes: int = 512 * 1024) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def latest_progress(text: str) -> tuple[int, int, float | None, str | None] | None:
    matches = list(PROGRESS_RE.finditer(text))
    if matches:
        match = matches[-1]
        return (
            int(match.group("done")),
            int(match.group("total")),
            float(match.group("seconds")),
            match.group("eta").strip(),
        )

    start_matches = list(START_RE.finditer(text))
    if start_matches:
        return (0, int(start_matches[-1].group("total")), None, None)
    return None


def render_bar(done: int, total: int, seconds: float | None, eta: str | None, width: int) -> str:
    total = max(total, 1)
    frac = min(max(done / total, 0.0), 1.0)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    rate = f"{seconds:.3f}s/img" if seconds is not None else "warming up"
    eta_text = eta or "ETA pending"
    return f"[{bar}] {done}/{total} ({frac * 100:5.2f}%) {rate} {eta_text}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=42)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    last_line = ""
    while True:
        if not log_path.exists():
            line = f"waiting for {log_path}"
        else:
            progress = latest_progress(read_tail(log_path))
            if progress is None:
                line = f"waiting for progress in {log_path}"
            else:
                line = render_bar(*progress, width=args.width)

        if args.once:
            print(line)
            return 0

        pad = " " * max(0, len(last_line) - len(line))
        print("\r" + line + pad, end="", flush=True)
        last_line = line
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
