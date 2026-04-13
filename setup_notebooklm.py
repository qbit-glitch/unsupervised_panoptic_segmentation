"""
setup_notebooklm.py — Bootstrap a NotebookLM notebook for MBPS BMVC 2026.

Creates notebook "MBPS BMVC 2026 — Panoptic Seg" and pre-loads:
  - 8 key arxiv papers as URL sources
  - 14 local markdown files (reports, plans, paper draft) as text sources

Usage:
    python setup_notebooklm.py

Prerequisites:
    notebooklm login   # one-time browser authentication
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

NOTEBOOK_NAME = "MBPS BMVC 2026 — Panoptic Seg"

VENV_CLI = Path("/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/notebooklm")

PROJECT_ROOT = Path(__file__).parent

ARXIV_SOURCES: list[dict] = [
    {"title": "CUPS — Unsupervised Panoptic Seg (CVPR 2025)", "url": "https://arxiv.org/abs/2504.01955"},
    {"title": "Panoptic Segmentation (Kirillov 2019)",         "url": "https://arxiv.org/abs/1801.00868"},
    {"title": "DINOv2 (Oquab 2023)",                          "url": "https://arxiv.org/abs/2304.07193"},
    {"title": "Depth Anything v3 (2024)",                     "url": "https://arxiv.org/abs/2412.04456"},
    {"title": "CAUSE / NeCo (van Gansbeke 2023)",             "url": "https://arxiv.org/abs/2306.02495"},
    {"title": "CutLER (Wang 2023)",                           "url": "https://arxiv.org/abs/2301.11750"},
    {"title": "Cascade Mask R-CNN (Cai 2019)",                "url": "https://arxiv.org/abs/1906.09756"},
    {"title": "STEGO (Hamilton 2022)",                        "url": "https://arxiv.org/abs/2203.08414"},
]

LOCAL_SOURCES: list[str] = [
    # Core paper
    "research_paper_draft.md",
    # Key ablation reports
    "reports/depth_model_ablation_study.md",
    "reports/novel_instance_ablation_final_report.md",
    "reports/dav3_k80_panoptic_pseudolabel_report.md",
    "reports/novel_instance_decomposition_brainstorm.md",
    "reports/coco_semantic_ablation_plan.md",
    "reports/unet_ablation_study.md",
    "reports/unet_phase2_architecture_ablation.md",
    "reports/cups_semantic_ablation_report.md",
    "reports/dinov3_stage3_cross_dataset_evaluation.md",
    # Plans
    "plans/2026-04-04-cups-backbone-ablation.md",
    "plans/novel_instance_ablation_da3_plan.md",
    # Discussion notes
    "chats_important_points/paper_writing_discussion_sesison.md",
    "chats_important_points/improving_falcon.md",
]

# ── CLI helpers ───────────────────────────────────────────────────────────────

def run(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run notebooklm CLI with given args."""
    cmd = [str(VENV_CLI)] + args
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return result


def run_json(args: list[str]) -> Optional[dict | list]:
    """Run CLI with --json flag and parse output."""
    result = run(args + ["--json"])
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


# ── Notebook management ───────────────────────────────────────────────────────

def get_notebook_id() -> Optional[str]:
    """Return notebook ID if a notebook named NOTEBOOK_NAME already exists."""
    data = run_json(["list"])
    # API returns {"notebooks": [...], "count": N}
    if isinstance(data, dict):
        notebooks = data.get("notebooks", [])
    elif isinstance(data, list):
        notebooks = data
    else:
        return None
    for nb in notebooks:
        if nb.get("title") == NOTEBOOK_NAME:
            return nb.get("id") or nb.get("notebookId")
    return None


def create_notebook() -> str:
    """Create the notebook and return its ID."""
    print(f"Creating notebook: {NOTEBOOK_NAME!r}")
    result = run(["create", NOTEBOOK_NAME])
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        sys.exit(1)
    # Re-fetch ID from list
    nb_id = get_notebook_id()
    if not nb_id:
        print("  ERROR: Notebook created but ID not found in list. Check `notebooklm list`.")
        sys.exit(1)
    return nb_id


def set_active_notebook(nb_id: str) -> None:
    """Set the notebook as active context."""
    result = run(["use", nb_id])
    if result.returncode != 0:
        print(f"  ERROR setting active notebook: {result.stderr.strip()}")
        sys.exit(1)


# ── Source management ─────────────────────────────────────────────────────────

def get_existing_source_titles() -> set[str]:
    """Return set of already-added source titles (for idempotency)."""
    data = run_json(["source", "list"])
    # API returns {"sources": [...], "count": N}
    if isinstance(data, dict):
        sources = data.get("sources", [])
    elif isinstance(data, list):
        sources = data
    else:
        return set()
    titles = set()
    for s in sources:
        t = s.get("title") or s.get("name") or ""
        if t:
            titles.add(t)
    return titles


def add_url_source(title: str, url: str, existing: set[str]) -> bool:
    """Add a URL source. Returns True if added, False if skipped."""
    if title in existing:
        print(f"  SKIP (exists): {title}")
        return False
    result = run(["source", "add", url, "--title", title])
    if result.returncode != 0:
        print(f"  FAIL: {title}\n    {result.stderr.strip()}")
        return False
    print(f"  + {title}")
    return True


def add_file_source(rel_path: str, existing: set[str]) -> bool:
    """Add a local file source. Returns True if added, False if skipped."""
    title = rel_path  # Use relative path as title for uniqueness
    if title in existing:
        print(f"  SKIP (exists): {rel_path}")
        return False
    abs_path = PROJECT_ROOT / rel_path
    if not abs_path.exists():
        print(f"  MISSING (skip): {rel_path}")
        return False
    result = run(["source", "add", str(abs_path), "--title", title])
    if result.returncode != 0:
        print(f"  FAIL: {rel_path}\n    {result.stderr.strip()}")
        return False
    print(f"  + {rel_path}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("NotebookLM Setup — MBPS BMVC 2026")
    print("=" * 60)

    # Check CLI is available
    if not VENV_CLI.exists():
        print(f"ERROR: CLI not found at {VENV_CLI}")
        print("Install with: pip install git+https://github.com/teng-lin/notebooklm-py.git")
        sys.exit(1)

    # 1. Create or reuse notebook
    nb_id = get_notebook_id()
    if nb_id:
        print(f"\nNotebook already exists (id={nb_id[:8]}…). Reusing.")
    else:
        nb_id = create_notebook()
        print(f"  Created (id={nb_id[:8]}…)")

    set_active_notebook(nb_id)
    print(f"Active notebook set.\n")

    # 2. Fetch existing sources (idempotency)
    existing = get_existing_source_titles()
    print(f"Existing sources: {len(existing)}\n")

    # 3. Add arxiv papers
    print(f"── ArXiv papers ({len(ARXIV_SOURCES)}) ──")
    url_added = 0
    for src in ARXIV_SOURCES:
        if add_url_source(src["title"], src["url"], existing):
            url_added += 1
            time.sleep(1.0)  # gentle rate-limit

    # 4. Add local files
    print(f"\n── Local files ({len(LOCAL_SOURCES)}) ──")
    file_added = 0
    for rel_path in LOCAL_SOURCES:
        if add_file_source(rel_path, existing):
            file_added += 1
            time.sleep(0.5)

    # 5. Summary
    total_new = url_added + file_added
    total_sources = len(existing) + total_new
    print(f"\n{'=' * 60}")
    print(f"Done. Added {total_new} new sources ({url_added} URLs + {file_added} files).")
    print(f"Total sources in notebook: ~{total_sources}")
    print(f"\nNext steps:")
    print(f"  notebooklm ask 'What is our best PQ result and which method achieved it?'")
    print(f"  notebooklm generate report briefing-doc")
    print(f"  notebooklm generate audio 'deep dive on pseudo-label compositing'")
    print("=" * 60)


if __name__ == "__main__":
    main()
