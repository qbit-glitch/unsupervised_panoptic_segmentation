"""Pytest configuration: expose the CUPS package to test modules.

The CUPS codebase lives under ``refs/cups/cups`` so it can coexist with
the upstream fork. Tests need to ``import cups.*`` without requiring a
``pip install -e``; adding ``refs/cups`` to ``sys.path`` here keeps the
import paths stable for every test collected under ``tests/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CUPS_ROOT = _PROJECT_ROOT / "refs" / "cups"

for _extra in (_PROJECT_ROOT, _CUPS_ROOT):
    _p = str(_extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)
