import tempfile
import textwrap
from pathlib import Path

from cups.config import get_default_config


def test_base_inheritance_simple_chain(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text(textwrap.dedent(
        """
        MODEL:
          META_ARCH: "Mask2FormerPanoptic"
          MASK2FORMER:
            NUM_QUERIES: 100
        """
    ))
    child = tmp_path / "child.yaml"
    child.write_text(textwrap.dedent(
        f"""
        _BASE_: "{parent.name}"
        MODEL:
          MASK2FORMER:
            NUM_QUERIES: 200
        """
    ))
    cfg = get_default_config(str(child))
    assert cfg.MODEL.META_ARCH == "Mask2FormerPanoptic"  # from parent
    assert cfg.MODEL.MASK2FORMER.NUM_QUERIES == 200       # child wins


def test_base_inheritance_two_levels(tmp_path: Path) -> None:
    grand = tmp_path / "grand.yaml"
    grand.write_text("MODEL:\n  META_ARCH: \"Mask2FormerPanoptic\"\n")
    mid = tmp_path / "mid.yaml"
    mid.write_text(f"_BASE_: \"{grand.name}\"\nMODEL:\n  MASK2FORMER:\n    NUM_QUERIES: 150\n")
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text(f"_BASE_: \"{mid.name}\"\nMODEL:\n  MASK2FORMER:\n    NUM_QUERIES: 250\n")
    cfg = get_default_config(str(leaf))
    assert cfg.MODEL.META_ARCH == "Mask2FormerPanoptic"
    assert cfg.MODEL.MASK2FORMER.NUM_QUERIES == 250
