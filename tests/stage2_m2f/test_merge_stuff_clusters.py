"""Regression guards for ``scripts/merge_stuff_clusters_k80_to_25.py``.

M1 shrinks the Stage-2 M2F output space from 80 to 25 by aggregating the
65 k=80 stuff clusters into 10 CAUSE-style super-classes while keeping
the 15 thing clusters standalone. The LUT is derived from the santosh
``cluster_to_class`` mapping and must preserve three invariants the
downstream ``PseudoLabelDataset`` relies on:

1. Every ``old_id`` in [0, 80) maps to exactly one ``new_id`` in [0, 25).
2. Thing clusters land contiguously at new IDs 10..24 (stuff occupy 0..9),
   so ``PseudoLabelDataset`` can still recover them via its
   ``distribution inside object proposals / distribution all pixels``
   ratio under ``THING_STUFF_THRESHOLD=0.05``.
3. The .pt distribution remap is linear (``index_add_``) so per-sample
   proposal density is preserved bucket-wise.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_merge_module():
    here = Path(__file__).resolve().parents[2] / "scripts" / "merge_stuff_clusters_k80_to_25.py"
    spec = importlib.util.spec_from_file_location("merge_stuff_clusters_k80_to_25", here)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


merge = _load_merge_module()


@pytest.fixture
def centroids_path() -> Path:
    p = (
        Path(__file__).resolve().parents[2]
        / "weights"
        / "kmeans_centroids_k80_santosh.npz"
    )
    if not p.is_file():
        pytest.skip(f"santosh centroids not present at {p}")
    return p


def test_build_lookup_is_bijective_on_thing_space(centroids_path: Path) -> None:
    lut = merge.build_lookup(str(centroids_path))
    assert lut.shape == (merge.NUM_OLD,)
    assert lut.min() >= 0
    assert lut.max() < merge.NUM_NEW

    things_old = set(merge.THINGS_OLD)
    for i, old in enumerate(merge.THINGS_OLD):
        # Things are stored in insertion order right after the stuff block.
        assert int(lut[old]) == merge.NUM_STUFFS + i, (
            f"thing cluster {old} should map to {merge.NUM_STUFFS + i}, "
            f"got {int(lut[old])}"
        )

    # No stuff cluster may land in the thing range.
    stuff_ids = {int(lut[i]) for i in range(merge.NUM_OLD) if i not in things_old}
    assert stuff_ids.issubset(set(range(merge.NUM_STUFFS)))


def test_build_lookup_every_stuff_bucket_is_nonempty(centroids_path: Path) -> None:
    """All 10 stuff super-classes must receive >=1 cluster (otherwise the
    downstream distribution has a zero row and CLASS_WEIGHTING=True will
    blow up computing 1/(p·N))."""
    lut = merge.build_lookup(str(centroids_path))
    things_old = set(merge.THINGS_OLD)
    bucket_counts = np.zeros(merge.NUM_STUFFS, dtype=np.int64)
    for old in range(merge.NUM_OLD):
        if old in things_old:
            continue
        bucket_counts[int(lut[old])] += 1
    assert (bucket_counts > 0).all(), (
        f"empty stuff bucket(s): {np.where(bucket_counts == 0)[0].tolist()}"
    )


def test_remap_png_linear_and_bounded(tmp_path, centroids_path: Path) -> None:
    """PNG remap must be the LUT applied pixel-wise, values clipped to [0, 25)."""
    from PIL import Image

    lut = merge.build_lookup(str(centroids_path))
    # Fabricate an 8x8 image covering a sweep of old IDs.
    arr = np.array(
        [[i + j for j in range(8)] for i in range(0, 64, 8)], dtype=np.uint8
    )
    assert arr.max() < merge.NUM_OLD
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    Image.fromarray(arr).save(src)

    merge.remap_png(src, dst, lut)
    out = np.array(Image.open(dst))
    expected = lut[arr].astype(np.uint8)
    assert np.array_equal(out, expected)
    assert out.max() < merge.NUM_NEW


def test_remap_pt_is_linear_and_conserves_mass(tmp_path, centroids_path: Path) -> None:
    """Remap must conserve per-key total mass and be exactly ``lut^T @ dist``."""
    lut = merge.build_lookup(str(centroids_path))

    dist_all = torch.randn(merge.NUM_OLD).abs()  # nonneg weights
    dist_in = torch.randn(merge.NUM_OLD).abs() * 0.1
    src = tmp_path / "src.pt"
    dst = tmp_path / "dst.pt"
    torch.save(
        {
            "distribution all pixels": dist_all,
            "distribution inside object proposals": dist_in,
        },
        src,
    )

    merge.remap_pt(src, dst, lut)
    out = torch.load(dst, weights_only=False)

    # Shape + dtype preservation.
    assert out["distribution all pixels"].shape == (merge.NUM_NEW,)
    assert out["distribution inside object proposals"].shape == (merge.NUM_NEW,)
    assert out["distribution all pixels"].dtype == dist_all.dtype

    # Mass conservation.
    torch.testing.assert_close(
        out["distribution all pixels"].sum(), dist_all.sum()
    )
    torch.testing.assert_close(
        out["distribution inside object proposals"].sum(), dist_in.sum()
    )

    # Explicit bucket check: manually sum via LUT.
    expected = torch.zeros(merge.NUM_NEW, dtype=dist_all.dtype)
    for old in range(merge.NUM_OLD):
        expected[int(lut[old])] += dist_all[old]
    torch.testing.assert_close(out["distribution all pixels"], expected)


def test_thing_new_ids_preserve_instance_proposal_ratio(tmp_path, centroids_path: Path) -> None:
    """After remap, each thing-only cluster must keep ratio ~1.

    DepthPro tau=0.20 proposals only fire on thing clusters, so
    ``inside/all`` is ~1 for things and ~0 for stuff. After the merge,
    new thing IDs (10..24) must preserve that property so
    ``PseudoLabelDataset`` still classifies them as thing via its
    ``THING_STUFF_THRESHOLD=0.05`` split.
    """
    lut = merge.build_lookup(str(centroids_path))
    dist_all = torch.zeros(merge.NUM_OLD)
    dist_in = torch.zeros(merge.NUM_OLD)
    # Assign each thing 1.0 inside=all (ratio=1); each stuff 1.0 all, 0.0 inside (ratio=0).
    things_old = set(merge.THINGS_OLD)
    for old in range(merge.NUM_OLD):
        dist_all[old] = 1.0
        if old in things_old:
            dist_in[old] = 1.0

    src = tmp_path / "sy.pt"
    dst = tmp_path / "sy_out.pt"
    torch.save(
        {
            "distribution all pixels": dist_all,
            "distribution inside object proposals": dist_in,
        },
        src,
    )
    merge.remap_pt(src, dst, lut)
    out = torch.load(dst, weights_only=False)
    ratio = out["distribution inside object proposals"] / (
        out["distribution all pixels"] + 1e-9
    )
    # New IDs 10..24 are the 15 thing clusters; each gets a single source
    # thing cluster so ratio must be 1.0 bucket-for-bucket.
    for new_id in range(merge.NUM_STUFFS, merge.NUM_NEW):
        assert float(ratio[new_id]) == pytest.approx(1.0)
    # New IDs 0..9 are pure stuff — ratio must be 0.
    for new_id in range(merge.NUM_STUFFS):
        assert float(ratio[new_id]) == pytest.approx(0.0)


def test_lut_rejects_bad_centroids_file(tmp_path) -> None:
    bad = tmp_path / "bad.npz"
    np.savez(bad, some_other_key=np.zeros(10))
    with pytest.raises(KeyError, match="cluster_to_class"):
        merge.build_lookup(str(bad))


def test_lut_rejects_unknown_cause_class(tmp_path) -> None:
    """If a stuff cluster's CAUSE class is outside ``STUFF_CAUSE_CLASSES``
    the merge must crash loudly instead of silently dropping pixels into
    bucket 0.
    """
    bad = tmp_path / "unknown_cause.npz"
    c2c = np.zeros(merge.NUM_OLD, dtype=np.uint8)
    # Pick a stuff cluster (not in THINGS_OLD) and plant an unknown cause.
    stuff_old = next(i for i in range(merge.NUM_OLD) if i not in merge.THINGS_OLD)
    c2c[stuff_old] = 99  # not in STUFF_CAUSE_CLASSES
    np.savez(bad, cluster_to_class=c2c, centroids=np.zeros((merge.NUM_OLD, 90)))
    with pytest.raises(ValueError, match="STUFF_CAUSE_CLASSES"):
        merge.build_lookup(str(bad))
