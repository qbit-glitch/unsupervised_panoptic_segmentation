from mbps_pytorch.ga_uniap.config import VARIANTS, weight_sweep


def test_variants_registry():
    assert VARIANTS["V0_vanilla"]["weights"] == (1.0, 0.0, 0.0)
    assert VARIANTS["V3_geom_only"]["weights"][0] == 0.0
    assert VARIANTS["V2_split"]["mode"] == "split"
    assert set(VARIANTS) == {"V0_vanilla", "V1_augment", "V2_split", "V3_geom_only"}


def test_weight_sweep_grid():
    sweep = weight_sweep()
    assert (1.0, 0.5, 0.3) in sweep
    assert all(w[0] == 1.0 for w in sweep)
    assert len(sweep) == 9
