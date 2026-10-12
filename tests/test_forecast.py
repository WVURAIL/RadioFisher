"""The cut-down forecast reproduces RadioFisher 6348429 (before the cut), and the
residual hook behaves as documented."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import radiofisher as rf
from radiofisher import baofisher as bf
from radiofisher.units import HRS_MHZ

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "reference_6348429.json").read_text())
ZMIN, ZMAX = 1.20, 1.25


def shelf(k, u, cn, cs):
    return cn * 1e-3 * (k < 0.2) * (1 + u)


@pytest.fixture(scope="module")
def inputs():
    cosmo = rf.load_power_spectrum(copy.deepcopy(rf.chime.cosmo), str(DATA / "pk_planck2018.dat"))
    fns = rf.background_evolution_splines(cosmo)
    expt = copy.deepcopy(rf.chime.CHIME)
    expt["ttot"] = 8760.0 * HRS_MHZ
    return cosmo, expt, fns


@pytest.fixture
def small_grid(monkeypatch):
    monkeypatch.setattr(bf, "NSAMP_K", 121)
    monkeypatch.setattr(bf, "NSAMP_U", 201)


@pytest.mark.parametrize("label,pres", [("scalar", 0.25), ("callable", shelf)])
def test_fisher_matches_the_uncut_code(inputs, small_grid, label, pres):
    cosmo, expt, fns = inputs
    F, names = rf.fisher(ZMIN, ZMAX, cosmo, dict(expt, P_res=pres), fns)
    ref = REFERENCE[label]
    assert names == ref["names"]
    np.testing.assert_allclose(F, np.array(ref["F"]), rtol=1e-11, atol=0)


def test_bin_setup_matches_the_uncut_code(inputs):
    cosmo, expt, fns = inputs
    c, e, Vfac, kgrid, ugrid = rf.prepare_bin(ZMIN, ZMAX, cosmo, expt, fns)
    for key, value in REFERENCE["setup"].items():
        got = e[key] if key == "dnutot" else c[key]
        assert got == pytest.approx(value, rel=1e-13), key
    assert Vfac > 0 and kgrid.size == bf.NSAMP_K and ugrid.size == bf.NSAMP_U


def test_integrands_match_the_uncut_code(inputs, small_grid):
    cosmo, expt, fns = inputs
    c, e, *_ = rf.prepare_bin(ZMIN, ZMAX, cosmo, dict(expt, P_res=shelf), fns)
    ref = REFERENCE["integrands"]
    D, names = rf.fisher_integrands(np.array(ref["k"]), np.array(ref["u"]), c, e)
    assert names == ref["names"]
    np.testing.assert_allclose(np.asarray(D), np.array(ref["D"]), rtol=1e-12, atol=1e-300)


def test_residual_never_enters_the_covariance(inputs, small_grid):
    """The '_Pres' row is the only change: every other element is the clean forecast."""
    cosmo, expt, fns = inputs
    clean, names = rf.fisher(ZMIN, ZMAX, cosmo, expt, fns)
    dirty, dnames = rf.fisher(ZMIN, ZMAX, cosmo, dict(expt, P_res=shelf), fns)
    assert dnames == names + ["_Pres"]
    np.testing.assert_array_equal(dirty[:-1, :-1], clean)


def test_bias_is_linear_in_the_residual(inputs, small_grid):
    cosmo, expt, fns = inputs
    one, _ = rf.fisher(ZMIN, ZMAX, cosmo, dict(expt, P_res=0.1), fns)
    two, _ = rf.fisher(ZMIN, ZMAX, cosmo, dict(expt, P_res=0.2), fns)
    np.testing.assert_allclose(two[:-1, -1], 2 * one[:-1, -1], rtol=1e-12)


def test_noise_cuts(inputs):
    cosmo, expt, fns = inputs
    c, e, *_ = rf.prepare_bin(ZMIN, ZMAX, cosmo, expt, fns)
    kfg = 2 * np.pi * e["nu_line"] / (e["survey_dnutot"] * c["rnu"])
    knl = e["k_nl0"] * (1 + c["z"]) ** (2 / (2 + c["ns"]))
    q = np.array([[c["r"] * 0.05, c["r"] * 0.05, c["r"] * 0.05]])
    y = np.array([[c["rnu"] * kfg / 2, c["rnu"] * 0.05, c["rnu"] * 2 * knl]])
    cn = rf.Cnoise(q, y, c, e)
    assert cn[0, 0] == bf.INF_NOISE and cn[0, 2] == bf.INF_NOISE
    assert 0 < cn[0, 1] < bf.INF_NOISE


def test_only_the_cylinder_is_supported(inputs):
    cosmo, expt, fns = inputs
    c, e, *_ = rf.prepare_bin(ZMIN, ZMAX, cosmo, expt, fns)
    with pytest.raises(ValueError):
        rf.Cnoise(np.ones((1, 1)), np.ones((1, 1)), c, dict(e, mode="dish"))
