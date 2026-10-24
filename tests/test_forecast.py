"""The forecast reproduces the full RadioFisher code at commit ae723f7, and the
residual hook behaves as documented.

The reference was recorded with numpy 2.3.5 and scipy 1.18.0, where this code
reproduces it exactly. Other numpy and scipy builds differ by about 1e-9 relative,
so the comparisons allow 1e-8."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import radiofisher as rf
from radiofisher import baofisher as bf
from radiofisher.units import HOUR_IN_MHZ_INVERSE

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "reference_ae723f7.json").read_text())
ZMIN, ZMAX = 1.20, 1.25


def shelf(k, u, cn, cs):
    return cn * 1e-3 * (k < 0.2) * (1 + u)


@pytest.fixture(scope="module")
def inputs():
    cosmo = rf.load_power_spectrum(copy.deepcopy(rf.chime.cosmo), str(DATA / "pk_planck2018.dat"))
    fns = rf.background_evolution_splines(cosmo)
    expt = copy.deepcopy(rf.chime.CHIME)
    expt["ttot"] = 8760.0 * HOUR_IN_MHZ_INVERSE
    return cosmo, expt, fns


@pytest.fixture
def small_grid(monkeypatch):
    monkeypatch.setattr(bf, "NSAMP_K", 121)
    monkeypatch.setattr(bf, "NSAMP_U", 201)


@pytest.mark.parametrize("label,pres", [("scalar", 0.25), ("callable", shelf)])
def test_fisher_matches_the_full_code(inputs, small_grid, label, pres):
    cosmo, expt, fns = inputs
    F, names = rf.fisher(ZMIN, ZMAX, cosmo, dict(expt, P_res=pres), fns)
    ref = REFERENCE[label]
    expected_names = list(ref["names"])
    expected_names[3] = "s_NL"  # the reference file names this variance row sigma_NL
    assert names == expected_names
    np.testing.assert_allclose(F, np.array(ref["F"]), rtol=1e-8, atol=0)


def test_bin_setup_matches_the_full_code(inputs):
    cosmo, expt, fns = inputs
    c, e, Vfac, kgrid, ugrid = rf.prepare_bin(ZMIN, ZMAX, cosmo, expt, fns)
    for key, value in REFERENCE["setup"].items():
        got = e[key] if key == "dnutot" else c[key]
        assert got == pytest.approx(value, rel=1e-13), key
    assert Vfac > 0 and kgrid.size == bf.NSAMP_K and ugrid.size == bf.NSAMP_U


def test_integrands_match_the_full_code(inputs, small_grid):
    cosmo, expt, fns = inputs
    c, e, *_ = rf.prepare_bin(ZMIN, ZMAX, cosmo, dict(expt, P_res=shelf), fns)
    ref = REFERENCE["integrands"]
    D, names = rf.fisher_integrands(np.array(ref["k"]), np.array(ref["u"]), c, e)
    expected_names = list(ref["names"])
    expected_names[3] = "s_NL"  # the reference file names this variance row sigma_NL
    assert names == expected_names
    np.testing.assert_allclose(np.asarray(D), np.array(ref["D"]), rtol=1e-8, atol=1e-300)


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


def test_paired_nodes_match_cartesian_grid(inputs):
    cosmo, expt, fns = inputs
    c, e, *_ = rf.prepare_bin(ZMIN, ZMAX, cosmo, dict(expt, P_res=shelf), fns)
    k, mu = np.array([0.04, 0.09, 0.15]), np.array([0.0, 0.3, 0.7, 1.0])
    grid, names = rf.fisher_integrands(k, mu, c, e)
    kk, mm = np.meshgrid(k, mu)
    paired, paired_names = rf.fisher_integrands(kk.ravel(), mm.ravel(), c, e, paired=True)
    assert paired_names == names
    np.testing.assert_array_equal(paired, np.asarray(grid).reshape(len(names), -1))
    scalar, _ = rf.fisher_integrands(k[0], mu[1], c, e, paired=True)
    np.testing.assert_array_equal(np.asarray(scalar)[:, 0], np.asarray(grid)[:, 1, 0])


def test_s_nl_is_the_damping_variance_derivative(inputs):
    cosmo, expt, fns = inputs
    c, e, *_ = rf.prepare_bin(ZMIN, ZMAX, cosmo, expt, fns)
    k, mu = np.array([0.04, 0.09]), np.array([0.3, 0.7])
    derivs, names = rf.fisher_integrands(k, mu, c, e)
    kk, mm = np.meshgrid(k, mu)
    q, y = c['r'] * kk * np.sqrt(1 - mm**2), c['rnu'] * kk * mm
    step = 1e-3  # perturb the variance, not the damping length
    high = dict(c, sigma_nl=np.sqrt(c['sigma_nl']**2 + step))
    low = dict(c, sigma_nl=np.sqrt(c['sigma_nl']**2 - step))
    total = rf.Csignal(q, y, c, e) + rf.Cnoise(q, y, c, e) + rf.Cfg(q, y, c, e)
    numeric = (rf.Csignal(q, y, high, e) - rf.Csignal(q, y, low, e)) / (2 * step * total)
    np.testing.assert_allclose(derivs[names.index('s_NL')], numeric, rtol=1e-8)


def test_simpson_grid_integrates_a_separable_polynomial():
    k = np.linspace(0, 2, 7)
    mu = np.linspace(-1, 1, 9)
    integrand = mu[:, None]**2 * k[None, :]**2
    assert bf.integrate_grid(integrand, k, mu) == pytest.approx(16 / 9)


def test_bao_derivative_grid_entirely_below_cutoff():
    k = np.linspace(0.001, 0.02, 7)
    derivative = bf.fbao_derivative(lambda nodes: nodes**2, k)
    np.testing.assert_array_equal(derivative(k), 0)


@pytest.mark.parametrize("bounds", [(1.3, 1.2), (1.2, 1.2), (-1, 1), (0, np.nan)])
def test_invalid_redshift_bin_is_refused(inputs, bounds):
    cosmo, expt, fns = inputs
    with pytest.raises(ValueError, match="redshift bounds"):
        rf.prepare_bin(*bounds, cosmo, expt, fns)
