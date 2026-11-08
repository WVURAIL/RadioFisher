"""CHORD, HIRAX and the outriggers: layouts, dish noise and a forecast that runs."""
import copy
from pathlib import Path

import numpy as np
import pytest

import radiofisher as rf
from radiofisher import arrays, baofisher as bf, baselines, instrument

DATA = Path(__file__).parent / "data"
FITTED = ("A", "s_NL", "aperp", "apar", "fs8", "bs8")
ZMIN, ZMAX = 1420.406 / 606 - 1, 1420.406 / 600 - 1     # 600 to 606 MHz, in every band


@pytest.fixture(scope="module")
def background():
    cosmo = rf.load_power_spectrum(copy.deepcopy(rf.chime.cosmo), str(DATA / "pk_planck2018.dat"))
    return cosmo, rf.background_evolution_splines(cosmo)


@pytest.mark.parametrize("name", arrays.LAYOUTS)
def test_density_counts_every_pair(name):
    x, n = arrays.density(name).T
    receivers = arrays.LAYOUTS[name]["rows"] * arrays.LAYOUTS[name]["columns"]
    assert np.sum(n * 2 * np.pi * x * (x[1] - x[0])) == pytest.approx(receivers * (receivers - 1) / 2)


@pytest.mark.parametrize("name", arrays.LAYOUTS)
def test_packaged_tables_match_the_layouts(name):
    np.testing.assert_allclose(np.loadtxt(arrays.density_file(name)), arrays.density(name), rtol=1e-15)


def test_build_to_explicit_directory(tmp_path):
    paths = arrays.build(tmp_path)
    assert [p.name for p in paths] == [f"nx_{n}.dat" for n in arrays.LAYOUTS]


def test_short_baselines_are_dropped():
    x = (np.arange(40) + 0.5) / 850
    full = baselines.grid_density(x, 2, 2, 5.0, 5.0)
    cut = baselines.grid_density(x, 2, 2, 5.0, 5.0, min_baseline_m=6.0)
    assert full.sum() > cut.sum() > 0      # the 5 m pairs go and the 7.07 m diagonals stay


def test_the_chime_density_is_the_grid_density():
    x = np.loadtxt(baselines.inter_file())[:, 0]
    np.testing.assert_array_equal(baselines.density(x), baselines.grid_density(x, 4, 256, 22.0, 0.3048,
                                                                               inter_row_only=True))


def test_dish_area_and_beam():
    dish = arrays.HIRAX
    assert instrument.collecting_area_m2(dish) == pytest.approx(0.7 * np.pi * 9.0)
    assert instrument.beam_width(0.5, dish) == pytest.approx(0.5 / 6.0)
    cylinder = rf.chime.CHIME
    assert instrument.beam_width(0.5, cylinder) == pytest.approx(np.sqrt(0.5 * np.pi * 0.5 / 20.0))


def test_an_unsupported_mode_is_refused(background):
    cosmo, fns = background
    c, e, *_ = rf.prepare_bin(ZMIN, ZMAX, cosmo, dict(arrays.HIRAX, mode="single"), fns)
    with pytest.raises(ValueError, match="interferometer modes"):
        bf.Cnoise(np.array([0.1]), np.array([0.05]), c, e)


@pytest.mark.parametrize("name", arrays.EXPERIMENTS)
def test_each_array_forecasts_a_bin(background, monkeypatch, name):
    monkeypatch.setattr(bf, "NSAMP_K", 121)
    monkeypatch.setattr(bf, "NSAMP_U", 201)
    cosmo, fns = background
    F, names = rf.fisher(ZMIN, ZMAX, cosmo, dict(arrays.EXPERIMENTS[name], P_res=1.0), fns)
    result = rf.summarize_fisher(F, names, FITTED)
    assert np.all(np.isfinite(result.sigma)) and np.all(result.sigma > 0)
    assert np.all(np.isfinite(result.bias))
