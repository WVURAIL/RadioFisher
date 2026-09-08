"""Cache integrity and units, with a deterministic external CAMB boundary."""
import copy
from pathlib import Path
import numpy as np
import pytest
from radiofisher import baofisher as rf, experiments


@pytest.mark.parametrize('mode', ['matterpower', 'transfer', 'cl', 'cls'])
def test_cache_generation_units_roundtrip_and_input_ownership(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    calls = []
    params = {'hubble': 70., 'scalar_spectral_index': .96}
    original = copy.deepcopy(params)
    raw = np.array([[.01, 2., 3.], [.1, 4., 5.], [1., 8., 9.]])
    def write_params(name, **p):
        calls.append(p)
    def run(*args, **kwargs):
        root = calls[-1]['output_root']
        for suffix in ['matterpower', 'transfer_out', 'scalCls']:
            np.savetxt(root+'_'+suffix+'.dat', raw)
        return {} if mode in ('cl', 'cls') else {'sigma8': .4}
    monkeypatch.setattr(rf, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(rf.camb, 'camb_params', write_params)
    monkeypatch.setattr(rf.camb, 'run_camb', run)
    cache = tmp_path/'cache.dat'
    expected = raw.T.copy()
    if mode in ('matterpower', 'transfer'):
        expected[0] *= .7
    if mode == 'matterpower':
        expected[1] *= 4/.7**3
    actual = rf.cached_camb_output(params, cache, cosmo={'h': .7, 'sigma_8': .8}, mode=mode)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(rf.cached_camb_output(params, cache, mode=mode), actual)
    assert len(calls) == 1 and params == original
    with pytest.raises(ValueError):
        rf.cached_camb_output({'hubble': 80}, cache, mode=mode)
    np.testing.assert_allclose(rf.cached_camb_output({'hubble': 80}, cache, mode=mode, force_load=True), actual)
    rf.cached_camb_output(params, cache, mode=mode, force=True)
    assert len(calls) == 2


def test_bad_cache_header_and_mode_fail_clearly(tmp_path):
    cache = tmp_path/'bad.dat'
    cache.write_text('1 2\n3 4\n')
    with pytest.raises(ValueError, match='hash'):
        rf.cached_camb_output({}, cache)
    with pytest.raises(ValueError, match='mode'):
        rf.cached_camb_output({}, cache, mode='unknown')


@pytest.mark.parametrize('mnu', [0., .06])
def test_neutrino_derivative_matches_analytic_spectrum(monkeypatch, tmp_path, mnu):
    k = np.geomspace(.001, 1, 100)
    def cache(params, *args, **kwargs):
        value = params['omnuh2']*rf.NEUTRINO_MASS_DENSITY_EV if mnu else params['massless_neutrinos']
        return np.array([k, k**-1.2*np.exp(.2*value)])
    monkeypatch.setattr(rf, 'cached_camb_output', cache)
    derivative = rf.deriv_neutrinos(experiments.cosmo, str(tmp_path/'nu'), mnu=mnu, dNeff=1e-3, dmnu=1e-3)
    np.testing.assert_allclose(derivative(k), .2, rtol=1e-7)


def test_transfer_derivative_and_spectrum_decomposition(monkeypatch, tmp_path):
    k = np.geomspace(1e-4, 10., 2000)
    table = np.ones((7, k.size))
    table[0] = k
    monkeypatch.setattr(rf, 'cached_camb_output', lambda *a, **kw: table.copy())
    inverse, derivative, mnu = rf.deriv_transfer(experiments.cosmo, str(tmp_path/'transfer'))
    np.testing.assert_allclose(inverse(k), k**-2)
    np.testing.assert_allclose(derivative(k[10:-10]), -2*k[10:-10]**-3, rtol=1e-4)
    assert mnu is None and inverse(20.) == 0
    pk = 1e4*k**-.8*(1+.03*np.sin(110*k)*np.exp(-(k/.3)**2))
    smooth, wiggles = rf.spline_pk_nobao(k, pk)
    np.testing.assert_allclose(smooth(k)*(1+wiggles(k)), pk, rtol=1e-12)
    monkeypatch.setattr(rf, 'cached_camb_output', lambda *a, **kw: np.array([k, pk]))
    c = copy.deepcopy(experiments.cosmo)
    assert rf.load_power_spectrum(c, str(tmp_path/'pk')) is c
    np.testing.assert_allclose(c['pk_nobao'](k)*(1+c['fbao'](k)), pk, rtol=1e-12)


@pytest.mark.parametrize('value', [0., np.array(0.), np.array([0., 2., -4.])])
def test_inverse_interpolation_handles_zeros_without_warnings(value):
    with np.errstate(all='raise'):
        result = rf.inverse_interpfn(value)
    np.testing.assert_array_equal(result, np.where(np.asarray(value) != 0, np.asarray(value) != 0, 0)*
                                  np.divide(1., value, out=np.zeros_like(np.asarray(value), dtype=float), where=np.asarray(value) != 0))
