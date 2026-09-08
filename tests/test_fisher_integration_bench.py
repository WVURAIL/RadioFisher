"""Run the real derivative/integration path on a small deterministic spectrum."""
import copy
import numpy as np
import pytest
from radiofisher import baofisher as rf, experiments, galaxy


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(rf, 'NSAMP_K', 65)
    monkeypatch.setattr(rf, 'NSAMP_U', 33)
    c = copy.deepcopy(experiments.cosmo)
    c.update(pk_nobao=lambda k: 1e4*(k/0.1)**-1.2,
             fbao=lambda k: 0.05*np.sin(100*k)*np.exp(-(k/0.3)**2), k_in_max=2)
    return c, copy.deepcopy(experiments.exptS), rf.background_evolution_splines(c)


def evaluate(model, **kwargs):
    c, e, fns = model
    return rf.fisher(0.5, 0.6, c, e, fns, kmin=0.005, kmax=0.3, **kwargs)


@pytest.mark.parametrize('fraction', [0, 0.25, 1])
def test_every_fisher_element_scales_with_surviving_volume(model, fraction):
    c, e, _ = model
    before_c, before_e = copy.deepcopy(c), copy.deepcopy(e)
    full, names = evaluate(model)
    assert c == before_c and e == before_e
    e['vol_frac'] = fraction
    partial, partial_names = evaluate(model)
    assert names == partial_names
    np.testing.assert_allclose(partial, fraction*full, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(full, full.T)
    assert np.isfinite(full).all()
    assert np.linalg.eigvalsh(full).min() >= -1e-10*np.linalg.norm(full, 2)


def test_more_noise_cannot_increase_information_and_cv_is_an_upper_bound(model):
    full, names = evaluate(model)
    cv, _ = evaluate(model, cv_limited=True)
    model[1]['ttot'] /= 2
    noisy, _ = evaluate(model)
    for difference in (full-noisy, cv-full):
        assert np.linalg.eigvalsh(difference).min() >= -1e-9*np.linalg.norm(cv, 2)
    assert full[names.index('A'), names.index('A')] > noisy[names.index('A'), names.index('A')]


def test_binned_power_and_optional_neutrino_and_bias_derivatives(model):
    F, centers, info, names = evaluate(model, return_pk=True, kbins=np.array([0.005, 0.03, 0.1, 0.3]))
    assert F.shape == (len(names), len(names)) and np.isfinite(F).all()
    np.testing.assert_allclose(centers, [0.0175, 0.065, 0.2])
    assert names[-3:] == ['pk0', 'pk1', 'pk2'] and info['Vfac'] > 0
    F, names = evaluate(model, massive_nu_fn=lambda k: np.full_like(k, 0.2),
                        Neff_fn=lambda k: np.full_like(k, 0.3), switches=['sdbias'])
    assert F.shape == (len(names), len(names)) and np.isfinite(F).all()
    assert {'Mnu', 'N_eff', 'b_1'} <= set(names)
    with pytest.raises(NameError, match='kbins'):
        evaluate(model, return_pk=True)


def test_galaxy_pipeline_uses_real_integrands(model):
    c, _, fns = model
    F, names = galaxy.fisher_galaxy_survey(0.5, 0.6, 1e-3, 1.5, c,
                                        {'fsky': 0.2, 'k_nl0': 0.14, 'use': experiments.USE}, fns)
    assert F.shape == (len(names), len(names)) and np.isfinite(F).all()
    np.testing.assert_allclose(F, F.T)
    assert np.trace(F) > 0


def test_quadrature_matches_polynomial_integrals():
    k = np.linspace(0.1, 0.9, 65)
    u = np.linspace(-1, 1, 33)
    K, U = np.meshgrid(k, u)
    assert rf.integrate_grid(K**2*(1+U**2), k, u) == pytest.approx((0.9**3-0.1**3)/3*8/3)
    derivatives = [np.ones_like(K), U]
    matrix = rf.integrate_fisher_elements(derivatives, k, u)
    np.testing.assert_allclose(matrix, np.diag([2, 2/3])*(0.9**3-0.1**3)/3, atol=1e-14)


def test_growth_and_bias_sigma8_columns_match_finite_differences(model, monkeypatch):
    original = rf.fisher_integrands
    def checked(kgrid, ugrid, cosmo, expt, **kwargs):
        derivatives, names = original(kgrid, ugrid, cosmo, expt, **kwargs)
        K, U = np.meshgrid(kgrid, ugrid)
        q, y = cosmo['r']*K*np.sqrt(1-U**2), cosmo['rnu']*K*U
        noise = rf.Cnoise(q, y, cosmo, expt)+rf.Cfg(q, y, cosmo, expt)
        for name, parameter in [('fs8', 'f'), ('bs8', 'bHI')]:
            plus, minus = copy.deepcopy(cosmo), copy.deepcopy(cosmo)
            step = 1e-5
            delta = step/(cosmo['D']*cosmo['sigma_8'])
            plus[parameter] += delta
            minus[parameter] -= delta
            if parameter == 'bHI':
                plus['btot'] = plus['bHI']
                minus['btot'] = minus['bHI']
            expected = (np.log(rf.Csignal(q, y, plus, expt)+noise)-
                        np.log(rf.Csignal(q, y, minus, expt)+noise))/(2*step)
            np.testing.assert_allclose(derivatives[names.index(name)], expected, rtol=2e-5, atol=2e-10)
        return derivatives, names
    monkeypatch.setattr(rf, 'fisher_integrands', checked)
    evaluate(model)


@pytest.mark.parametrize('switch', ['mg', 'oldmg'])
def test_modified_gravity_and_scale_bins_are_finite(model, switch):
    model[0]['fs8_kbins'] = [0.005, .1, .3]
    model[0]['f0_kbins'] = [0.005, .1, .3]
    F, names = evaluate(model, switches=[switch])
    assert {'gamma0', 'gamma1', 'eta0', 'eta1', 'A_xi', 'logkmg', 'k0fs8', 'k1bs8', 'f0k0'} <= set(names)
    assert np.isfinite(F).all()


def test_galaxy_binned_spectrum_and_photometric_damping(model):
    c, _, fns = model
    F, centers, info, names = galaxy.fisher_galaxy_survey(.5, .6, 1e-3, 1.5, c,
        {'fsky': .2, 'k_nl0': .14, 'use': experiments.USE, 'sigma_z0': .01},
        fns, kbins=np.array([.005, .03, .1, .3]), return_pk=True)
    assert names[-3:] == ['pk0', 'pk1', 'pk2']
    assert F.shape[0] == len(names) and np.isfinite(F).all() and info['Vsurvey'] > 0


def test_galaxy_pipeline_uses_its_own_rsd_model(model, monkeypatch):
    monkeypatch.setattr(rf, 'RSD_FUNCTION', 'kaiser')
    monkeypatch.setattr(galaxy, 'RSD_FUNCTION', 'loeb')
    original = rf.fisher_integrands
    def checked(k, u, c, expt, **kwargs):
        derivs, names = original(k, u, c, expt, **kwargs)
        K, U = np.meshgrid(k, u)
        q, y = c['r']*K*np.sqrt(1-U**2), c['rnu']*K*U
        plus, minus = copy.deepcopy(c), copy.deepcopy(c)
        plus['f'] += 1e-5
        minus['f'] -= 1e-5
        expected = (np.log(galaxy.Csignal_galaxy(q, y, plus, expt)+1/c['ngal'])-
                    np.log(galaxy.Csignal_galaxy(q, y, minus, expt)+1/c['ngal']))/2e-5
        np.testing.assert_allclose(derivs[names.index('f')], expected, rtol=1e-5, atol=1e-9)
        return derivs, names
    monkeypatch.setattr(rf, 'fisher_integrands', checked)
    c, _, fns = model
    galaxy.fisher_galaxy_survey(.5, .6, .001, 1.5, c,
        {'fsky': .2, 'k_nl0': .14, 'use': experiments.USE}, fns)
