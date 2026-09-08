"""Differentiate the actual covariance independently of the Fisher code."""
import copy
import numpy as np
import pytest
from radiofisher import baofisher as rf, experiments, galaxy


@pytest.fixture
def signal_model():
    c = copy.deepcopy(experiments.cosmo)
    c.update(z=1., r=2000., rnu=3000., f=.8, D=.6, bHI=2., bgal=2., btot=2.,
             Tb=.15, ngal=.001, pk_nobao=lambda k: 1e4*(k/.1)**-1.2,
             fbao=lambda k: .02*np.sin(100*k), dfbao_dk=lambda k: 2*np.cos(100*k))
    e = copy.deepcopy(experiments.exptS)
    e.update(dnutot=10., kfg_fac=0., k_nl0=.5)
    return c, e


@pytest.mark.parametrize('model', ['kaiser', 'loeb'])
@pytest.mark.parametrize('is_galaxy', [False, True])
def test_named_derivatives_match_covariance_finite_differences(signal_model, monkeypatch, model, is_galaxy):
    c, e = signal_model
    monkeypatch.setattr(rf, 'RSD_FUNCTION', model)
    monkeypatch.setattr(galaxy, 'RSD_FUNCTION', model)
    if is_galaxy:
        e['sigma_z0'] = .002
    k = np.geomspace(.035, .2, 13)
    u = np.array([-.9, -.4, 0., .4, .9])
    K, U = np.meshgrid(k, u)
    q, y = c['r']*K*np.sqrt(1-U**2), c['rnu']*K*U
    signal = galaxy.Csignal_galaxy if is_galaxy else rf.Csignal
    derivs, names = rf.fisher_integrands(k, u, c, e, galaxy_survey=is_galaxy, cs_galaxy=signal)
    noise = 1/c['ngal'] if is_galaxy else rf.Cnoise(q, y, c, e)+rf.Cfg(q, y, c, e)
    for name, key in [('f', 'f'), ('fs8', 'f'), ('bs8', 'bHI'), ('aperp', 'aperp'), ('apar', 'apar')]:
        plus, minus = copy.deepcopy(c), copy.deepcopy(c)
        step = 1e-5
        delta = step/(c['D']*c['sigma_8']) if name in ('fs8', 'bs8') else step
        plus[key] += delta
        minus[key] -= delta
        if key == 'bHI':
            plus['btot'] = plus['bgal'] = plus['bHI']
            minus['btot'] = minus['bgal'] = minus['bHI']
        expected = (np.log(signal(q, y, plus, e)+noise)-np.log(signal(q, y, minus, e)+noise))/(2*step)
        np.testing.assert_allclose(derivs[names.index(name)], expected, rtol=3e-5, atol=2e-9,
                                   err_msg=f'{model}, galaxy={is_galaxy}, {name}')


@pytest.mark.parametrize('wedge', ['horizon', '3pb'])
def test_wedge_is_symmetric_in_signed_radial_modes(signal_model, wedge):
    c, e = signal_model
    e['wedge'] = wedge
    e['Ddish'] = 5.
    q = np.full(4, 100.)
    y = np.array([-400., -5., 5., 400.])
    noise = rf.Cnoise(q, y, c, e)
    np.testing.assert_array_equal(noise, noise[::-1])
    assert noise[0] < rf.INF_NOISE and noise[1] == rf.INF_NOISE


def test_full_ap_derivative_matches_sum_of_individually_enabled_terms(signal_model):
    c, e = signal_model
    k, u = np.geomspace(.035, .2, 13), np.linspace(-1., 1., 9)
    full, names = rf.fisher_integrands(k, u, copy.deepcopy(c), e)
    e['use'] = {key: True for key in e['use']}
    e['use']['alpha_all'] = False
    split, _ = rf.fisher_integrands(k, u, copy.deepcopy(c), e)
    for label in ['aperp', 'apar']:
        np.testing.assert_allclose(split[names.index(label)], full[names.index(label)], atol=1e-14)


@pytest.mark.parametrize('mode', ['i', 'ipaf', 'iaa', 'icyl', 'dish', 'paf'])
@pytest.mark.parametrize('frequency', [600., 900.])
def test_receiver_modes_obey_time_and_polarization_scaling(signal_model, mode, frequency):
    c, e = signal_model
    c['z'] = e['nu_line']/frequency-1
    e.update(mode=mode, Ddish=5., Dmin=10., Dmax=200., Ndish=50, Nbeam=1,
             nu_crit=700., Ncyl=4, cyl_area=500., npol=2)
    q, y = np.array([200., 250.]), np.array([100., 150.])
    base = rf.Cnoise(q, y, c, e)
    assert np.all((base > 0) & (base < rf.INF_NOISE))
    deeper = dict(e, ttot=e['ttot']*4)
    np.testing.assert_allclose(rf.Cnoise(q, y, c, deeper), base/4)
    single_pol = dict(e, npol=1)
    np.testing.assert_allclose(rf.Cnoise(q, y, c, single_pol), 2*base)
