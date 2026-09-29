"""Independent quadrature/finite-difference oracles for parameter projections."""
import copy
import numpy as np
import pytest
from scipy.integrate import quad
from radiofisher import baofisher as rf, experiments, euclid, mg_growth


def observables(c, z, fs8):
    om, ol = c['omega_M_0'], c['omega_lambda_0']
    ok = 1-om-ol
    def E(z):
        return np.sqrt(om*(1+z)**3 + ok*(1+z)**2 +
                       ol*(1+z)**(3*(1+c['w0']+c['wa']))*np.exp(-3*c['wa']*z/(1+z)))
    def growth(z):
        return (om*(1+z)**3/E(z)**2)**c['gamma']
    chi = quad(lambda t: 1/E(t), 0, z, epsabs=1e-11)[0]
    distance = chi if abs(ok) < 1e-12 else (
        np.sinh(np.sqrt(ok)*chi)/np.sqrt(ok) if ok > 0 else
        np.sin(np.sqrt(-ok)*chi)/np.sqrt(-ok))
    f = growth(z)
    if fs8:
        f *= c['sigma_8']*np.exp(-quad(lambda t: growth(t)/(1+t), 0, z)[0])
    return np.array([f, distance/c['h'], c['h']*E(z)])


@pytest.mark.parametrize('curvature', [-0.08, -1e-8, 0., 1e-8, 0.08])
@pytest.mark.parametrize('dark_energy', [(-1., 0.), (-0.9, 0.2)])
@pytest.mark.parametrize('fs8', [False, True])
def test_eos_projection_against_independent_quad(curvature, dark_energy, fs8):
    c = copy.deepcopy(experiments.cosmo)
    c.update(omega_lambda_0=1-c['omega_M_0']-curvature,
             w0=dark_energy[0], wa=dark_energy[1])
    before = copy.deepcopy(c)
    derivs = rf.eos_fisher_matrix_derivs(c, rf.background_evolution_splines(c), fsigma8=fs8)
    assert c == before
    for column, param in enumerate(['curvature', 'omega_lambda_0', 'w0', 'wa', 'h', 'gamma', 'sigma_8']):
        plus, minus = copy.deepcopy(c), copy.deepcopy(c)
        step = 1e-5
        if param == 'curvature':
            plus['omega_M_0'] -= step
            minus['omega_M_0'] += step
        else:
            plus[param] += step
            minus[param] -= step
            if param == 'omega_lambda_0':
                plus['omega_M_0'] -= step
                minus['omega_M_0'] += step
        for z in [0.1, 0.8, 2., 5.]:
            expected = (observables(plus, z, fs8)-observables(minus, z, fs8))/(2*step)
            fid = observables(c, z, fs8)
            expected *= [1., -1/fid[1], 1/fid[2]]
            actual = [group[column](1/(1+z)) for group in derivs]
            np.testing.assert_allclose(actual, expected, rtol=8e-5, atol=3e-6)
    np.testing.assert_allclose([d(1.) for d in derivs[1]], [0, 0, 0, 0, 1/c['h'], 0, 0])


def test_scale_independent_growth_derivatives_are_defined_and_validated():
    c = copy.deepcopy(experiments.cosmo)
    ks = np.array([0.01, 0.1, 1.])
    ds = mg_growth.growth_param_derivs(1., ks, c)
    assert len(ds) == 4
    assert all(np.shape(d) == ks.shape and np.isfinite(d).all() for d in ds)
    refined = mg_growth.growth_param_derivs(1., ks, c, dx=[5e-4]*4)
    np.testing.assert_allclose(ds, refined, rtol=1e-5)
    with pytest.raises(ValueError):
        mg_growth.growth_param_derivs(1., ks, c, dx=[0])


def test_lss_distance_jacobian_and_no_input_mutation():
    F = np.array([[3., .2, .1], [.2, 5., .3], [.1, .3, 7.]])
    original = F.copy()
    names = ['DA', 'H', 'bias']
    z, DA, H = 1., 1700., 120.
    DV = ((1+z)**2*DA**2*rf.C*z/H)**(1/3)
    AP = (1+z)*DA*H/rf.C
    J = np.array([[DA/DV, DA/(3*AP), 0], [-H/DV, 2*H/(3*AP), 0], [0, 0, 1.]])
    J = np.diag([1/100., 1/1000., 1.])@J
    actual, labels = rf.transform_to_lss_distances(z, F, names, DA=DA, H=H, rescale_da=100, rescale_h=1000)
    np.testing.assert_allclose(actual, J.T@F@J)
    np.testing.assert_array_equal(F, original)
    assert labels == ['DV', 'F', 'bias'] and names == ['DA', 'H', 'bias']


@pytest.mark.parametrize('omegab', [False, True])
def test_detf_prior_projection_identity_basis(tmp_path, omegab):
    p = tmp_path/'prior.dat'
    np.savetxt(p, np.column_stack([np.arange(45), np.arange(45), np.ones(45)]))
    c = copy.deepcopy(experiments.cosmo)
    F = euclid.detf_to_rf(p, c, omegab=omegab)
    assert F.shape == (8, 8) and np.linalg.eigvalsh(F).min() > 0
    assert F[1, 1] == pytest.approx(36.)
    assert F[1, 2] == pytest.approx(sum((np.arange(36)+.5)*.025))
    assert F[7, 7] == pytest.approx(4/c['sigma_8']**2)
    assert F[0, 0] == pytest.approx((.963/c['ns'])**2)


@pytest.mark.parametrize('data', [[[0.5, 44, 1]], [[-1, 44, 1]], [[44, 44, np.nan]], [[0, 0, 1]], [[44, 44]]])
def test_detf_rejects_malformed_tables(tmp_path, data):
    path = tmp_path/'invalid.dat'
    np.savetxt(path, data)
    with pytest.raises(ValueError):
        euclid.detf_to_rf(path, experiments.cosmo)


def test_camb_density_projection_agrees_with_finite_difference():
    c = experiments.cosmo
    x = np.array([c['ns'], -1, 0, c['omega_b_0'], 1-c['omega_M_0']-c['omega_lambda_0'], c['omega_lambda_0'], c['h']])
    def convert(x):
        ns, w0, wa, ob, ok, ode, h = x
        return np.array([ns, w0, wa, ob*h*h, ok, (1-ok-ode-ob)*h*h, h])
    J = np.column_stack([(convert(x+e*1e-5)-convert(x-e*1e-5))/2e-5 for e in np.eye(7)])
    np.testing.assert_allclose(euclid.camb_to_baofisher(np.eye(7), c), J.T@J, atol=1e-10)
    transformed = euclid.euclid_to_rf(np.eye(7), c)
    assert np.linalg.eigvalsh(transformed).min() > 0
