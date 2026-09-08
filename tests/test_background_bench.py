"""Background and modified-growth checks against analytic universes and quadrature."""
import copy

import numpy as np
import pytest
from scipy.integrate import quad

from radiofisher import baofisher as rf, experiments, mg_growth as mg


@pytest.mark.parametrize("curvature", [-0.08, 0.0, 0.08])
def test_distances_and_growth_against_independent_quadrature(curvature):
    c = dict(experiments.cosmo, omega_M_0=0.3, omega_lambda_0=0.7-curvature, w0=-0.9, wa=0.15)
    H, distance, growth, rate = rf.background_evolution_splines(c, nsamples=4001, zmax=4)
    def e(z):
        return np.sqrt(0.3*(1+z)**3 + curvature*(1+z)**2
                       + c['omega_lambda_0']*(1+z)**(3*(1+c['w0']+c['wa']))*np.exp(-3*c['wa']*z/(1+z)))
    for z in [0, 0.5, 1.0, 2.0, 3.5]:
        chi = quad(lambda zz: 1/e(zz), 0, z)[0]
        radial = chi if curvature == 0 else ((np.sinh if curvature > 0 else np.sin)(np.sqrt(abs(curvature))*chi)
                                             / np.sqrt(abs(curvature)))
        assert H(z) == pytest.approx(100*c['h']*e(z), rel=1e-8)
        assert distance(z) == pytest.approx(rf.C/(100*c['h'])*radial, rel=4e-7, abs=1e-8)
        f = lambda zz: (0.3*(1+zz)**3/e(zz)**2)**0.55
        assert rate(z) == pytest.approx(f(z))
        assert growth(z) == pytest.approx(np.exp(-quad(lambda zz: f(zz)/(1+zz), 0, z)[0]), rel=4e-7)
    assert np.isnan(H(5))


@pytest.mark.parametrize("z", [0, 0.5, 2, 5])
def test_einstein_de_sitter_growth_solution(z):
    c = dict(experiments.cosmo, omega_M_0=1.0, omega_lambda_0=0.0)
    f, D = mg.growth_k(z, c, nsamp=8)
    k = np.geomspace(1e-3, 1, 7)
    np.testing.assert_allclose(f(k), 1, rtol=1e-6)
    np.testing.assert_allclose(D(k), 1/(1+z), rtol=1e-6)
    assert rf.omegaM_z(z, c) == pytest.approx(1)
    np.testing.assert_allclose(rf.fgrowth_k(c, z, k), [f(k), D(k)])
    assert mg.w(1/(1+z), c) == -1


@pytest.mark.parametrize("fsigma8", [False, True])
def test_modified_growth_derivative_converges_without_mutating_input(fsigma8):
    c = copy.deepcopy(experiments.cosmo)
    k = np.array([0.03, 0.1, 0.3])
    coarse = mg.growth_derivs(1, k, c, mg_params=['A_xi'], dx=[1e-4], fsigma8=fsigma8)[0]
    fine = mg.growth_derivs(1, k, c, mg_params=['A_xi'], dx=[5e-5], fsigma8=fsigma8)[0]
    np.testing.assert_allclose(coarse, fine, rtol=1e-3)
    assert np.isfinite(fine).all() and np.any(fine != 0)
    assert c == experiments.cosmo
    with pytest.raises(KeyError, match='not found'):
        mg.growth_derivs(1, k, c, mg_params=['absent'])


def test_sigma8_derivative_and_growth_parameter_branches():
    c = dict(experiments.cosmo, gamma0=0.5, gamma1=0.1, eta0=0.02, eta1=0.03)
    z = np.array([0.1, 1, 2])
    deriv = rf.fsigma8_derivs(z, c, params=['sigma_8', 'gamma'], dx=[1e-4, 1e-4])
    np.testing.assert_allclose(deriv[0], rf.fsigma8_for_params(z, c)/c['sigma_8'], rtol=1e-9)
    assert np.isfinite(deriv[1]).all()
    assert not np.allclose(rf.fgrowth(c, z), rf.fgrowth(c, z, usegamma=True))
