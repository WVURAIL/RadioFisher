"""Legacy CAMB helpers remain usable on supported SciPy versions."""
import copy
import numpy as np
import pytest
from scipy.integrate import quad
from radiofisher import camb_wrapper as camb, experiments


@pytest.mark.parametrize('curvature', [-0.1, 0, 0.1])
def test_distance_matches_quadrature_with_radiation_and_dark_energy(curvature):
    c = dict(experiments.cosmo, omega_lambda_0=0.684-curvature, w0=-0.9, wa=0.2)
    a = 0.2
    radiation = 2.47e-5/c['h']**2*(1+7/8*(4/11)**(4/3)*3.046)
    def integrand(x):
        de = c['omega_lambda_0']*np.exp(3*c['wa']*(x-1))*x**(-3*(1+c['w0']+c['wa']))
        return 1/np.sqrt(c['omega_M_0']*x+curvature*x*x+radiation+de*x**4)
    chi = quad(integrand, a, 1)[0]
    mapped = chi if curvature == 0 else ((np.sinh if curvature > 0 else np.sin)(np.sqrt(abs(curvature))*chi)
                                          / np.sqrt(abs(curvature)))
    assert camb.comoving_dist(a, c) == pytest.approx(camb.C/(100*c['h'])*mapped, rel=1e-7)
    assert 100 < camb.rsound(1/1090, c) < 180


def test_acoustic_angle_round_trip_does_not_mutate_fiducial():
    c = copy.deepcopy(experiments.cosmo)
    original = copy.deepcopy(c)
    theta = camb.cmb_to_theta(c, 0.69)
    assert c == original
    assert 0.009 < theta < 0.012
    assert camb.find_h_for_theta100(100*theta, c, nsamp=50) == pytest.approx(0.69, rel=1e-5)
    assert c == original


def test_parameter_writer_renames_array_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'paramfiles').mkdir()
    camb.camb_params('test.ini', hubble=68.32, scalar_spectral_index__1___=0.9742)
    text = (tmp_path/'paramfiles'/'test.ini').read_text()
    assert 'hubble=68.32\n' in text and 'scalar_spectral_index(1)=0.9742\n' in text
