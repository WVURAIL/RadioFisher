"""Check background projections against quadrature and finite differences.

Run: python -m unittest discover -s tests -p test_cosmology_projection.py -v
Requires NumPy and SciPy. Set RADIOFISHER_SOURCE to test another baofisher.py.
"""
import ast
import copy
import os
from pathlib import Path
import unittest

import numpy as np
import scipy.integrate
import scipy.interpolate


def load_model():
    source = Path(os.environ.get('RADIOFISHER_SOURCE', str(
        Path(__file__).resolve().parents[1] / 'radiofisher' / 'baofisher.py')))
    required = {'Ez', 'fgrowth', 'omegaM_z', 'background_evolution_splines',
                'fsigma8_for_params', 'fsigma8_derivs', 'eos_fisher_matrix_derivs'}
    scope = dict(np=np, copy=copy, scipy=scipy, C=3e5)
    if not hasattr(scipy.integrate, 'cumtrapz'):
        scipy.integrate.cumtrapz = scipy.integrate.cumulative_trapezoid
    # Load unchanged function bodies without legacy package imports or CAMB.
    selected = [node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in required]
    if {node.name for node in selected} != required:
        raise RuntimeError('Required functions missing from ' + str(source))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), scope)
    return scope


def observables(c, z, fs8):
    """Independent redshift-space quadrature; no production background helpers."""
    om, ol = c['omega_M_0'], c['omega_lambda_0']
    ok = 1-om-ol
    def expansion(z):
        return np.sqrt(om*(1+z)**3+ok*(1+z)**2
                       + ol*(1+z)**(3*(1+c['w0']+c['wa']))
                       * np.exp(-3*c['wa']*z/(1+z)))
    def growth(z):
        return (om*(1+z)**3/expansion(z)**2)**c['gamma']
    chi = scipy.integrate.quad(lambda x: 1/expansion(x), 0, z, epsabs=1e-11)[0]
    if abs(ok) < 1e-12:
        distance = chi
    elif ok > 0:
        distance = np.sinh(np.sqrt(ok)*chi)/np.sqrt(ok)
    else:
        distance = np.sin(np.sqrt(-ok)*chi)/np.sqrt(-ok)
    f = growth(z)
    if fs8:
        f *= c['sigma_8']*np.exp(-scipy.integrate.quad(
            lambda x: growth(x)/(1+x), 0, z, epsabs=1e-11)[0])
    return np.array([f, distance/c['h'], c['h']*expansion(z)])


class CosmologyProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def check_projection(self, fs8):
        parameters = ('curvature', 'omega_lambda_0', 'w0', 'wa', 'h', 'gamma', 'sigma_8')
        for curvature in (-.08, -1e-8, 0., 1e-8, .08):
            for w0, wa in ((-1., 0.), (-.9, .2)):
                c = dict(omega_M_0=.316, omega_lambda_0=.684-curvature,
                         h=.67, w0=w0, wa=wa, gamma=.55, sigma_8=.834)
                before = copy.deepcopy(c)
                background = self.model['background_evolution_splines'](c)
                derivatives = self.model['eos_fisher_matrix_derivs'](c, background, fsigma8=fs8)
                self.assertEqual(c, before)
                for column, parameter in enumerate(parameters):
                    step = 1e-5
                    plus, minus = copy.deepcopy(c), copy.deepcopy(c)
                    if parameter == 'curvature':
                        plus['omega_M_0'] -= step
                        minus['omega_M_0'] += step
                    else:
                        plus[parameter] += step
                        minus[parameter] -= step
                        if parameter == 'omega_lambda_0':
                            plus['omega_M_0'] -= step
                            minus['omega_M_0'] += step
                    for z in (.1, .8, 2., 5.):
                        with self.subTest(fs8=fs8, curvature=curvature, w0=w0,
                                          wa=wa, parameter=parameter, z=z):
                            fid = observables(c, z, fs8)
                            expected = (observables(plus, z, fs8)
                                        - observables(minus, z, fs8))/(2*step)
                            expected *= [1., -1/fid[1], 1/fid[2]]
                            actual = [group[column](1/(1+z)) for group in derivatives]
                            np.testing.assert_allclose(actual, expected, rtol=8e-5, atol=3e-6)
                expected_origin = [0., 0., 0., 0., 1/c['h'], 0., 0.]
                np.testing.assert_allclose([d(1.) for d in derivatives[1]], expected_origin)

    def test_growth_projection(self):
        self.check_projection(False)

    def test_fsigma8_projection(self):
        self.check_projection(True)


if __name__ == '__main__':
    unittest.main()
