"""Check distance derivatives using an analytic spectrum and constant noise.

Run: python -m unittest discover -s tests -p test_distance_bao_derivatives.py -v
Requires NumPy. Set RADIOFISHER_SOURCE to test another baofisher.py.
"""
import ast
import copy
import os
from pathlib import Path
import unittest

import numpy as np


def load_model():
    source = Path(os.environ.get('RADIOFISHER_SOURCE', str(
        Path(__file__).resolve().parents[1] / 'radiofisher' / 'baofisher.py')))
    required = {'Ez', 'inverse_interpfn', 'logpk_derivative',
                'Csignal', 'fisher_integrands'}
    scope = dict(np=np, copy=copy, C=3e5, RSD_FUNCTION='kaiser')
    scope['Cnoise'] = lambda q, y, c, e: np.full_like(q, 1e-9)
    scope['Cfg'] = lambda q, y, c, e: np.zeros_like(q)
    # Load unchanged function bodies without legacy package imports or CAMB.
    selected = [node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in required]
    if {node.name for node in selected} != required:
        raise RuntimeError('Required functions missing from ' + str(source))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), scope)
    return scope


class DistanceBAODerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def setUp(self):
        self.k = np.geomspace(.035, .2, 13)
        self.mu = np.array([-.9, -.4, .4, .9])
        self.K, self.U = np.meshgrid(self.k, self.mu)
        self.c = dict(omega_M_0=.316, omega_lambda_0=.684, h=.67,
                      sigma_8=.834, w0=-1., wa=0., gamma=.55,
                      ns=.962, k_piv=.05, z=1., r=2000., rnu=3000.,
                      f=np.float64(.8), D=np.float64(.6), bHI=2., btot=2.,
                      Tb=.15, b_1=0., k0_bias=.1, aperp=1., apar=1.,
                      A=1., sigma_nl=7., pk_nobao=lambda k: 1e4*(k/.1)**-1.2,
                      fbao=lambda k: .02*np.sin(100*k),
                      dfbao_dk=lambda k: 2*np.cos(100*k))

    def derivatives(self, **switches):
        use = dict(f_rsd=True, f_growthfactor=False, alpha_all=False,
                   alpha_volume=False, alpha_rsd_angle=False,
                   alpha_rsd_shift=False, alpha_bao_shift=False,
                   alpha_pk_shift=False)
        use.update(switches)
        with np.errstate():
            values, names = self.model['fisher_integrands'](
                self.k, self.mu, self.c, dict(use=use))
        return np.array([values[names.index(p)] for p in ('aperp', 'apar')])

    def log_covariance(self, aperp=1., apar=1., component='all'):
        # Inverse-distance convention: alpha_perp=r_fid/r, alpha_par=H/H_fid.
        kp = aperp*self.K*np.sqrt(1-self.U**2)
        kz = apar*self.K*self.U
        shifted_k = np.sqrt(kp**2+kz**2)
        k = shifted_k if component == 'all' else self.K
        u2 = (kz/k)**2 if component == 'all' else self.U**2
        smooth_k = shifted_k if component in ('all', 'smooth') else self.K
        bao_k = shifted_k if component in ('all', 'bao') else self.K
        c = self.c
        signal = ((c['bHI']+c['f']*u2)**2*np.exp(-u2*(k*c['sigma_nl'])**2)
                  * c['D']**2*c['pk_nobao'](smooth_k)*(1+c['A']*c['fbao'](bao_k))
                  * c['Tb']**2/(c['r']**2*c['rnu']))
        if component == 'all':
            signal *= aperp**2*apar
        return np.log(signal+1e-9)

    def check_finite_difference(self, component, **switches):
        for amplitude in (0., .7, 1.):
            self.c['A'] = amplitude
            actual = self.derivatives(**switches)
            for i, parameter in enumerate(('aperp', 'apar')):
                for step in (1e-5, 5e-6):
                    with self.subTest(A=amplitude, parameter=parameter, step=step):
                        plus = self.log_covariance(component=component, **{parameter: 1+step})
                        minus = self.log_covariance(component=component, **{parameter: 1-step})
                        np.testing.assert_allclose(actual[i], (plus-minus)/(2*step),
                                                   rtol=2e-6, atol=1e-8)

    def test_full_distance_derivatives(self):
        self.check_finite_difference('all', alpha_all=True)

    def test_bao_only_derivatives(self):
        self.check_finite_difference('bao', alpha_bao_shift=True)

    def test_smooth_only_derivatives(self):
        self.check_finite_difference('smooth', alpha_pk_shift=True)

    def test_split_terms_sum_to_full_derivative(self):
        full = self.derivatives(alpha_all=True)
        split = self.derivatives(alpha_volume=True, alpha_rsd_angle=True,
                                 alpha_rsd_shift=True, alpha_bao_shift=True,
                                 alpha_pk_shift=True)
        np.testing.assert_allclose(split, full, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
