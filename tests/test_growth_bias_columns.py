"""Check fs8/bs8 labels against finite differences of the signal covariance.

Run: python -m unittest discover -s tests -p test_growth_bias_columns.py -v
Requires NumPy. Set RADIOFISHER_SOURCE to test a different baofisher.py.

Load only the required, unmodified function bodies to avoid obsolete package
import dependencies (notably scipy.misc.derivative) and a CAMB installation.
Constant noise and an analytic power spectrum isolate the derivative check
from the instrument model.
"""
import ast
import copy
import os
from pathlib import Path
import unittest

import numpy as np


def load_model():
    source = Path(os.environ.get(
        'RADIOFISHER_SOURCE',
        str(Path(__file__).resolve().parents[1] / 'radiofisher' / 'baofisher.py'),
    ))
    required = {'Ez', 'inverse_interpfn', 'logpk_derivative',
                'Csignal', 'fisher_integrands'}
    scope = dict(np=np, copy=copy, C=3e5, RSD_FUNCTION='kaiser')
    scope['Cnoise'] = lambda q, y, c, e: np.full_like(q, 1e-9)
    scope['Cfg'] = lambda q, y, c, e: np.zeros_like(q)
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in required]
    if {node.name for node in selected} != required:
        raise RuntimeError('Required functions missing from ' + str(source))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), scope)
    return scope


class GrowthBiasColumns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def check_column(self, name, parameter):
        # Several angles distinguish growth (mu^2 dependence) from bias.
        k = np.geomspace(.035, .2, 13)
        mu = np.array([-.9, -.4, .4, .9])
        K, U = np.meshgrid(k, mu)
        expt = dict(use=dict(f_rsd=True, f_growthfactor=False, alpha_all=True))
        for growth, bias, D, sigma8 in [(.8, 2., .6, .834), (.95, 1.3, .4, .812)]:
            with self.subTest(f=growth, bias=bias, D=D, sigma8=sigma8):
                c = dict(omega_M_0=.316, omega_lambda_0=.684, h=.67,
                         sigma_8=sigma8, w0=-1., wa=0., gamma=.55,
                         ns=.962, k_piv=.05, z=1., r=2000., rnu=3000.,
                         f=np.float64(growth), D=np.float64(D),
                         bHI=bias, btot=bias, Tb=.15, b_1=0., k0_bias=.1,
                         aperp=1., apar=1., A=1., sigma_nl=7.,
                         pk_nobao=lambda x: 1e4*(x/.1)**-1.2,
                         fbao=lambda x: .02*np.sin(100*x),
                         dfbao_dk=lambda x: 2*np.cos(100*x))
                q, y = c['r']*K*np.sqrt(1-U**2), c['rnu']*K*U
                # Restore NumPy's error handling after the legacy helper call.
                with np.errstate():
                    derivs, names = self.model['fisher_integrands'](k, mu, c, expt)
                actual = derivs[names.index(name)]
                # fs8 = f*D*sigma8 and bs8 = b*D*sigma8, with D,sigma8 fixed.
                for step in (1e-5, 5e-6):
                    plus, minus = copy.deepcopy(c), copy.deepcopy(c)
                    delta = step/(D*sigma8)
                    plus[parameter] += delta
                    minus[parameter] -= delta
                    if parameter == 'bHI':
                        plus['btot'], minus['btot'] = plus['bHI'], minus['bHI']
                    signal = self.model['Csignal']
                    expected = (np.log(signal(q, y, plus, expt)+1e-9)
                                - np.log(signal(q, y, minus, expt)+1e-9))/(2*step)
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8,
                                               err_msg=name+' must match its named physical derivative')

    def test_fs8_matches_growth_finite_difference(self):
        self.check_column('fs8', 'f')

    def test_bs8_matches_bias_finite_difference(self):
        self.check_column('bs8', 'bHI')


if __name__ == '__main__':
    unittest.main()
