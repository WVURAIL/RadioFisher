"""Versioned fiducials; observational parameters and H I models are separate."""
from copy import deepcopy
from types import MappingProxyType

from . import experiments
from .baofisher import NEUTRINO_MASS_DENSITY_EV

DEFAULT_COSMOLOGY = "cmbspa2026"
# Omori et al. 2026, arXiv:2608.31136v1, Table 8, fourth numerical
# column: kappa_GPA + CMB_SPA + BAO_DESI. All values are posterior means
# from this one flat-LambdaCDM combination, not a mixture of datasets.
CMBSPA2026 = MappingProxyType({
    "h": 0.6832, "ombh2": 0.022506, "omch2": 0.11755,
    "sigma_8": 0.8117, "ns": 0.9742, "mnu": 0.06,
})
CMBSPA2026_SOURCE = MappingProxyType({
    "url": "https://arxiv.org/html/2608.31136v1#S7.T8",
    "arxiv": "2608.31136v1", "date": "2026-08-31", "table": "8",
    "combination": "kappa_GPA + CMB_SPA + DESI DR2 BAO",
    "statistic": "posterior means", "model": "flat LambdaCDM",
})


def get_cosmology(name=DEFAULT_COSMOLOGY):
    """Return an independent fiducial dictionary with explicit total matter.

    ``experiments.cosmo`` retains the historical Planck 2013 comparison.
    New forecasts should use this named factory. The 2026 density fractions
    are derived from the physical densities and h, rather than independently
    rounded Omega_m values. N_eff=3.046 retains the backend/cache convention.
    """
    c = deepcopy(experiments.cosmo)
    if name == "planck2013":
        return c
    if name != DEFAULT_COSMOLOGY:
        raise ValueError(f"unknown fiducial cosmology {name!r}; choose cmbspa2026 or planck2013")
    c.update(CMBSPA2026)
    c['omnuh2'] = c['mnu'] / NEUTRINO_MASS_DENSITY_EV
    c['omega_b_0'] = c['ombh2'] / c['h']**2
    c['omega_cdm_0'] = c['omch2'] / c['h']**2
    c['omega_nu_0'] = c['omnuh2'] / c['h']**2
    c['omega_M_0'] = c['omega_b_0'] + c['omega_cdm_0'] + c['omega_nu_0']
    c['omega_lambda_0'] = 1 - c['omega_M_0']
    c.update(w0=-1.0, wa=0.0, omega_M_0_includes_neutrinos=True)
    c['fiducial_cosmology'] = name
    c['fiducial_source'] = dict(CMBSPA2026_SOURCE)
    return c
