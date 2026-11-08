"""Instrument geometry and mode cuts shared by noise and residual models."""

import numpy as np

from .units import FORECAST_LIGHT_SPEED_KM_S


def wavelength_m(frequency_mhz):
    """Wavelength in meters, using the forecast's speed-of-light convention."""
    return FORECAST_LIGHT_SPEED_KM_S * 1000 / (frequency_mhz * 1e6)


MODES = ("icyl", "idish")   # cylinder and dish interferometers


def collecting_area_m2(expt):
    """Effective collecting area per receiver, in square meters."""
    if expt["mode"] == "idish":
        return expt.get("effic", 0.7) * 0.25 * np.pi * expt["Ddish"] ** 2
    return expt.get("effic", 0.7) * expt["Ncyl"] * expt["cyl_area"] / expt["Ndish"]


def beam_width(wavelength, expt):
    """Square root of the field of view in steradians.

    A cylinder sees about 90 deg x lambda/D, and a dish about (lambda/D)^2
    (Bull et al. 2015).
    """
    if expt["mode"] == "idish":
        return wavelength / expt["Ddish"]
    return np.sqrt(0.5 * np.pi * wavelength / expt["Ddish"])


def mode_limits(cosmo, expt):
    """Foreground k_parallel minimum and nonlinear |k| maximum, in Mpc^-1."""
    foreground = 2 * np.pi * expt["nu_line"] / (expt["survey_dnutot"] * cosmo["rnu"])
    foreground *= expt.get("kfg_fac", 1.0)
    nonlinear = expt["k_nl0"] * (1 + cosmo["z"]) ** (2 / (2 + cosmo["ns"]))
    return foreground, nonlinear
