"""Forecast 21 cm BAO measurements with a cylinder interferometer.

This package is the part of RadioFisher (Bull et al. 2015) used by the CHIME
Overview forecast. See baofisher.py for the call sequence and chime.py for the
CHIME inputs.
"""
__version__ = "1.0.0"

from . import chime
from .baofisher import (
    Cfg,
    Cnoise,
    Csignal,
    background_evolution_splines,
    fisher,
    fisher_integrands,
    integrate_fisher_elements,
    interferometer_response,
    load_power_spectrum,
    prepare_bin,
)

__all__ = [
    "Cfg",
    "Cnoise",
    "Csignal",
    "background_evolution_splines",
    "chime",
    "fisher",
    "fisher_integrands",
    "integrate_fisher_elements",
    "interferometer_response",
    "load_power_spectrum",
    "prepare_bin",
]
