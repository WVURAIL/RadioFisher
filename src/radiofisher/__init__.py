"""Forecast 21 cm BAO measurements with a cylinder interferometer.

This package retains the RadioFisher path used for that calculation.
See baofisher.py for the call sequence and chime.py for the CHIME inputs.
"""
__version__ = "1.1.0"

from . import chime
from .baofisher import (
    Cfg,
    Cnoise,
    Csignal,
    DirectBAODerivative,
    background_evolution_splines,
    direct_fbao_derivative,
    fisher,
    fisher_integrands,
    integrate_fisher_elements,
    interferometer_response,
    load_power_spectrum,
    prepare_bin,
)
from .results import FisherResult, ProjectedResult, summarize_fisher

__all__ = [
    "Cfg",
    "Cnoise",
    "Csignal",
    "DirectBAODerivative",
    "FisherResult",
    "ProjectedResult",
    "background_evolution_splines",
    "chime",
    "direct_fbao_derivative",
    "fisher",
    "fisher_integrands",
    "integrate_fisher_elements",
    "interferometer_response",
    "load_power_spectrum",
    "prepare_bin",
    "summarize_fisher",
]
