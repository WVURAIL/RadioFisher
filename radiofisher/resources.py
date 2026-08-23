"""Bundled experiment data and explicit markers for unavailable resources."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


BASELINE_DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "array_config"


class UnsupportedExperimentDataError(RuntimeError):
    """Raised when a forecast uses a preset whose required data is absent."""


@dataclass(frozen=True)
class UnavailableExperimentData:
    """Marker for a historical input that is not distributed with 1.0."""

    preset: str
    key: str
    reference: str

    def error_message(self):
        return (
            "experiment preset %r requires unavailable %s data %r; replace "
            "expt[%r] with a verified path or callable before forecasting"
            % (self.preset, self.key, self.reference, self.key)
        )

    def __call__(self, *args, **kwargs):
        raise UnsupportedExperimentDataError(self.error_message())


def bundled_baseline_path(reference):
    """Return the bundled baseline path matching *reference*, if present."""

    candidate = BASELINE_DATA_DIRECTORY / Path(reference).name
    return candidate if candidate.is_file() else None


def validate_experiment_resources(expt):
    """Reject unresolved resource markers in an experiment dictionary."""

    if not isinstance(expt, Mapping):
        raise TypeError("expt must be a mapping")
    for value in expt.values():
        if isinstance(value, UnavailableExperimentData):
            raise UnsupportedExperimentDataError(value.error_message())
