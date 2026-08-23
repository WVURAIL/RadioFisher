import importlib.util
from pathlib import Path

import pytest

from radiofisher import experiments
from radiofisher import baofisher
from radiofisher.extensions import validate_experiment_extensions
from radiofisher.resources import (
    UnavailableExperimentData,
    UnsupportedExperimentDataError,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_active_tree_contains_no_historical_root_frontends():
    assert not list(REPOSITORY_ROOT.glob("*.py"))
    assert not (REPOSITORY_ROOT / "plotting").exists()


def test_chime_overview_configuration_uses_tracked_asbuilt_baseline():
    module_path = REPOSITORY_ROOT / "chime2021" / "experiments_CHIME.py"
    spec = importlib.util.spec_from_file_location("experiments_CHIME", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    baseline = REPOSITORY_ROOT / "chime2021" / module.CHIME["n(x)"]
    assert module.CHIME["Ncyl"] == 4
    assert module.CHIME["Ndish"] == 1024
    assert baseline.is_file()


def test_supported_experiment_inventory_has_no_missing_resources():
    public_presets = {
        name
        for name, value in vars(experiments).items()
        if not name.startswith("_")
        and isinstance(value, dict)
        and "mode" in value
    }
    inventory = set(experiments.SUPPORTED_EXPERIMENT_PRESETS) | set(
        experiments.UNSUPPORTED_EXPERIMENT_PRESETS
    )

    assert public_presets == inventory
    assert not (
        set(experiments.SUPPORTED_EXPERIMENT_PRESETS)
        & set(experiments.UNSUPPORTED_EXPERIMENT_PRESETS)
    )
    for preset in experiments.SUPPORTED_EXPERIMENT_PRESETS.values():
        reference = preset.get("n(x)")
        assert not isinstance(reference, UnavailableExperimentData)
        if isinstance(reference, str):
            assert Path(reference).is_file()


def test_unavailable_experiment_data_fails_explicitly_until_replaced():
    assert "CHIME" in experiments.UNSUPPORTED_EXPERIMENT_PRESETS
    for name, marker in experiments.UNSUPPORTED_EXPERIMENT_PRESETS.items():
        assert isinstance(marker, UnavailableExperimentData)
        with pytest.raises(UnsupportedExperimentDataError, match=name):
            validate_experiment_extensions(getattr(experiments, name))

    configured = dict(experiments.CHIME)
    configured["n(x)"] = lambda x: x
    validate_experiment_extensions(configured)


def test_forecast_boundary_rejects_unavailable_preset_before_other_work():
    with pytest.raises(UnsupportedExperimentDataError, match="CHIME"):
        baofisher.fisher(0.8, 0.9, {}, experiments.CHIME, ())
