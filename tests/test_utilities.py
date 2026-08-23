import copy
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import radiofisher
from radiofisher import baofisher
from radiofisher import experiments


def test_figure_of_merit_accepts_numpy_covariance():
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
    assert baofisher.figure_of_merit(0, 1, None, cov=covariance) == pytest.approx(
        1.0 / np.sqrt(35.0)
    )


def test_plot_ellipse_uses_four_value_ellipse_contract():
    figure, axis = plt.subplots()
    baofisher.plot_ellipse(np.eye(2), 0, 1, (0.0, 0.0), ("x", "y"), ax=axis)
    assert len(axis.patches) == 2
    plt.close(figure)


def test_triangle_plot_uses_four_value_ellipse_contract(monkeypatch):
    monkeypatch.setattr(baofisher.P, "show", lambda: None)
    baofisher.triangle_plot([0.0, 0.0], np.eye(2), ["x", "y"])
    plt.close("all")


def test_add_fisher_list_handles_nonexpanding_and_excluded_parameters():
    matrices = [np.eye(2), np.diag([2.0, 3.0])]
    labels = [["a", "b"], ["a", "b"]]

    total, total_labels = baofisher.add_fisher_list(matrices, labels)
    reduced, reduced_labels = baofisher.add_fisher_list(
        matrices, labels, exclude=["b"]
    )

    assert total == pytest.approx(np.diag([3.0, 4.0]))
    assert total_labels == ["a", "b"]
    assert reduced == pytest.approx(np.array([[3.0]]))
    assert reduced_labels == ["a"]


def test_add_fisher_list_can_expand_to_union_of_parameters():
    total, labels = baofisher.add_fisher_list(
        [np.array([[2.0]]), np.array([[3.0]])],
        [["a"], ["b"]],
        expand=True,
    )

    assert labels == ["a", "b"]
    assert total == pytest.approx(np.diag([2.0, 3.0]))


def test_split_width_binning_uses_integer_counts():
    edges, centers = baofisher.zbins_split_width(
        experiments.CHIME, dz=(0.1, 0.3), zsplit=2.0
    )
    assert len(edges) == len(centers) + 1
    assert np.all(np.diff(edges) > 0.0)


def test_overlap_key_is_removed_using_value_equality():
    dynamic_overlap_key = "".join(["over", "lap"])
    first = {"survey_numax": 800.0, "survey_dnutot": 400.0, "nu_line": 1420.0}
    second = {"survey_numax": 900.0, "survey_dnutot": 400.0, "nu_line": 1420.0}
    experiment = {dynamic_overlap_key: (first, second), "Sarea": 1.0}

    result = baofisher.overlapping_expts(experiment)

    assert "overlap" not in result


def test_octave_preset_has_one_unambiguous_supported_name():
    assert not hasattr(experiments, "MID_B1_Octave")
    assert not hasattr(experiments, "MID_B1_Octave_Legacy")
    assert experiments.MID_B1_Octave_Updated["survey_numax"] == 1015.0


def test_hybrid_noise_path_uses_second_system_temperature():
    experiment = copy.deepcopy(experiments.MID_B1_Octave_Updated)
    experiment["dnutot"] = 10.0
    cosmology = {
        "z": experiment["nu_line"] / 800.0 - 1.0,
        "aperp": 1.0,
        "apar": 1.0,
        "r": 2000.0,
        "rnu": 3000.0,
        "ns": 0.96,
    }

    noise = baofisher.Cnoise(
        np.array([100.0]), np.array([100.0]), cosmology, experiment
    )

    assert noise.shape == (1,)
    assert np.all(np.isfinite(noise))


def test_fully_excised_frequency_band_returns_infinite_noise_sentinel():
    experiment = copy.deepcopy(experiments.MID_B1_Octave_Updated)
    experiment["dnutot"] = 10.0
    experiment["noise_freq_weight"] = lambda nu: np.full(nu.shape, np.nan)
    cosmology = {
        "z": experiment["nu_line"] / 800.0 - 1.0,
        "aperp": 1.0,
        "apar": 1.0,
        "r": 2000.0,
        "rnu": 3000.0,
        "ns": 0.96,
    }

    noise = baofisher.Cnoise(
        np.array([100.0, 200.0]),
        np.array([100.0, 200.0]),
        cosmology,
        experiment,
    )

    assert np.all(noise == baofisher.INF_NOISE)


def test_n_im_requires_explicit_volume_context():
    with pytest.raises(ValueError, match="explicit zmin, zmax, and cosmo_fns"):
        baofisher.n_IM(
            np.array([0.01, 0.02]),
            np.array([-1.0, 1.0]),
            {},
            {},
        )


def test_n_im_returns_density_and_volume_with_explicit_context(monkeypatch):
    monkeypatch.setattr(
        baofisher, "Cnoise", lambda q, y, cosmo, expt: np.ones_like(q)
    )
    monkeypatch.setattr(
        baofisher, "Cfg", lambda q, y, cosmo, expt: np.zeros_like(q)
    )
    cosmology = {"r": 10.0, "rnu": 20.0, "Tb": 2.0}
    experiment = {"Sarea": 0.5}
    functions = (
        lambda z: np.full_like(z, 100.0),
        lambda z: 10.0 * z,
        lambda z: np.ones_like(z),
        lambda z: np.ones_like(z),
    )

    density, volume = baofisher.n_IM(
        np.linspace(0.01, 0.1, 9),
        np.linspace(-1.0, 1.0, 11),
        cosmology,
        experiment,
        zmin=0.0,
        zmax=1.0,
        cosmo_fns=functions,
    )

    assert density > 0.0
    assert volume == pytest.approx(baofisher.C * experiment["Sarea"] / 3.0)


def test_empty_fisher_matrix_placeholder_is_removed():
    assert not hasattr(baofisher, "FisherMatrix")
    assert "FisherMatrix" not in radiofisher.__all__


def test_obsolete_planck_file_helpers_are_removed():
    assert not hasattr(radiofisher.euclid, "add_planck_prior")
    assert not hasattr(radiofisher.euclid, "add_detf_planck_prior")


def test_logpk_derivative_preserves_numpy_error_policy_without_warning():
    original_policy = np.geterr().copy()

    def bounded_power(k):
        return np.where((k >= 0.1) & (k <= 1.0), k**2, 0.0)

    with warnings.catch_warnings(record=True) as warnings_seen:
        warnings.simplefilter("always")
        derivative = baofisher.logpk_derivative(
            bounded_power, np.array([0.09999999, 0.2, 1.00000001])
        )

    assert not warnings_seen
    assert np.all(np.isfinite(derivative))
    assert np.geterr() == original_policy
