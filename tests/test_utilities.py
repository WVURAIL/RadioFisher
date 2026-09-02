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


def test_delay_cut_removes_only_modes_below_kpar_min():
    """expt['kpar_min_fn'] excises low-delay modes exactly like the
    foreground cut: infinite noise below the threshold, untouched above."""
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
    # kpar = y / (apar * rnu), so these y values straddle kpar = 0.05.
    y = np.array([0.03, 0.07]) * 3000.0
    q = np.array([100.0, 100.0])

    uncut = baofisher.Cnoise(q, y, cosmology, experiment)
    experiment["kpar_min_fn"] = lambda z: 0.05
    cut = baofisher.Cnoise(q, y, cosmology, experiment)

    assert np.all(np.isfinite(uncut))
    assert cut[0] == baofisher.INF_NOISE
    assert cut[1] == uncut[1]


def test_zero_delay_cut_is_a_no_op():
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
    y = np.array([0.03, 0.07]) * 3000.0
    q = np.array([100.0, 100.0])

    uncut = baofisher.Cnoise(q, y, cosmology, experiment)
    experiment["kpar_min_fn"] = lambda z: 0.0

    assert np.array_equal(
        baofisher.Cnoise(q, y, cosmology, experiment), uncut)


def test_delay_cut_is_evaluated_at_the_bin_redshift():
    experiment = copy.deepcopy(experiments.MID_B1_Octave_Updated)
    experiment["dnutot"] = 10.0
    z = experiment["nu_line"] / 800.0 - 1.0
    cosmology = {
        "z": z, "aperp": 1.0, "apar": 1.0,
        "r": 2000.0, "rnu": 3000.0, "ns": 0.96,
    }
    seen = []

    def kpar_min_fn(redshift):
        seen.append(redshift)
        return 0.0

    experiment["kpar_min_fn"] = kpar_min_fn
    baofisher.Cnoise(np.array([100.0]), np.array([100.0]), cosmology,
                     experiment)

    assert seen == [z]


def _signal_fixture():
    """A minimal cosmology for Csignal on a two-point (q, y) grid."""
    experiment = copy.deepcopy(experiments.MID_B1_Octave_Updated)
    cosmology = {
        "z": 1.0, "aperp": 1.0, "apar": 1.0, "r": 2000.0, "rnu": 3000.0,
        "f": 0.8, "D": 0.6, "btot": 1.5, "sigma_nl": 7.0, "A": 1.0,
        "Tb": 0.1, "fbao": lambda k: 0.0 * k, "pk_nobao": lambda k: 1e4 / k,
    }
    q = np.array([100.0, 100.0])
    y = np.array([0.03, 0.07]) * 3000.0   # kpar = 0.03, 0.07
    return experiment, cosmology, q, y


def test_signal_transfer_scales_only_the_signal_where_asked():
    experiment, cosmology, q, y = _signal_fixture()
    bare = baofisher.Csignal(q, y, cosmology, experiment)
    seen = []

    def transfer(kpar, z):
        seen.append((np.array(kpar), z))
        return np.where(kpar < 0.05, 0.25, 1.0)

    experiment["kpar_transfer_fn"] = transfer
    soft = baofisher.Csignal(q, y, cosmology, experiment)

    assert soft[0] == pytest.approx(0.25 * bare[0])
    assert soft[1] == pytest.approx(bare[1])
    assert seen[0][1] == 1.0
    assert seen[0][0] == pytest.approx([0.03, 0.07])
    # noise is untouched by a signal transfer
    experiment["dnutot"] = 10.0
    cosmology["ns"] = 0.96
    with_transfer = baofisher.Cnoise(q, y, cosmology, experiment)
    del experiment["kpar_transfer_fn"]
    assert np.array_equal(baofisher.Cnoise(q, y, cosmology, experiment),
                          with_transfer)


def test_signal_transfer_rejects_out_of_range_or_misshapen_values():
    experiment, cosmology, q, y = _signal_fixture()
    experiment["kpar_transfer_fn"] = lambda kpar, z: 1.5 * np.ones_like(kpar)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        baofisher.Csignal(q, y, cosmology, experiment)
    experiment["kpar_transfer_fn"] = lambda kpar, z: np.ones(3)
    with pytest.raises(ValueError, match="shape"):
        baofisher.Csignal(q, y, cosmology, experiment)


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
