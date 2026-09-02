import numpy as np
import pytest
import scipy.interpolate

from radiofisher.extensions import (
    DELAY_TRANSITION_FACTOR,
    delay_cut_kpar_min,
    frequency_noise_penalty,
    validate_experiment_extensions,
    validate_kpar_min,
    validate_volume_fraction,
)
from radiofisher.units import C, PI


FREQUENCIES = np.linspace(400.0, 800.0, 9)


def test_scalar_frequency_weight_is_broadcast():
    assert frequency_noise_penalty(lambda nu: 0.5, FREQUENCIES) == pytest.approx(2.0)


def test_exact_shape_weight_supports_both_modes():
    weights = np.linspace(0.5, 1.0, FREQUENCIES.size)
    weight_fn = lambda nu: weights

    assert frequency_noise_penalty(weight_fn, FREQUENCIES, "invvar") == pytest.approx(
        1.0 / np.mean(weights)
    )
    assert frequency_noise_penalty(weight_fn, FREQUENCIES, "fourier") == pytest.approx(
        np.mean(1.0 / weights)
    )


def test_nan_means_excised_and_all_nan_means_infinite_noise():
    weights = np.ones(FREQUENCIES.shape)
    weights[0] = np.nan
    assert frequency_noise_penalty(lambda nu: weights, FREQUENCIES) == 1.0
    assert np.isinf(
        frequency_noise_penalty(
            lambda nu: np.full(nu.shape, np.nan), FREQUENCIES
        )
    )


def test_wrong_frequency_weight_shape_is_rejected():
    with pytest.raises(ValueError, match="returned shape"):
        frequency_noise_penalty(lambda nu: np.ones(3), FREQUENCIES)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, np.inf, -np.inf])
def test_invalid_frequency_weights_are_rejected(value):
    with pytest.raises(ValueError):
        frequency_noise_penalty(lambda nu: value, FREQUENCIES)


def test_invalid_frequency_mode_is_rejected():
    with pytest.raises(ValueError, match="noise_freq_mode"):
        frequency_noise_penalty(lambda nu: 1.0, FREQUENCIES, "average")


@pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
def test_volume_fraction_accepts_closed_unit_interval(value):
    assert validate_volume_fraction(value) == value


@pytest.mark.parametrize("value", [True, np.bool_(False), [0.5]])
def test_volume_fraction_rejects_booleans_and_non_scalars(value):
    with pytest.raises(TypeError):
        validate_volume_fraction(value)


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf])
def test_volume_fraction_rejects_out_of_range_or_nonfinite_values(value):
    with pytest.raises(ValueError):
        validate_volume_fraction(value)


def test_experiment_extension_validation_fails_closed():
    with pytest.raises(TypeError, match="callable"):
        validate_experiment_extensions({"noise_freq_weight": [1.0]})
    with pytest.raises(ValueError, match="noise_freq_mode"):
        validate_experiment_extensions({"noise_freq_mode": "unknown"})
    with pytest.raises(TypeError, match="kpar_min_fn"):
        validate_experiment_extensions({"kpar_min_fn": 0.3})


# H(z=1.16) for the Planck-2018 fiducial of the CHIME Overview forecasts
# (h = 0.6732, Omega_m = 0.3158), in km/s/Mpc.
PLANCK2018_H_AT_1P16 = 132.378728823083
PLANCK2018_H = 0.6732


def test_delay_cut_reproduces_the_published_chime_window():
    """200 ns cut + 280 ns mask -> k_par,min = 0.35 h/Mpc at z = 1.16.

    This is the window the z ~ 1 auto-spectrum detection actually starts
    from (Amiri et al. 2025), and it is the anchor for every other delay.
    """
    kpar_min_fn = delay_cut_kpar_min(
        200e-9, lambda z: PLANCK2018_H_AT_1P16, 1420.406)

    kpar_min = kpar_min_fn(1.16)

    assert kpar_min / PLANCK2018_H == pytest.approx(0.351, abs=5e-4)


def test_delay_cut_matches_its_closed_form_and_scales_linearly():
    hubble_fn = lambda z: 70.0 * np.sqrt(0.3 * (1.0 + z)**3 + 0.7)
    kpar_min_fn = delay_cut_kpar_min(100e-9, hubble_fn, 1420.406)

    expected = (DELAY_TRANSITION_FACTOR * 100e-9 * 2.0 * PI * 1420.406e6
                * hubble_fn(1.5) / (C * 2.5**2))
    assert kpar_min_fn(1.5) == pytest.approx(expected)

    doubled = delay_cut_kpar_min(200e-9, hubble_fn, 1420.406)
    assert doubled(1.5) == pytest.approx(2.0 * kpar_min_fn(1.5))

    untransitioned = delay_cut_kpar_min(
        100e-9, hubble_fn, 1420.406, transition=1.0)
    assert untransitioned(1.5) == pytest.approx(
        kpar_min_fn(1.5) / DELAY_TRANSITION_FACTOR)


def test_delay_cut_accepts_a_spline_valued_hubble_rate():
    """SciPy interp1d returns a 0-d array, not a float."""
    spline = scipy.interpolate.interp1d([0.0, 3.0], [70.0, 350.0])
    kpar_min_fn = delay_cut_kpar_min(50e-9, spline, 1420.406)

    assert isinstance(kpar_min_fn(1.0), float)
    assert kpar_min_fn(1.0) > 0.0


@pytest.mark.parametrize("value", [True, np.bool_(False), [0.1, 0.2], "0.1"])
def test_kpar_min_rejects_booleans_and_non_scalars(value):
    with pytest.raises(TypeError):
        validate_kpar_min(value)


@pytest.mark.parametrize("value", [-1e-3, np.nan, np.inf])
def test_kpar_min_rejects_negative_or_nonfinite_values(value):
    with pytest.raises(ValueError):
        validate_kpar_min(value)


def test_delay_cut_rejects_malformed_inputs():
    with pytest.raises(TypeError, match="hubble_fn"):
        delay_cut_kpar_min(100e-9, 70.0, 1420.406)
    with pytest.raises(ValueError, match="tau_cut_s"):
        delay_cut_kpar_min(-1e-9, lambda z: 70.0, 1420.406)
    with pytest.raises(ValueError, match="transition"):
        delay_cut_kpar_min(100e-9, lambda z: 70.0, 1420.406, transition=0.0)
    with pytest.raises(ValueError, match="nu_line_mhz"):
        delay_cut_kpar_min(100e-9, lambda z: 70.0, 0.0)
    with pytest.raises(ValueError, match="H"):
        delay_cut_kpar_min(100e-9, lambda z: -70.0, 1420.406)(1.0)
