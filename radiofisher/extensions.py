"""Validation and calculations for optional experiment extensions."""

from collections.abc import Mapping
from numbers import Real

import numpy as np

from .resources import validate_experiment_resources
from .units import C, PI


NOISE_FREQUENCY_SAMPLES = 2049
DEFAULT_NOISE_FREQ_MODE = "invvar"
NOISE_FREQ_MODES = frozenset({"invvar", "fourier"})
MIN_VOLUME_FRACTION = 0.0
MAX_VOLUME_FRACTION = 1.0

# Ratio of the delay actually retained to the nominal high-pass filter delay.
# A DAYENU-style filter at tau_cut does not recover instantly above it, so an
# analysis masks the transition zone as well. CHIME's z~1 auto-spectrum cuts
# at 200 ns and keeps nothing below 280 ns (Amiri et al. 2025,
# arXiv:2511.19620), i.e. a factor of 1.4.
DELAY_TRANSITION_FACTOR = 1.4


def _finite_scalar(value, name):
    """Return *value* as a finite float, rejecting booleans and arrays."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("%s must be a real scalar" % name)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def validate_volume_fraction(value):
    """Validate and return a surviving survey-volume fraction in ``[0, 1]``."""

    fraction = _finite_scalar(value, "vol_frac")
    if not MIN_VOLUME_FRACTION <= fraction <= MAX_VOLUME_FRACTION:
        raise ValueError("vol_frac must be between 0 and 1 inclusive")
    return fraction


def _finite_scalar_value(value, name):
    """Like ``_finite_scalar``, but also accept a 0-d/size-1 NumPy value.

    SciPy's ``interp1d`` returns a 0-d array rather than a Python float, so
    an H(z) spline is not an instance of ``numbers.Real``.  Sequences with
    more than one element are still rejected.
    """

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a real scalar" % name)
    if isinstance(value, Real):
        return _finite_scalar(value, name)
    array = np.asarray(value)
    if (array.size != 1
            or not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)
            or np.issubdtype(array.dtype, np.bool_)):
        raise TypeError("%s must be a real scalar" % name)
    result = float(array.reshape(()))
    if not np.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def validate_kpar_min(value):
    """Validate and return a minimum retained ``|k_par|`` in Mpc^-1."""

    kpar_min = _finite_scalar_value(value, "kpar_min")
    if kpar_min < 0.0:
        raise ValueError("kpar_min must not be negative")
    return kpar_min


def delay_cut_kpar_min(tau_cut_s, hubble_fn, nu_line_mhz,
                       transition=DELAY_TRANSITION_FACTOR):
    """Return ``k_par,min(z)`` [Mpc^-1] for a hard delay-domain foreground cut.

    A high-pass delay filter at ``tau_cut`` removes every line-of-sight mode
    whose delay is smaller than the retained delay ``transition * tau_cut``.
    Delay and radial wavenumber are related by the same 21 cm line mapping
    used for ``rnu = c (1+z)^2 / H(z)`` elsewhere in this package, so

        k_par,min(z) = transition * tau_cut * 2 pi nu_21 H(z) / (c (1+z)^2),

    with ``nu_21`` in Hz and ``tau_cut`` in seconds.

    ``hubble_fn`` is H(z) in km/s/Mpc -- the first element of the
    ``cosmo_fns`` tuple returned by ``background_evolution_splines``.
    ``nu_line_mhz`` is the experiment's ``nu_line``. The returned callable
    takes a redshift and is suitable for ``expt['kpar_min_fn']``.

    This is the hard cut: modes below ``k_par,min`` are deleted outright,
    which is what an analysis that masks its filter's transition zone
    actually does. ``delay_transfer_fn`` is the soft version, built from
    the filter's measured residual response as a function of tau/tau_cut.
    """

    tau_cut_s = _finite_scalar(tau_cut_s, "tau_cut_s")
    if tau_cut_s < 0.0:
        raise ValueError("tau_cut_s must not be negative")
    transition = _finite_scalar(transition, "transition")
    if transition <= 0.0:
        raise ValueError("transition must be positive")
    nu_line_mhz = _finite_scalar(nu_line_mhz, "nu_line_mhz")
    if nu_line_mhz <= 0.0:
        raise ValueError("nu_line_mhz must be positive")
    if not callable(hubble_fn):
        raise TypeError("hubble_fn must be callable")

    # nu_line is quoted in MHz; the delay-to-wavenumber map needs Hz.
    prefactor = transition * tau_cut_s * 2.0 * PI * (nu_line_mhz * 1e6) / C

    def kpar_min_fn(z):
        redshift = _finite_scalar_value(z, "z")
        if redshift < 0.0:
            raise ValueError("z must not be negative")
        hubble = _finite_scalar_value(hubble_fn(redshift), "H(z)")
        if hubble <= 0.0:
            raise ValueError("H(z) must be positive")
        return prefactor * hubble / (1.0 + redshift)**2.0

    return kpar_min_fn


def validate_kpar_transfer(values, shape):
    """Validate a signal transfer array: finite, in ``[0, 1]``, of ``shape``."""

    transfer = np.asarray(values, dtype=float)
    if transfer.shape != tuple(shape):
        raise ValueError(
            "kpar_transfer_fn returned shape %s; expected %s"
            % (transfer.shape, tuple(shape)))
    if not np.all(np.isfinite(transfer)):
        raise ValueError("kpar_transfer_fn must return finite values")
    if np.any(transfer < 0.0) or np.any(transfer > 1.0):
        raise ValueError("kpar_transfer_fn values must be in [0, 1]")
    return transfer


def delay_transfer_fn(tau_cut_s, hubble_fn, nu_line_mhz, delay_ratio,
                      response, plateau=None):
    """Return ``T(|k_par|, z)``: a soft delay-filter transfer for the signal.

    ``delay_ratio`` and ``response`` tabulate a high-pass delay filter's
    RMS residual response against ``tau / tau_cut``, as measured on a unit
    sinusoid (e.g. Amiri et al. 2025, Fig. 10, for CHIME's DAYENU filter
    at 200 ns). The response is normalised to ``plateau`` (its high-delay
    value; default the table's maximum) and squared, so that the signal
    *power* surviving at a mode of delay tau is ``T = (R / R_plateau)^2``.
    Below the first tabulated ratio the filter is taken to remove
    everything (``T = 0``); above the last, the last value holds.

    Delay follows from ``k_par`` by the same mapping ``delay_cut_kpar_min``
    inverts: ``tau = k_par c (1+z)^2 / (2 pi nu_21 H(z))``. The returned
    callable takes ``(kpar, z)`` with ``kpar`` an array in Mpc^-1 and is
    suitable for ``expt['kpar_transfer_fn']``.

    Applied to the signal only, this is a conservative bound: the same
    linear filter also attenuates the noise, so per-mode S/N is really
    lost only where the residual is foreground- rather than
    noise-dominated. The hard cut and this soft version bracket that.
    """

    tau_cut_s = _finite_scalar(tau_cut_s, "tau_cut_s")
    if tau_cut_s <= 0.0:
        raise ValueError("tau_cut_s must be positive")
    nu_line_mhz = _finite_scalar(nu_line_mhz, "nu_line_mhz")
    if nu_line_mhz <= 0.0:
        raise ValueError("nu_line_mhz must be positive")
    if not callable(hubble_fn):
        raise TypeError("hubble_fn must be callable")

    ratio = np.asarray(delay_ratio, dtype=float)
    resp = np.asarray(response, dtype=float)
    if ratio.ndim != 1 or ratio.size < 2 or resp.shape != ratio.shape:
        raise ValueError(
            "delay_ratio and response must be matching 1-D tables with at "
            "least two points")
    if not np.all(np.isfinite(ratio)) or not np.all(np.isfinite(resp)):
        raise ValueError("delay_ratio and response must be finite")
    if np.any(np.diff(ratio) <= 0.0) or ratio[0] < 0.0:
        raise ValueError("delay_ratio must be non-negative and increasing")
    if np.any(resp < 0.0):
        raise ValueError("response must be non-negative")
    if plateau is None:
        plateau = float(resp.max())
    plateau = _finite_scalar(plateau, "plateau")
    if plateau <= 0.0:
        raise ValueError("plateau must be positive")
    transfer_table = np.clip((resp / plateau)**2.0, 0.0, 1.0)

    # tau = kpar / delay_scale(z); delay_scale = 2 pi nu_21 H(z) / (c (1+z)^2)
    prefactor = 2.0 * PI * (nu_line_mhz * 1e6) / C

    def kpar_transfer_fn(kpar, z):
        redshift = _finite_scalar_value(z, "z")
        if redshift < 0.0:
            raise ValueError("z must not be negative")
        hubble = _finite_scalar_value(hubble_fn(redshift), "H(z)")
        if hubble <= 0.0:
            raise ValueError("H(z) must be positive")
        delay_scale = prefactor * hubble / (1.0 + redshift)**2.0
        x = np.abs(np.asarray(kpar, dtype=float)) / delay_scale / tau_cut_s
        transfer = np.interp(x, ratio, transfer_table,
                             left=0.0, right=transfer_table[-1])
        return validate_kpar_transfer(transfer, x.shape)

    return kpar_transfer_fn


def validate_experiment_extensions(expt):
    """Fail closed when optional experiment extension values are malformed.

    The function intentionally does not mutate the caller's experiment
    dictionary.  It is cheap enough to call at the public calculation
    boundaries.
    """

    if not isinstance(expt, Mapping):
        raise TypeError("expt must be a mapping")

    validate_experiment_resources(expt)

    mode = expt.get("noise_freq_mode", DEFAULT_NOISE_FREQ_MODE)
    if mode not in NOISE_FREQ_MODES:
        raise ValueError(
            "noise_freq_mode must be one of %s" % sorted(NOISE_FREQ_MODES)
        )

    if "noise_freq_weight" in expt and not callable(expt["noise_freq_weight"]):
        raise TypeError("noise_freq_weight must be callable")

    if "vol_frac" in expt:
        validate_volume_fraction(expt["vol_frac"])

    if "kpar_min_fn" in expt and not callable(expt["kpar_min_fn"]):
        raise TypeError("kpar_min_fn must be callable")

    if "kpar_transfer_fn" in expt and not callable(expt["kpar_transfer_fn"]):
        raise TypeError("kpar_transfer_fn must be callable")


def frequency_noise_penalty(
    weight_fn, frequencies_mhz, mode=DEFAULT_NOISE_FREQ_MODE
):
    """Return the thermal-noise penalty for frequency-dependent flagging.

    ``weight_fn`` may return a scalar (broadcast across the band) or an array
    with exactly the same shape as ``frequencies_mhz``.  Finite weights are
    surviving-time fractions and must be in ``(0, 1]``.  NaNs explicitly mark
    excised slices.  Infinite values are rejected instead of being silently
    interpreted as excision.  If every slice is excised, ``np.inf`` is
    returned.
    """

    if mode not in NOISE_FREQ_MODES:
        raise ValueError(
            "noise_freq_mode must be one of %s" % sorted(NOISE_FREQ_MODES)
        )
    if not callable(weight_fn):
        raise TypeError("noise_freq_weight must be callable")

    frequencies = np.asarray(frequencies_mhz, dtype=float)
    if frequencies.size == 0:
        raise ValueError("frequencies_mhz must not be empty")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies_mhz must contain only finite values")

    weights = np.asarray(weight_fn(frequencies), dtype=float)
    if weights.ndim == 0:
        weights = np.full(frequencies.shape, float(weights), dtype=float)
    elif weights.shape != frequencies.shape:
        raise ValueError(
            "noise_freq_weight returned shape %s; expected %s"
            % (weights.shape, frequencies.shape)
        )

    if np.any(np.isinf(weights)):
        raise ValueError("noise_freq_weight must not return infinite values")

    surviving = ~np.isnan(weights)
    if not np.any(surviving):
        return np.inf

    surviving_weights = weights[surviving]
    if np.any((surviving_weights <= 0.0) | (surviving_weights > 1.0)):
        raise ValueError(
            "finite noise_freq_weight values must be in the interval (0, 1]"
        )

    if mode == "fourier":
        return float(np.mean(1.0 / surviving_weights))
    return float(1.0 / np.mean(surviving_weights))
