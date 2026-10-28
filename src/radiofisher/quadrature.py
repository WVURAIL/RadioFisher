"""Segmented quadrature for the retained RadioFisher BAO model.

Positive-mu intervals are reflected with a factor of two. This requires
an integrand even in mu; callers supply residual-specific boundaries.
"""
import itertools

import numpy as np
from numpy.polynomial.legendre import leggauss

from . import baofisher
from .instrument import mode_limits


def radial_edges(cosmo, expt, knots, *, kmin=1e-7, kmax=1.0, bao_step=1e-7):
    """BAO, power-spectrum, baseline, and mode-cut breakpoints.

    These edges match the directly evaluated BAO derivative, including
    its shifted knots and 0.03 Mpc^-1 cutoff.
    """
    if not np.isfinite([kmin, kmax, bao_step]).all() or not 0 <= kmin < kmax or bao_step <= 0:
        raise ValueError("invalid quadrature range or BAO derivative step")
    kfg, knl = mode_limits(cosmo, expt)
    nu = expt["nu_line"] / (1 + cosmo["z"])
    template = np.asarray(cosmo["fbao"].x)
    pk = np.asarray(cosmo["pk_nobao"].x)
    h = bao_step / 2
    active = template[(template + h >= .03) & (template - h <= kmax)]
    baseline = 2 * np.pi * nu * np.asarray(knots) / cosmo["r"]
    parts = [np.array([kmin, .03, kmax, knl, kfg]), template, pk, active - h,
             active + h, baseline, np.sqrt(baseline ** 2 + kfg ** 2)]
    if expt["use"]["alpha_pk_shift"]:
        parts.extend([pk - 5e-8, pk + 5e-8])
    edges = np.unique(np.concatenate(parts))
    return edges[(edges >= kmin) & (edges <= kmax)]


def angular_edges(k, cosmo, expt, knots):
    """Positive-mu breakpoints of baseline interpolation and foreground cuts."""
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive")
    kfg, _ = mode_limits(cosmo, expt)
    nu = expt["nu_line"] / (1 + cosmo["z"])
    xratio = 2 * np.pi * nu * np.asarray(knots) / (cosmo["r"] * k)
    xratio = xratio[(xratio > 0) & (xratio < 1)]
    mu = np.sqrt(np.maximum(0, 1 - xratio * xratio))
    edge = kfg / k
    return np.unique(np.r_[0., 1., mu, edge if 0 < edge < 1 else []])


def _edges(values, name, *, angular=False):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or values.size < 2 or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0) or values[0] < 0
            or (angular and values[-1] > 1)):
        raise ValueError(f"{name} must be finite, nonnegative, strictly increasing edges"
                         + (" within [0, 1]" if angular else ""))
    return values


def _order(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def quadrature_segments(radial_edges, angular_edges, radial_order=8, angular_order=16):
    """Yield paired k, mu and weights 2 k² dk dmu, one radial segment at a time.

    angular_edges(k) supplies the positive-mu boundaries at each k. Partial
    intervals within [0, 1] are allowed. The factor two assumes an equal
    contribution from their negative-mu reflections.
    """
    edges = _edges(radial_edges, "radial_edges")
    qk, wk = leggauss(_order(radial_order, "radial_order"))
    qu, wu = leggauss(_order(angular_order, "angular_order"))
    for left, right in itertools.pairwise(edges):
        nodes, angles, weights = [], [], []
        for k, kw in zip((left + right) / 2 + (right - left) * .5 * qk, (right - left) * .5 * wk):
            angular = _edges(angular_edges(k), "angular_edges", angular=True)
            width = np.diff(angular)
            midpoint = (angular[:-1] + angular[1:]) / 2
            mu = (midpoint[:, None] + width[:, None] * .5 * qu).ravel()
            angular_weight = (width[:, None] * .5 * wu).ravel()
            nodes.append(np.full(mu.size, k))
            angles.append(mu)
            weights.append(2 * k * k * kw * angular_weight)
        yield np.concatenate(nodes), np.concatenate(angles), np.concatenate(weights)


def integrate_segments(segments, cosmo, expt, volume_factor):
    """Integrate shared parameter and residual rows over paired node batches.

    Weights contain the k-space measure and any symmetry factor. Apply the
    volume factor once after summing segments in their supplied order.
    """
    if not np.isfinite(volume_factor) or volume_factor <= 0:
        raise ValueError("volume_factor must be finite and positive")
    total, names = None, None
    for k, mu, weight in segments:
        k, mu, weight = (np.asarray(v) for v in (k, mu, weight))
        if (k.ndim != 1 or mu.shape != k.shape or weight.shape != k.shape or k.size == 0
                or not all(np.isfinite(v).all() for v in (k, mu, weight))
                or np.any(k < 0) or np.any(np.abs(mu) > 1) or np.any(weight < 0)):
            raise ValueError("segments require finite paired k, mu and nonnegative weights")
        deriv, current_names = baofisher.fisher_integrands(k, mu, cosmo, expt, paired=True)
        current_names = list(current_names)
        if names is not None and current_names != names:
            raise ValueError("parameter names changed between quadrature segments")
        names = current_names
        derivatives = np.asarray(deriv)
        segment = (derivatives * weight) @ derivatives.T
        total = segment if total is None else total + segment
    if total is None:
        raise ValueError("at least one quadrature segment is required")
    return volume_factor * total, names
