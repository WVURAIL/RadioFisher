"""Quadrature measures, moving discontinuities, and direct BAO derivatives."""
import numpy as np
import pytest
from radiofisher import baofisher
from radiofisher.quadrature import integrate_segments, quadrature_segments


def test_polynomial_moments_and_positive_mu_symmetry():
    segments = list(quadrature_segments([0., .2, .7, 1.], lambda k: [0., .3, 1.], 4, 3))
    k, mu, w = (np.concatenate([s[i] for s in segments]) for i in range(3))
    assert w.sum() == pytest.approx(2 / 3)
    assert (w * k**2).sum() == pytest.approx(2 / 5)
    assert (w * mu**2).sum() == pytest.approx(2 / 9)


def test_breakpoints_follow_each_radial_node():
    segments = list(quadrature_segments([0., .4, 1.], lambda k: [0., k / 2, 1.], 4, 2))
    k, mu, w = (np.concatenate([s[i] for s in segments]) for i in range(3))
    # Integrates 2*k^2*(3-k) after the angular step is integrated.
    assert (w * np.where(mu < k / 2, 1., 3.)).sum() == pytest.approx(1.5)


def test_partial_positive_mu_interval():
    segments = quadrature_segments([0., 1.], lambda k: [.2, .8], 3, 2)
    assert sum(w.sum() for _, _, w in segments) == pytest.approx(.4)


def test_segment_reduction_matches_independent_outer_products(monkeypatch):
    def integrands(k, mu, cosmo, expt, *, paired):
        assert paired
        return [np.ones_like(k), k, mu], ["constant", "k", "_Pres"]
    monkeypatch.setattr(baofisher, "fisher_integrands", integrands)
    segments = list(quadrature_segments([.1, .3, .8], lambda k: [0., .4, 1.], 3, 3))
    got, names = integrate_segments(iter(segments), {}, {}, 2.5)
    expected = np.zeros((3, 3))
    for k, mu, weight in segments:
        for kval, uval, wval in zip(k, mu, weight):
            row = np.array([1., kval, uval])
            expected += wval * np.outer(row, row)
    np.testing.assert_allclose(got, 2.5 * expected, rtol=1e-14, atol=1e-14)
    assert names == ["constant", "k", "_Pres"]


def test_direct_derivative_cutoff_and_centered_rule():
    def fbao(k):
        return np.asarray(k) ** 3
    derivative = baofisher.direct_fbao_derivative(fbao)
    k = np.array([.01, .03, .031, .1])
    expected = np.where(k <= .03, 0., (fbao(k + 1e-7 / 2) - fbao(k - 1e-7 / 2)) / 1e-7)
    np.testing.assert_array_equal(derivative(k), expected)
    assert derivative.fbao is fbao and derivative.step == 1e-7 and derivative.cutoff == .03


@pytest.mark.parametrize("edges", [[0.], [0., 0., 1.], [1., 0.], [0., np.nan]])
def test_invalid_radial_edges(edges):
    with pytest.raises(ValueError, match="radial_edges"):
        list(quadrature_segments(edges, lambda k: [0., 1.]))


@pytest.mark.parametrize("edges", [[-.1, 1.], [0., 1.1], [.5], [0., 0., 1.]])
def test_invalid_angular_edges(edges):
    with pytest.raises(ValueError, match="angular_edges"):
        list(quadrature_segments([.1, 1.], lambda k: edges))


@pytest.mark.parametrize("order", [0, -1, 2.5, True])
def test_invalid_orders(order):
    with pytest.raises(ValueError):
        list(quadrature_segments([.1, 1.], lambda k: [0., 1.], radial_order=order))


def test_empty_integration():
    with pytest.raises(ValueError, match="at least one"):
        integrate_segments([], {}, {}, 1.)


@pytest.mark.parametrize("step", [0., -1., np.inf, np.nan])
def test_invalid_derivative_steps(step):
    with pytest.raises(ValueError):
        baofisher.direct_fbao_derivative(lambda k: k, step)
