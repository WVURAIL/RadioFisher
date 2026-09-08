"""Rendered uncertainty geometry must agree with the covariance matrix."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from radiofisher import baofisher as rf


@pytest.fixture(autouse=True)
def no_interactive_windows(monkeypatch):
    monkeypatch.setattr(rf.P, 'show', lambda: None)
    monkeypatch.setattr(matplotlib.figure.Figure, 'show', lambda self: None)


@pytest.mark.parametrize('covariance', [np.diag([4., 1.]), np.diag([1., 9.]), np.array([[4., 1.], [1., 3.]])])
def test_drawn_ellipse_recovers_covariance(covariance, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    rf.plot_ellipse(np.linalg.inv(covariance), 0, 1, [0., 0.], ['x', 'y'])
    patch = plt.gca().patches[0]
    theta = np.deg2rad(patch.angle)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    inferred = R@np.diag([(patch.width/(2*1.52))**2, (patch.height/(2*1.52))**2])@R.T
    np.testing.assert_allclose(inferred, covariance, atol=1e-12)
    plt.close('all')


def test_triangle_skip_retains_marginalized_errors_and_priors(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    rf.triangle_plot([0., 1., 2.], np.diag([1., 4., 9.]), ['x', 'y', 'z'], priors=[.5, np.inf, .1], skip=[1])
    fig = plt.gcf()
    assert len(fig.axes) == 3
    assert [ax.get_title() for ax in fig.axes if ax.get_title()] == ['x', 'z']
    fig.canvas.draw()
    plt.close('all')


def test_correlation_plot_returns_renderable_figure(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    F = np.array([[4., 1.], [1., 9.]])
    fig = rf.plot_corrmat(F, ['x', 'y'])
    np.testing.assert_allclose(fig.axes[0].images[0].get_array(), [[1., 1/6], [1/6, 1.]])
    fig.canvas.draw()
    assert rf.figure_of_merit(0, 1, F, twosigma=True) == pytest.approx(np.sqrt(35)/4)
    up, down = rf.fix_log_plot(np.array([1., 2., 3.]), np.array([2., .1, np.inf]))
    assert np.all(down < [1., 2., 3.]) and np.isfinite(up).all()
    plt.close(fig)
