"""Named Fisher reduction and propagation against independent known answers."""
import numpy as np
import pytest
from radiofisher import summarize_fisher

NAMES = ("x", "y", "_Pres")
FISHER = np.array([[4., 1.], [1., 3.]])
PROJECTION = np.array([5., -7.])
COVARIANCE = np.array([[3., -1.], [-1., 4.]]) / 11
SHIFT = np.array([2., -3.])


def augmented():
    matrix = np.zeros((3, 3))
    matrix[:2, :2] = FISHER
    matrix[:2, 2] = matrix[2, :2] = PROJECTION
    return matrix


def test_known_nondiagonal_answer_and_requested_order():
    result = summarize_fisher(augmented(), NAMES, ("y", "x"))
    assert result.names == ("y", "x")
    np.testing.assert_allclose(result.fisher, FISHER[::-1, ::-1])
    np.testing.assert_allclose(result.covariance, COVARIANCE[::-1, ::-1])
    np.testing.assert_allclose(result.bias_projection, PROJECTION[::-1])
    np.testing.assert_allclose(result.bias, SHIFT[::-1])
    np.testing.assert_allclose(result.sigma, np.sqrt(np.diag(COVARIANCE))[::-1])


def test_omitted_parameter_is_fixed_not_marginalized():
    fixed = summarize_fisher(augmented(), NAMES, ("x",))
    marginal = summarize_fisher(augmented(), NAMES, ("x", "y"))
    assert fixed.covariance[0, 0] == pytest.approx(1 / 4)
    assert fixed.bias[0] == pytest.approx(5 / 4)
    assert marginal.covariance[0, 0] == pytest.approx(3 / 11)
    assert marginal.bias[0] == pytest.approx(2)


def test_explicit_projection_matches_embedded_column():
    result = summarize_fisher(FISHER, ("x", "y"), ("y", "x"), bias_projection=PROJECTION)
    expected = summarize_fisher(augmented(), NAMES, ("y", "x"))
    np.testing.assert_array_equal(result.covariance, expected.covariance)
    np.testing.assert_array_equal(result.bias, expected.bias)


def test_clean_forecast_has_no_assumed_bias():
    result = summarize_fisher(FISHER, ("x", "y"), ("x", "y"))
    np.testing.assert_allclose(result.covariance, COVARIANCE)
    assert result.bias is None and result.bias_projection is None
    assert result.project([1, 0], ("x",)).bias is None


def test_projection_preserves_correlations_sign_and_cancellation():
    result = summarize_fisher(augmented(), NAMES, ("x", "y"))
    projected = result.project([[3., 2.], [-1., 0.]], ("cancel", "negative_x"))
    np.testing.assert_allclose(projected.bias, [0, -2], atol=1e-14)
    np.testing.assert_allclose(projected.covariance, np.array([[31., -7.], [-7., 3.]]) / 11)
    np.testing.assert_allclose(result.project([3, 2], ("cancel",)).sigma, [np.sqrt(31 / 11)])


def test_parameter_units_preserve_result_and_condition():
    result = summarize_fisher(augmented(), NAMES, ("x", "y"))
    units = np.array([1e-6, 1e6])
    changed = summarize_fisher(FISHER * np.outer(units, units), ("x", "y"), ("x", "y"),
                               bias_projection=PROJECTION * units)
    np.testing.assert_allclose(changed.bias * units, result.bias)
    np.testing.assert_allclose(changed.covariance * np.outer(units, units), result.covariance)
    assert changed.condition == pytest.approx(result.condition)


def test_projection_sources_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either"):
        summarize_fisher(augmented(), NAMES, ("x", "y"), bias_projection=np.zeros(3))


@pytest.mark.parametrize("matrix,names,parameters", [
    (np.eye(2), ("x",), ("x",)),
    (np.eye(2), ("x", "x"), ("x",)),
    (np.eye(2), ("x", "y"), ("x", "x")),
    (np.eye(2), ("x", "y"), ()),
    (np.eye(2), ("x", "y"), ("missing",)),
    (np.eye(2), ("x", "_Pres"), ("x", "_Pres")),
    (np.array([[1, 0], [1, 1]]), ("x", "y"), ("x", "y")),
    (np.array([[1, np.nan], [np.nan, 1]]), ("x", "y"), ("x", "y")),
    (np.eye(2, dtype=complex), ("x", "y"), ("x", "y")),
    (np.diag([1, 0]), ("x", "y"), ("x", "y")),
    (np.array([[1, 2], [2, 1]]), ("x", "y"), ("x", "y")),
    (np.ones((2, 2)), ("x", "y"), ("x", "y")),
    (np.array([[1, 1-1e-10], [1-1e-10, 1]]), ("x", "y"), ("x", "y")),
])
def test_invalid_or_unidentifiable_fit_is_refused(matrix, names, parameters):
    with pytest.raises(ValueError):
        summarize_fisher(matrix, names, parameters)


@pytest.mark.parametrize("vector", [[1], [[1, 2]], [np.inf, 0], [1j, 0]])
def test_invalid_explicit_projection_is_refused(vector):
    with pytest.raises(ValueError):
        summarize_fisher(FISHER, ("x", "y"), ("x", "y"), bias_projection=vector)


@pytest.mark.parametrize("limit", [0, -1, np.inf, np.nan])
def test_invalid_condition_limit_is_refused(limit):
    with pytest.raises(ValueError, match="max_condition"):
        summarize_fisher(FISHER, ("x", "y"), ("x", "y"), max_condition=limit)


@pytest.mark.parametrize("jacobian,names", [
    ([1], ("derived",)), ([1, 2], ("first", "second")),
    ([[1, 2], [3, 4]], ("same", "same")), ([np.nan, 0], ("derived",)), ([1j, 0], ("derived",)),
])
def test_invalid_projection_is_refused(jacobian, names):
    result = summarize_fisher(FISHER, ("x", "y"), ("x", "y"))
    with pytest.raises(ValueError):
        result.project(jacobian, names)
