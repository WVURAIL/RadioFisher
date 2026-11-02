"""Named clean covariance and first-order bias from a Fisher matrix."""
from dataclasses import dataclass

import numpy as np


def _names(values, label):
    if isinstance(values, str):
        raise TypeError(f"{label} must be a sequence of names")
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{label} must contain nonempty string names")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    return values


def _real_array(values, label):
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{label} must be real")
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric and real") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array


@dataclass
class ProjectedResult:
    """Bias and covariance of derived quantities in the supplied order."""
    names: tuple[str, ...]
    covariance: np.ndarray
    bias: np.ndarray | None

    @property
    def sigma(self):
        return np.sqrt(np.diag(self.covariance))


@dataclass
class FisherResult:
    """A fitted block, its clean covariance, and its first-order residual bias.

    bias_projection is B; bias is the solved shift F^-1 B. Both are None
    when no residual was supplied. sigma describes statistical uncertainty,
    not uncertainty on the residual model.
    """
    names: tuple[str, ...]
    fisher: np.ndarray
    covariance: np.ndarray
    bias_projection: np.ndarray | None
    bias: np.ndarray | None
    condition: float

    @property
    def sigma(self):
        return np.sqrt(np.diag(self.covariance))

    def project(self, jacobian, names):
        """Propagate covariance and bias through a caller-supplied Jacobian.

        Columns follow self.names; rows follow the output names. A 1-D
        Jacobian represents one derived quantity, evaluated at the fiducial.
        """
        names = _names(names, "projected names")
        jacobian = _real_array(jacobian, "jacobian")
        if jacobian.ndim == 1:
            jacobian = jacobian[None, :]
        if jacobian.shape != (len(names), len(self.names)):
            raise ValueError("jacobian shape must match output and fitted names")
        return ProjectedResult(names, jacobian @ self.covariance @ jacobian.T,
                               None if self.bias is None else jacobian @ self.bias)


def summarize_fisher(F, names, parameters, *, residual="_Pres", bias_projection=None, max_condition=1e8):
    """Return clean covariance and first-order bias for selected parameters.

    Omitted parameters are held fixed. Included nuisance parameters are
    marginalized when reading the relevant covariance submatrix or projection.
    B is read from the residual column, or supplied as bias_projection in the
    complete input names order. These two forms are mutually exclusive.
    The residual is never fitted. The condition limit applies after diagonal
    normalization, independently of positive changes of parameter units.
    """
    names = _names(names, "matrix names")
    parameters = _names(parameters, "fitted parameters")
    matrix = _real_array(F, "Fisher matrix")
    if matrix.shape != (len(names), len(names)):
        raise ValueError("Fisher matrix shape must match matrix names")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Fisher matrix must be symmetric")
    if residual is not None and (not isinstance(residual, str) or not residual):
        raise ValueError("residual must be a nonempty name or None")
    if residual is not None and residual in parameters:
        raise ValueError("the residual projection is not a fitted parameter")
    missing = set(parameters) - set(names)
    if missing:
        raise ValueError(f"unknown fitted parameters: {sorted(missing)}")
    if not np.isfinite(max_condition) or max_condition < 1:
        raise ValueError("max_condition must be finite and at least one")
    indices = [names.index(name) for name in parameters]
    fitted = matrix[np.ix_(indices, indices)]
    diagonal = np.diag(fitted)
    if np.any(diagonal <= 0):
        raise ValueError("a fitted parameter has no positive information")
    embedded = residual is not None and residual in names
    if embedded and bias_projection is not None:
        raise ValueError("supply either a residual column or bias_projection, not both")
    if embedded:
        projection = matrix[indices, names.index(residual)].copy()
    elif bias_projection is not None:
        vector = _real_array(bias_projection, "bias_projection")
        if vector.shape != (len(names),):
            raise ValueError("bias_projection shape must match matrix names")
        projection = vector[indices].copy()
    else:
        projection = None
    scale = np.sqrt(diagonal)
    scaled = fitted / np.outer(scale, scale)
    eigenvalues = np.linalg.eigvalsh(scaled)
    if eigenvalues[0] <= 0:
        raise ValueError("invalid or ill-conditioned fitted Fisher matrix")
    condition = float(eigenvalues[-1] / eigenvalues[0])
    if not np.isfinite(condition) or condition > max_condition:
        raise ValueError("invalid or ill-conditioned fitted Fisher matrix")
    covariance = np.linalg.solve(scaled, np.eye(len(parameters))) / np.outer(scale, scale)
    bias = None if projection is None else np.linalg.solve(scaled, projection / scale) / scale
    return FisherResult(parameters, fitted, covariance, projection, bias, condition)
