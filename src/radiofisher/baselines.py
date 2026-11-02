"""CHIME baseline density from the array layout.

The density n(x) = n(u) nu[MHz]**2 counts unordered feed pairs, with
x = baseline_length / 300 m under the forecast light-speed convention.
The inter-cylinder and all-baseline selections share one geometry.
"""
from importlib import resources
from pathlib import Path

import numpy as np

from . import chime
from .units import FORECAST_LIGHT_SPEED_KM_S

CYLINDERS, CYLINDER_SPACING_M = 4, 22.0
FEEDS, FEED_SPACING_M = 256, 0.3048
METERS_PER_X = FORECAST_LIGHT_SPEED_KM_S / 1000            # x = d / 300 m, the speed of light RadioFisher's file uses
INTER_FILE = "nx_CHIME_inter.dat"


def radiofisher_file() -> Path:
    """RadioFisher's CHIME density: every pair of feeds."""
    return Path(chime.CHIME["n(x)"])


def inter_file() -> Path:
    """The packaged inter-cylinder density."""
    return Path(str(resources.files("radiofisher").joinpath("data", INTER_FILE)))


def density(x, inter_only: bool = True) -> np.ndarray:
    """Compute n(x) from the array layout at the supplied bin centers x.

    The bins are uniform and start at zero.
    """
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    edges = np.r_[x - dx / 2, x[-1] + dx / 2]
    e = np.arange(-(CYLINDERS - 1), CYLINDERS)
    n = np.arange(-(FEEDS - 1), FEEDS)
    E, N = np.meshgrid(e, n, indexing="ij")
    pairs = np.outer(CYLINDERS - np.abs(e), FEEDS - np.abs(n)).astype(float)   # ordered pairs per offset
    d = np.hypot(E * CYLINDER_SPACING_M, N * FEED_SPACING_M) / METERS_PER_X
    keep = (d > 0) & (E != 0) if inter_only else d > 0
    counts, _ = np.histogram(d[keep], bins=edges, weights=pairs[keep])
    return counts / (2 * np.pi * x * dx) / 2


def build(path=None) -> Path:
    """Write the inter-cylinder density on RadioFisher's grid."""
    x = np.loadtxt(radiofisher_file())[:, 0]
    path = Path(path) if path is not None else Path(__file__).parent / "data" / INTER_FILE
    np.savetxt(path, np.c_[x, density(x, inter_only=True)])
    return path


if __name__ == "__main__":  # pragma: no cover
    print(build())
