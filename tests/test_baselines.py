"""CHIME geometry reproduces both density tables and receiver-pair counts."""
import numpy as np
import pytest
from radiofisher import baselines


@pytest.mark.parametrize("inter_only", [False, True])
def test_density_matches_table_and_pair_count(inter_only):
    path = baselines.inter_file() if inter_only else baselines.radiofisher_file()
    x, expected = np.loadtxt(path).T
    got = baselines.density(x, inter_only=inter_only)
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-14)
    pairs = np.sum(got * 2 * np.pi * x * (x[1] - x[0]))
    assert pairs == pytest.approx(393216 if inter_only else 523776)


def test_build_to_explicit_path(tmp_path):
    path = baselines.build(tmp_path / "density.dat")
    np.testing.assert_array_equal(np.loadtxt(path), np.loadtxt(baselines.inter_file()))
