import numpy as np
import pytest

from radiofisher.baofisher import (
    NEUTRINO_MASS_DENSITY_EV,
    convert_to_camb,
    physical_density_parameters,
)


def _cosmology(**updates):
    h = 0.678
    omnuh2 = 0.06 / NEUTRINO_MASS_DENSITY_EV
    values = {
        "h": h,
        "omega_b_0": 0.0226 / h**2,
        "omega_M_0": (0.0226 + 0.118 + omnuh2) / h**2,
        "omega_lambda_0": 1.0 - (0.0226 + 0.118 + omnuh2) / h**2,
        "mnu": 0.06,
        "ns": 0.96,
        "w0": -1.0,
        "wa": 0.0,
    }
    values.update(updates)
    return values


def test_total_matter_does_not_double_count_massive_neutrinos():
    params = convert_to_camb(_cosmology())

    assert params["ombh2"] == pytest.approx(0.0226)
    assert params["omch2"] == pytest.approx(0.118)
    assert params["omnuh2"] == pytest.approx(
        0.06 / NEUTRINO_MASS_DENSITY_EV
    )


def test_massless_legacy_cosmology_is_unchanged():
    cosmo = _cosmology(mnu=0.0)
    cosmo["omega_M_0"] = (0.0226 + 0.118) / cosmo["h"] ** 2

    params = physical_density_parameters(cosmo)

    assert params == pytest.approx(
        {"ombh2": 0.0226, "omch2": 0.118, "omnuh2": 0.0}
    )


def test_explicit_physical_density_triplet_takes_precedence():
    cosmo = _cosmology(ombh2=0.02, omch2=0.11, omnuh2=0.002)

    assert physical_density_parameters(cosmo) == {
        "ombh2": 0.02,
        "omch2": 0.11,
        "omnuh2": 0.002,
    }


def test_partial_explicit_density_triplet_is_rejected():
    with pytest.raises(ValueError, match="supplied together"):
        physical_density_parameters(_cosmology(omch2=0.118))


def test_legacy_cdm_plus_baryon_convention_has_explicit_opt_out():
    cosmo = _cosmology(omega_M_0_includes_neutrinos=False)
    expected = (cosmo["omega_M_0"] - cosmo["omega_b_0"]) * cosmo["h"] ** 2

    assert physical_density_parameters(cosmo)["omch2"] == pytest.approx(expected)


@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf])
def test_invalid_explicit_density_is_rejected(value):
    with pytest.raises(ValueError):
        physical_density_parameters(
            _cosmology(ombh2=0.02, omch2=value, omnuh2=0.001)
        )
