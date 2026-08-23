import numpy as np

from radiofisher import galaxy


def test_galaxy_signal_is_data_independent_and_finite():
    cosmology = {
        "aperp": 1.0,
        "apar": 1.0,
        "r": 1500.0,
        "rnu": 2500.0,
        "bgal": 1.5,
        "f": 0.8,
        "sigma_nl": 5.0,
        "D": 0.7,
        "A": 1.0,
        "z": 1.0,
        "fbao": lambda k: np.zeros_like(k),
        "pk_nobao": lambda k: np.ones_like(k),
    }

    signal = galaxy.Csignal_galaxy(
        np.array([100.0, 200.0]),
        np.array([50.0, 75.0]),
        cosmology,
        {},
    )

    assert signal.shape == (2,)
    assert np.all(np.isfinite(signal))
    assert np.all(signal > 0.0)


def test_generic_galaxy_fisher_accepts_caller_supplied_survey(monkeypatch):
    monkeypatch.setattr(galaxy.rf, "NSAMP_K", 4)
    monkeypatch.setattr(galaxy.rf, "NSAMP_U", 3)
    monkeypatch.setattr(
        galaxy.rf, "fbao_derivative", lambda fbao, kgrid: lambda k: 0.0
    )
    monkeypatch.setattr(
        galaxy.rf,
        "fisher_integrands",
        lambda kgrid, ugrid, *args, **kwargs: (
            [np.ones((ugrid.size, kgrid.size))],
            ["A"],
        ),
    )
    monkeypatch.setattr(
        galaxy.rf,
        "integrate_fisher_elements",
        lambda derivs, kgrid, ugrid: np.array([[2.0]]),
    )
    cosmology = {
        "ns": 0.96,
        "fbao": lambda k: np.zeros_like(k),
        "k_in_max": 10.0,
    }
    functions = (
        lambda z: np.ones_like(z, dtype=float) * 100.0,
        lambda z: np.asarray(z, dtype=float) * 1000.0 + 1.0,
        lambda z: np.ones_like(z, dtype=float) * 0.7,
        lambda z: np.ones_like(z, dtype=float) * 0.8,
    )

    fisher, names = galaxy.fisher_galaxy_survey(
        0.8,
        0.9,
        ngal=1e-3,
        bias=1.5,
        cosmo=cosmology,
        expt={"fsky": 0.25, "k_nl0": 0.14},
        cosmo_fns=functions,
    )

    assert fisher.shape == (1, 1)
    assert fisher[0, 0] > 0.0
    assert names == ["A"]
