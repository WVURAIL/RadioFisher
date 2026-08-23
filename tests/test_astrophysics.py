import pytest

from radiofisher import astrophysics
from radiofisher import baofisher
from radiofisher import experiments


def test_named_profiles_are_explicit_and_preserve_current_defaults():
    assert astrophysics.get_astrophysical_profile("bull2015") == {
        "Tb_model": "powerlaw",
        "bias_HI_model": "powerlaw",
        "omega_HI_model": "powerlaw",
    }
    assert astrophysics.get_astrophysical_profile("chime_overview_2022") == {
        "Tb_model": "hall",
        "bias_HI_model": "castorina",
        "omega_HI_model": "crighton",
    }
    assert astrophysics.resolve_astrophysical_models({}) == {
        "Tb_model": "hall",
        "bias_HI_model": "castorina",
        "omega_HI_model": "crighton",
    }


def test_bull2015_profile_values_at_z_one():
    cosmo = astrophysics.with_astrophysical_profile(
        experiments.cosmo, "bull2015"
    )

    assert baofisher.Tb(1.0, cosmo, cosmo["Tb_model"]) == pytest.approx(
        0.264203
    )
    assert baofisher.bias_HI(
        1.0, cosmo, cosmo["bias_HI_model"]
    ) == pytest.approx(0.8942843000716286)
    assert baofisher.omega_HI(
        1.0, cosmo, cosmo["omega_HI_model"]
    ) == pytest.approx(0.000806481)


def test_explicit_model_key_can_override_named_profile():
    selection = astrophysics.resolve_astrophysical_models(
        {
            "astrophysical_model_profile": "bull2015",
            "Tb_model": "hall",
        }
    )
    assert selection == {
        "Tb_model": "hall",
        "bias_HI_model": "powerlaw",
        "omega_HI_model": "powerlaw",
    }


@pytest.mark.parametrize(
    ("cosmo", "match"),
    [
        ({"astrophysical_model_profile": "version1"}, "profile"),
        ({"Tb_model": "default"}, "Tb_model"),
        ({"bias_HI_model": "hall"}, "bias_HI_model"),
        ({"omega_HI_model": "castorina"}, "omega_HI_model"),
    ],
)
def test_invalid_profile_or_model_is_rejected(cosmo, match):
    with pytest.raises(ValueError, match=match):
        astrophysics.resolve_astrophysical_models(cosmo)


@pytest.mark.parametrize(
    ("function", "model"),
    [
        (baofisher.Tb, "mystery"),
        (baofisher.bias_HI, "mystery"),
        (baofisher.omega_HI, "mystery"),
    ],
)
def test_signal_functions_reject_unknown_models(function, model):
    with pytest.raises(ValueError):
        function(1.0, experiments.cosmo, formula=model)


def test_fisher_passes_resolved_models_to_signal_functions(monkeypatch):
    captured = {}

    class ModelsCaptured(Exception):
        pass

    def fake_omega(z, cosmo, formula):
        captured["omega_HI_model"] = formula
        return 1.0

    def fake_bias(z, cosmo, formula):
        captured["bias_HI_model"] = formula
        return 1.0

    def fake_tb(z, cosmo, formula):
        captured["Tb_model"] = formula
        raise ModelsCaptured

    monkeypatch.setattr(baofisher, "omega_HI", fake_omega)
    monkeypatch.setattr(baofisher, "bias_HI", fake_bias)
    monkeypatch.setattr(baofisher, "Tb", fake_tb)

    with pytest.raises(ModelsCaptured):
        baofisher.fisher(
            0.8,
            0.9,
            {"astrophysical_model_profile": "bull2015"},
            {"mode": "dish", "nu_line": 1420.406},
            (lambda z: 1.0, lambda z: 1.0, lambda z: 1.0, lambda z: 1.0),
        )

    assert captured == {
        "Tb_model": "powerlaw",
        "bias_HI_model": "powerlaw",
        "omega_HI_model": "powerlaw",
    }
