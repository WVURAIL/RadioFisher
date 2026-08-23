"""Named, validated models for the neutral-hydrogen signal."""

from collections.abc import Mapping
from types import MappingProxyType


ASTROPHYSICAL_PROFILE_KEY = "astrophysical_model_profile"
ASTROPHYSICAL_MODEL_KEYS = (
    "Tb_model",
    "bias_HI_model",
    "omega_HI_model",
)

ASTROPHYSICAL_MODEL_OPTIONS = MappingProxyType(
    {
        "Tb_model": frozenset({"chang", "hall", "powerlaw", "santos"}),
        "bias_HI_model": frozenset({"castorina", "old", "powerlaw"}),
        "omega_HI_model": frozenset({"crighton", "old", "powerlaw"}),
    }
)

DEFAULT_ASTROPHYSICAL_MODELS = MappingProxyType(
    {
        "Tb_model": "hall",
        "bias_HI_model": "castorina",
        "omega_HI_model": "crighton",
    }
)

ASTROPHYSICAL_MODEL_PROFILES = MappingProxyType(
    {
        "bull2015": MappingProxyType(
            {
                "Tb_model": "powerlaw",
                "bias_HI_model": "powerlaw",
                "omega_HI_model": "powerlaw",
            }
        ),
        "chime_overview_2022": MappingProxyType(
            dict(DEFAULT_ASTROPHYSICAL_MODELS)
        ),
    }
)


def validate_astrophysical_model(key, model):
    """Validate one astrophysical model name and return it unchanged."""

    if key not in ASTROPHYSICAL_MODEL_OPTIONS:
        raise ValueError("unknown astrophysical model key %r" % key)
    if not isinstance(model, str) or model not in ASTROPHYSICAL_MODEL_OPTIONS[key]:
        raise ValueError(
            "%s must be one of %s"
            % (key, sorted(ASTROPHYSICAL_MODEL_OPTIONS[key]))
        )
    return model


def get_astrophysical_profile(name):
    """Return a mutable copy of a named astrophysical model profile."""

    if not isinstance(name, str) or name not in ASTROPHYSICAL_MODEL_PROFILES:
        raise ValueError(
            "astrophysical model profile must be one of %s"
            % sorted(ASTROPHYSICAL_MODEL_PROFILES)
        )
    return dict(ASTROPHYSICAL_MODEL_PROFILES[name])


def resolve_astrophysical_models(cosmo):
    """Resolve and validate the model selection stored in ``cosmo``.

    An optional named profile establishes all three models. Explicit scalar
    model keys then override that profile. If neither is provided, the current
    Hall/Castorina/Crighton behavior is retained.
    """

    if not isinstance(cosmo, Mapping):
        raise TypeError("cosmo must be a mapping")

    selection = dict(DEFAULT_ASTROPHYSICAL_MODELS)
    profile = cosmo.get(ASTROPHYSICAL_PROFILE_KEY)
    if profile is not None:
        selection.update(get_astrophysical_profile(profile))

    for key in ASTROPHYSICAL_MODEL_KEYS:
        if key in cosmo:
            selection[key] = cosmo[key]
        validate_astrophysical_model(key, selection[key])
    return selection


def with_astrophysical_profile(cosmo, profile):
    """Copy a cosmology and attach an explicit named model profile."""

    if not isinstance(cosmo, Mapping):
        raise TypeError("cosmo must be a mapping")
    result = dict(cosmo)
    result[ASTROPHYSICAL_PROFILE_KEY] = profile
    result.update(get_astrophysical_profile(profile))
    return result
