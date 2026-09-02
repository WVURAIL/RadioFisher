"""Stable integration metadata for downstream RadioFisher clients.

The scientific code predates package-level API versioning.  Downstream tools
must not infer support for experiment extensions from a branch name or from
the fact that arbitrary dictionary keys happen to be accepted.  This module
provides a deliberately small, immutable capability contract.
"""

BACKEND_ID = "radiofisher"
BACKEND_VERSION = "1.0.0"
BACKEND_API_VERSION = 1

BACKEND_CAPABILITIES = frozenset(
    {
        "P_res",
        "astrophysical_model_profiles",
        "explicit_physical_densities",
        "kpar_min_fn",
        "noise_freq_mode",
        "noise_freq_weight",
        "vol_frac",
    }
)


def get_backend_capabilities():
    """Return the immutable set of supported experiment extensions."""

    return BACKEND_CAPABILITIES
