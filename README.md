# RadioFisher

RadioFisher forecasts cosmological constraints from neutral-hydrogen 21-cm
intensity-mapping experiments and spectroscopic galaxy surveys. The formalism
is described in Bull, Ferreira, Patel, and Santos (2015).

## Supported surface

Version 1.0 is a Python 3 package rather than a collection of executable
forecast scripts. The supported surface is the importable `radiofisher`
package, its explicit `__all__`, and the tests in `tests/`. Python 3.10 or newer
is required.

Historical Python 2, MPI, plotting, and campaign-specific frontends were
removed from the active tree for 1.0. They depended on missing private inputs,
fixed output paths, or obsolete interfaces. They remain available from Git
history when reproducing an older publication. Port a required workflow to the
current package and validate all external inputs instead of running an old
script unchanged.

Install a checkout and run the supported tests with:

```console
python -m pip install -e '.[test]'
python -m pytest
```

NumPy, SciPy, and matplotlib are installed as core dependencies. An external
CAMB executable is optional: it is needed to generate a new matter-power
spectrum, but not to load a validated precomputed spectrum.

## Backend integration contract

Downstream software should inspect the stable metadata instead of inferring
features from a branch name or arbitrary experiment-dictionary keys:

```python
import radiofisher

assert radiofisher.BACKEND_ID == "radiofisher"
assert radiofisher.BACKEND_VERSION == "1.0.0"
assert radiofisher.BACKEND_API_VERSION == 1
capabilities = radiofisher.get_backend_capabilities()
```

The capability set is immutable. Backend API version 1 supports explicit
physical densities, named astrophysical-model profiles,
frequency-dependent noise weights in `invvar` or `fourier` mode, a surviving
survey-volume fraction, and the `P_res` additive-bias response. Extension
values are validated and malformed inputs fail closed.

The package release is 1.0.0, while the backend API remains version 1 because
these integration semantics did not change during the cleanup.

## Cosmology and signal conventions

`omega_M_0` is interpreted as total matter, including massive neutrinos.
`physical_density_parameters()` therefore subtracts both baryon and neutrino
density when deriving cold dark matter. For an unambiguous conversion, provide
the complete `ombh2`, `omch2`, and `omnuh2` triplet. Set
`omega_M_0_includes_neutrinos=False` only when intentionally reading an older
cosmology in which `omega_M_0` meant baryons plus cold dark matter.

Signal evolution can be pinned to a named profile:

```python
cosmo = radiofisher.with_astrophysical_profile(cosmo, "bull2015")
# Tb_model = bias_HI_model = omega_HI_model = "powerlaw"

cosmo = radiofisher.with_astrophysical_profile(
    cosmo, "chime_overview_2022"
)
# Tb_model / bias_HI_model / omega_HI_model = hall / castorina / crighton
```

`fisher()` also accepts those three model keys directly. If they are absent,
the Hall/Castorina/Crighton defaults are used. Unknown profiles or model names
raise `ValueError`.

## Repository data

`radiofisher/data/array_config/` contains the baseline tables used by runnable
experiment presets and is included in the wheel. Public presets are split into
`experiments.SUPPORTED_EXPERIMENT_PRESETS` and
`experiments.UNSUPPORTED_EXPERIMENT_PRESETS`. The latter retain useful
instrument geometry but carry an explicit unavailable-data marker; forecasting
raises `UnsupportedExperimentDataError` until the caller replaces `n(x)` with
a verified path or callable. The exception is available as
`radiofisher.UnsupportedExperimentDataError`. Stale optional `n(x)` references were removed
from autocorrelation-only dish and hybrid presets.

The old `experiments_galaxy` preset catalog was removed because none of its 36
file-backed number-density inputs were distributed. The generic
`radiofisher.galaxy.fisher_galaxy_survey()` algorithm remains available for
callers that supply number density, bias, cosmology, and survey configuration
directly.

`chime2021/experiments_CHIME.py` and
`chime2021/array_config/nx_CHIME_800.dat` preserve the CHIME Overview
as-built configuration used by the supported BAO integration. They are source
checkout data rather than installed package modules. See
`chime2021/README.md` for the boundary.

Experiment dictionaries are mutable. Copy a preset before applying overrides,
and archive the resolved dictionary, backend metadata, physical-density
convention, external-input hashes, and CAMB settings with every scientific
output.

## 1.0 migration notes

- The ambiguous unqualified `experiments.MID_B1_Octave` name and its legacy
  duplicate were removed. Use `experiments.MID_B1_Octave_Updated`.
- The empty `FisherMatrix` placeholder was removed. Fisher matrices are NumPy
  arrays manipulated by the module-level matrix functions.
- Package-root star imports no longer leak implementation-module names.
- `n_IM()` now requires its redshift range and cosmology functions explicitly.
- Obsolete Planck-prior helpers with hard-coded, unavailable files were
  removed from `euclid`.
- The illustrative `exptL` preset was removed because its combined observing
  mode has never been implemented.
- The unusable file-backed `experiments_galaxy` preset catalog was removed;
  the data-independent galaxy Fisher algorithm remains supported.

## Citation and license

If you use RadioFisher in scientific work, cite Philip Bull, Pedro G.
Ferreira, Prina Patel, and Mario Santos, “Late-time cosmology with 21 cm
intensity mapping experiments,” *The Astrophysical Journal* **803**, 21
(2015), [arXiv:1405.1452](https://arxiv.org/abs/1405.1452),
[doi:10.1088/0004-637X/803/1/21](https://doi.org/10.1088/0004-637X/803/1/21).

RadioFisher is distributed under the Academic Free License 3.0. The original
author is Philip Bull; the current repository is maintained by WVURAIL.
