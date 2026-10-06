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

For the component and numerical verification suite, coverage gates, and the
scope of the independent physics oracles, see [testing](docs/testing.md).
New forecasts can use `radiofisher.get_cosmology()`, whose default is the
August 2026 `cmbspa2026` combination. `experiments.cosmo` retains Bull's
historical Planck 2013 fiducial, and `chime2021/experiments_CHIME.py` retains
Foreman's CHIME Overview fiducial. These historical inputs remain named
references for reproduction; they do not select the new forecast default.

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
values are validated and malformed inputs fail closed. Delay-filter settings
(`kpar_min_fn`, `kpar_transfer_fn`, and `wedge`) are no longer supported;
passing any of them raises an error.

The package release is 1.0.0 and the backend API number stayed at 1, but the
cleanup did change what a forecast integrates. The removed hooks deleted
line-of-sight modes below a delay-filter cut (`kpar_min_fn`), scaled the
signal by a filter transfer (`kpar_transfer_fn`), and cut the foreground wedge
(`wedge`); none of that is applied now, and old settings raise instead of
running unfiltered. Two hard cuts remain in `Cnoise()`, applied as infinite
noise: modes with `|k_par|` below the survey's fundamental line-of-sight mode,
`kfg_fac * 2 pi nu_line / (survey_dnutot * r_nu)` (400 MHz of bandwidth for the
CHIME presets, `kfg_fac` = 1 by default), and modes with `k` above the
nonlinear scale `k_nl0 * (1 + z)**(2 / (2 + n_s))` (`k_nl0` = 0.14 Mpc^-1
unless set).

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

The Fisher parameters `aperp` and `apar` are the dilations applied in
`Csignal()`: `aperp = D_M(fid) / D_M` and `apar = H / H(fid)`, with the sound
horizon held at its fiducial value. The BAO dilations used downstream (RFIsher
and the CHIME DTV dissertation), `alpha_perp = (D_M / r_d) / (D_M / r_d)_fid`
and `alpha_par = (H r_d)_fid / (H r_d)`, are their inverses:
`alpha_perp = 1 / aperp` and `alpha_par = 1 / apar`. At the fiducial point both
equal 1 and the Jacobian is minus the identity, so marginal errors and the
`alpha_perp`-`alpha_par` correlation carry over unchanged; only cross terms with
other parameters change sign. The Fisher row named `sigma_NL` is the
derivative with respect to `sigma_nl**2` (Mpc^2), not `sigma_nl`. Banks that
keep the label hold the same quantity; convert with
`sigma(sigma_nl) = sigma(sigma_nl**2) / (2 * sigma_nl)`.

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
