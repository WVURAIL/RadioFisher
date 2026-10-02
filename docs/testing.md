# Component and numerical verification

```sh
python -m pip install -e '.[test]'
python -m pytest -q --cov --cov-report=term-missing:skip-covered --cov-report=json --cov-report=html
python scripts/check_test_coverage.py
python -m pip check
python -m pip wheel --no-deps --wheel-dir dist .
```

CI runs the suite on Python 3.10 and 3.13. Coverage must measure branches,
include every Python module in `radiofisher`, exceed 85% overall and reach
80% in each module. New modules cannot silently disappear from the report.
Tests, archived exploratory scripts and generated products are outside this
runtime-module gate. The September 2026 local audit used Python 3.12.

| Component | Verification |
|---|---|
| `backend`, `__init__`, resources, experiment presets and units | API/capability contracts, import isolation, bundled baseline access, installation layout, calibrated constants and preset compatibility |
| `astrophysics`, `castorina` | Named profile copying/validation and published H I model values |
| `read_config` | Real files with comments, types, lists and numerical grids; malformed values |
| Background evolution and `mg_growth` | Independent adaptive quadrature with curvature/CPL dark energy; Einstein–de Sitter analytic solution; finite-difference refinement; normalization and input ownership |
| `camb_wrapper` | Parameter writing, subprocess failure modes, radiation/CPL distances, acoustic-angle round trips without caller mutation |
| Spectrum helpers and caches | Independent unit/normalization algebra, cache hash mismatch/corruption, force/reuse, transfer and neutrino derivatives against analytic spectra, smooth/wiggle reconstruction |
| `euclid` | CAMB-to-forecast Jacobian via finite differences; DETF basis matrices and 36-bin projection; malformed table rejection |
| Fisher construction | Real signal/noise/foreground derivatives and integration; finite difference checks of named growth/bias columns; PSD/symmetry, thermal/CV bounds, volume scaling, odd angular grids, optional neutrino/bias/MG parameters and power bins |
| Fisher transformations | Independent background-derivative oracles for curvature, dark energy, growth and distances; design-matrix quadratic forms for combination, expansion and sampling |
| Survey geometry/noise | Binning coverage and spacing, weighted overlapping experiments, beam/baseline bounds and radiometer time scaling |
| `fisher_galaxy` | Real integration, finite matrices, photometric damping, and binned power-spectrum output |
| Plotting | Drawn ellipse reconstructs the input covariance; marginalized triangle subsets/priors; renderable correlation figure and finite log error bars |
| `extensions` | Frequency weights, surviving volume, shape/type/range rejection, retired filter settings, direct noise response |

The fast numerical benches use deterministic analytic spectra and smaller
integration grids; they are not timing benchmarks. CAMB executable calls use
controlled fixtures to test the I/O boundary. Full-resolution RFIsher bank
builds and its `scripts/verify_bank.py` separately exercise the real backend
with CAMB-generated spectra and off-grid direct calculations. MPI execution
and every historical stand-alone research script are not covered end to end.
Coverage is a regression floor, not a proof of physical completeness.

## Corrections exposed by the benches

- Transverse modes (`mu=0`) no longer generate singular angular derivatives.
- `fs8` and `bs8` now label the correct derivatives; old matrices using those
  columns require rebuilding, even when their BAO-amplitude results agree.
- Dark-energy projections use consistent closure, the correct redshift,
  continuous curvature geometry, and growth-factor derivatives for `f*sigma8`.
- CAMB distance helpers include the requested CPL equation of state and no
  longer mutate a caller's Hubble parameter during acoustic-angle inversion.
- Growth ODE precision supports converged parameter derivatives. The formerly
  empty scale-independent growth-derivative helper now returns its result.
- DETF floating-point table indices are validated and converted to integers.
- Scalar zero interpolation, `cls` cache loading, cache input ownership and
  distance-matrix input ownership now follow their documented contracts.
- Error-ellipse axes, triangle-subset handling, and correlation plotting work
  with current Matplotlib and preserve covariance geometry.
- Signal distance rescaling agrees with its volume Jacobian and derivatives.
  The all-constraints distance projection includes the BAO shift, and the
  separately enabled smooth-spectrum term no longer cancels that shift.
- Alternate RSD growth derivatives are defined and match finite differences.
  Galaxy forecasts use their own RSD model, total bias, and photometric
  damping derivatives rather than inheriting the H I RSD switch.

The distance-derivative corrections change all-constraints forecasts,
including newly computed Bull-profile banks. Historical numerical outputs
must retain their original source identity. The Foreman BAO-shift-only
prescription remains a separate explicit configuration.

## The current cosmology

`get_cosmology("cmbspa2026")` uses the posterior means in the fourth numerical
column of [Omori et al. (2026), Table 8](https://arxiv.org/html/2608.31136v1#S7.T8).
The preset is immutable, returned dictionaries are independent, and physical
densities include neutrinos once. The fiducial includes its source, dataset
combination, date, and statistic. This defines a flat ΛCDM forecast point;
it does not add a new dark-energy likelihood or replace the H I prescription.
The inherited `N_eff=3.046` convention remains explicit. The three primary
uses are Bull/Planck 2013 reproduction, Foreman/CHIME Overview reproduction,
and this current default. RFIsher additionally keeps its P-ACT 2025 comparison.
