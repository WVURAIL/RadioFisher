# RadioFisher (CHIME forecast path)

A Fisher forecast of the baryon acoustic oscillation (BAO) dilations from a 21 cm
intensity-mapping survey with a cylinder interferometer. This is
RadioFisher (Bull, Ferreira, Patel & Santos 2015, ApJ 803, 21,
[arXiv:1405.1452](https://arxiv.org/abs/1405.1452)) cut down to the one path the CHIME Overview forecast uses (CHIME Collaboration 2022,
ApJS 261, 29, Appendix A), with one hook for a power the model does not include.
[RFIsher](https://github.com/WVURAIL/RFIsher) uses that hook to find how much residual
DTV power each 6 MHz channel can tolerate.

![RadioFisher call graph](docs/radiofisher_functions.svg)

## What it computes

For one redshift bin, `fisher()` returns the Fisher matrix of twelve parameters:
the BAO amplitude `A`, the HI bias `b_HI`, the brightness temperature `Tb`, the damping
variance `s_NL = sigma_nl^2` (Mpc squared), `sigma8tot`, `n_s`, the growth rate `f`, the two dilations `aperp` and
`apar`, `fs8`, `bs8` and the power amplitude `pk`. Each element is

    F_ij = Vfac * Int Int k^2 d_i d_j dk dmu,   d_i = (d ln C_S / d theta_i) * C_S / C_tot

with `C_tot = C_S + C_N + C_FG`: the HI signal (`Csignal`), the thermal noise of the
cylinder array (`Cnoise`) and the foreground residual (`Cfg`, zero for CHIME here).

## The residual hook

If `expt['P_res']` is set, `fisher_integrands` adds one more row, `_Pres`, equal to
`dP / C_tot`. The residual `dP` is either a multiple of the noise power or a callable
`dP(k, mu, C_N, C_S)`. It never enters `C_tot`, so the rest of the matrix is the clean
forecast. The `_Pres` column gives the first-order bias: with `C` the inverse of the
fitted block and `B` the `_Pres` column, the bias of the fitted parameters is `C B`.
The `summarize_fisher()` result API performs that separation, checks the fitted
block, and returns named covariance, uncertainty, and bias arrays. The original
`fisher()` matrix API remains available.

```python
parameters = ("A", "s_NL", "aperp", "apar", "fs8", "bs8")
result = rf.summarize_fisher(F, names, parameters)
print(result.names, result.sigma, result.bias)
# Physical distance: ln D_V = const - 2/3 ln alpha_perp - 1/3 ln alpha_par.
distance = result.project([0, 0, -2/3, -1/3, 0, 0], ("ln_DV",))
print(distance.sigma, distance.bias)
```

Here `F, names` are the output of `fisher()`, as in the example below. Parameters
omitted from `parameters` are held fixed; included nuisance parameters are
marginalized through the covariance. A clean forecast has `bias=None`. An
explicit `bias_projection` vector can also be supplied instead of an embedded
residual column. `sigma` always describes the clean statistical uncertainty.

## Use

For a standalone checkout, start in a new directory with Git and Python 3.10
or newer. No telescope data or GPU is needed.

```bash
git clone https://github.com/WVURAIL/RadioFisher.git
cd RadioFisher
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
```

Run the following from the checkout root. The small P(k) fixture is included
in this repository under `tests/data/`; it is not installed with the library.

```bash
python - <<'PY'
import copy
import radiofisher as rf
from radiofisher.units import HOUR_IN_MHZ_INVERSE

cosmo = rf.load_power_spectrum(copy.deepcopy(rf.chime.cosmo), "tests/data/pk_planck2018.dat")
fns = rf.background_evolution_splines(cosmo)
expt = dict(rf.chime.CHIME, ttot=8760 * HOUR_IN_MHZ_INVERSE, P_res=0.01)
F, names = rf.fisher(1.20, 1.25, cosmo, expt, fns)
print(F.shape, names[-1])
PY
python -m pytest -q
```

The example prints `(13, 13) _Pres`: twelve forecast parameters plus the
residual row. The tests check the matrices against the recorded reference.

If you are using RFIsher, let its installer select RadioFisher's pinned
revision. Use this separate environment to develop RadioFisher itself instead
of replacing the dependency in a forecast-reproduction environment.

The P(k) table is a linear matter power spectrum at z = 0 (columns k in Mpc^-1 and
P in Mpc^3). RFIsher ships the Planck 2018 table and checks it against the cosmology.

`prepare_bin()` also supports `bao_derivative="direct"` with an explicit
`bao_step` (default `1e-7` Mpc^-1). The default `"grid"` preserves the original
forecast. `quadrature.py` provides segmented Gauss–Legendre nodes, the BAO and
instrument breakpoints, and integration of the shared covariance integrands.
RFIsher selects this path and adds the DTV wedge boundary.

`quadrature_segments(edges, angular_edges, ...)` generates positive-mu nodes
with weights `2 k^2 dk dmu`. This reflection requires an integrand even in mu.
`integrate_segments(segments, cosmo, expt, volume_factor)` evaluates and sums
those weighted products. A residual that is not even in mu needs appropriate
full-domain nodes and weights; the integrator accepts those directly.

## Units and conventions

`src/radiofisher/units.py` owns the shared constants. `DEG_TO_RAD` uses
Python's `math.pi`; `HOUR_IN_MHZ_INVERSE = 3600 * 1_000_000` is an exact unit
conversion.
`FORECAST_LIGHT_SPEED_KM_S = 300_000` and `NU21_MHZ = 1420.406` are the
CHIME Overview forecast conventions. The baseline-density table also uses
the rounded speed of light, so replacing it with the SI value would require
a consistent new model and regenerated inputs.

`instrument.py` supplies wavelength, effective collecting area and mode cuts
for both RadioFisher's noise and RFIsher's residual. `fisher_integrands` accepts
`paired=True` for pointwise batches; its default evaluates a Cartesian grid.
The Simpson integrator evaluates the angular axis in NumPy arrays. Independent
parameter-pair loops keep memory bounded.

## Files

| File | Contents |
|---|---|
| `src/radiofisher/baofisher.py` | the forecast: inputs, background, signal and noise, integrands, `fisher()` |
| `src/radiofisher/units.py`, `instrument.py` | shared numerical conventions and instrument geometry |
| `src/radiofisher/results.py` | named fitted-block solve, clean covariance, bias, and Jacobian projections |
| `src/radiofisher/quadrature.py` | segmented quadrature, model breakpoints, and weighted integration |
| `src/radiofisher/baselines.py` | CHIME all-pair and inter-cylinder baseline construction |
| `src/radiofisher/chime.py` | CHIME and the Planck 2018 cosmology, as in the CHIME Overview |
| `src/radiofisher/castorina.py` | the HI bias fit of Castorina & Villaescusa-Navarro (2017) |
| `src/radiofisher/data/nx_CHIME_800.dat`, `nx_CHIME_inter.dat` | CHIME all-pair and inter-cylinder densities |
| `tools/generate_call_graph.py` | draws the call graph above |

## Package boundary

RadioFisher owns the forecasting physics and reusable numerical operations.
It accepts the fitted parameter names, residual function, and projections from
the caller. It does not select DTV channels, assign a residual strength, or
choose a tolerance policy. RFIsher supplies those study choices and records
the tables and figures. CHIME baseline selection is likewise a caller choice;
the original experiment default still uses all baselines.

## Scope

Everything off this path: the other telescopes and survey modes, galaxy surveys,
modified gravity, neutrino and non-Gaussianity parameters, CAMB execution, the
redshift-binning and plotting helpers, and the CHIME Overview's multi-bin combination.
The full code is at commit ae723f7 of this repository. The tests compare the matrices with
that commit's output (`tests/data/reference_ae723f7.json`) to 1e-8 relative. The reference
file names the fourth row `sigma_NL`; it is the same variance as `s_NL`.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Credits

Philip Bull and Pedro G. Ferreira wrote RadioFisher (2013 to 2014). Simon Foreman wrote the
CHIME configuration for the CHIME Overview. If you use this code, cite Bull, Ferreira,
Patel & Santos (2015), [doi:10.1088/0004-637X/803/1/21](https://doi.org/10.1088/0004-637X/803/1/21).
Academic Free License v3.0; see `LICENSE`.

### Figure colors

Plots and diagrams use the WVU colors from the
[Beamer template](https://github.com/WVURAIL/wvu_beamer_template), with a standalone
copy in `tools/wvu_palette.py`. Update it with the template's
`tools/export_plot_palette.py`. Blue denotes measurements; old gold provides
contrast for model curves and small labels. Additional series use WVU accents
and distinct markers or line styles where needed. Grayscale masks keep their
binary meaning. Journal requirements take precedence. The palette affects
figures only, not the numerical results.
