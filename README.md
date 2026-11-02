# RadioFisher (CHIME forecast path)

RadioFisher computes a Fisher forecast of the baryon acoustic oscillation (BAO)
dilations from a 21 cm intensity-mapping survey with a cylinder interferometer.
This repository is RadioFisher (Bull, Ferreira, Patel & Santos 2015, ApJ 803, 21,
[arXiv:1405.1452](https://arxiv.org/abs/1405.1452)) cut down to the one path used by
the CHIME Overview forecast (CHIME Collaboration 2022, ApJS 261, 29, Appendix A).
One hook is added for a power that is not part of the model.
[RFIsher](https://github.com/WVURAIL/RFIsher) uses this hook to determine how much
residual digital television (DTV) power each 6 MHz channel can tolerate.

## Fisher matrix

For one redshift bin, `fisher()` returns the Fisher matrix for six fitted
parameters: BAO amplitude `A`, damping variance `s_NL = sigma_nl^2`,
the dilations `aperp` and `apar`, and `fs8` and `bs8`. Each element is

    F_ij = Vfac * Int Int k^2 d_i d_j dk dmu,   d_i = (d ln C_S / d theta_i) * C_S / C_tot

The clean covariance is `C_tot = C_S + C_N`: HI signal plus thermal noise.
Dilation derivatives shift the BAO feature only. The reference forecast's
foreground mode limit and nonlinear scale are retained. Other dilation
switches and nonzero foreground-residual amplitudes are refused.

## Residual hook

If `expt['P_res']` is set, `fisher_integrands` adds one more row, `_Pres`, equal to
`dP / C_tot`. The residual `dP` is either a multiple of the noise power or a callable
`dP(k, mu, C_N, C_S)`. It never enters `C_tot`, so the rest of the matrix is the clean
forecast. The `_Pres` column gives the first-order bias. If `B` is the `_Pres` column
and `C` is the inverse of the fitted block, the bias of the fitted parameters is `C B`.
The `_Pres` row and column must be removed before the matrix is inverted.
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

## Usage

A standalone checkout is made in a new directory and needs Git and Python 3.10 or
newer. No telescope data or GPU is required.

```bash
git clone https://github.com/WVURAIL/RadioFisher.git
cd RadioFisher
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
```

Run the following from the checkout root. The example reads the small P(k)
fixture in `tests/data/`, which is part of the repository and is not installed
with the library.

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

The example prints `(7, 7) _Pres`: six fitted parameters plus the
residual row. The tests check the matrices against the recorded reference.

RFIsher installs its own pinned revision of RadioFisher, and forecasts are
reproduced with that revision. The environment above is for developing
RadioFisher itself.

The P(k) table is a linear matter power spectrum at z = 0 (columns k in Mpc^-1 and
P in Mpc^3). RFIsher ships the Planck 2018 table and checks it against the cosmology.

`prepare_bin()` returns the set-up that `fisher()` integrates over, so a caller can
evaluate `fisher_integrands()` at its own quadrature nodes. RFIsher does this because
the bias row needs a more accurate quadrature than RadioFisher's Simpson grid.

`prepare_bin()` also supports `bao_derivative="direct"` with an explicit
`bao_step` (default `1e-7` Mpc^-1). The default `"grid"` preserves the original
forecast. `quadrature.py` provides segmented Gauss–Legendre nodes, the BAO and
instrument breakpoints, and integration of the shared covariance integrands.
RFIsher uses this path to integrate the clean forecast and additive residual response.

`quadrature_segments(edges, angular_edges, ...)` generates positive-mu nodes
with weights `2 k^2 dk dmu`. This reflection requires an integrand even in mu.
`integrate_segments(segments, cosmo, expt, volume_factor)` evaluates and sums
those weighted products. A residual that is not even in mu needs appropriate
full-domain nodes and weights; the integrator accepts those directly.

## Units and conventions

The shared constants are in `src/radiofisher/units.py`. `DEG_TO_RAD` uses Python's
`math.pi`, and `HOUR_IN_MHZ_INVERSE = 3600 * 1_000_000` is an exact unit conversion.
`FORECAST_LIGHT_SPEED_KM_S = 300_000` and `NU21_MHZ = 1420.406` are the
conventions of the CHIME Overview forecast. The baseline-density table also uses
the rounded speed of light, so a change to the SI value would require a consistent
new model and regenerated inputs.

`instrument.py` supplies the wavelength, the effective collecting area, and the mode
cuts for both RadioFisher's noise and RFIsher's residual. By default,
`fisher_integrands` evaluates the Cartesian grid of k and mu, and with `paired=True`
it evaluates pointwise batches. The Simpson integrator is vectorized over the mu
axis. To keep memory bounded, the parameter pairs are integrated one at a time.

## Files

| File | Contents |
|---|---|
| `src/radiofisher/baofisher.py` | the forecast (inputs, background, signal and noise, integrands, `fisher()`) |
| `src/radiofisher/units.py`, `instrument.py` | shared numerical conventions and instrument geometry |
| `src/radiofisher/results.py` | named fitted-block solve, clean covariance, bias, and Jacobian projections |
| `src/radiofisher/quadrature.py` | segmented quadrature, model breakpoints, and weighted integration |
| `src/radiofisher/baselines.py` | CHIME all-pair and inter-cylinder baseline construction |
| `src/radiofisher/chime.py` | CHIME and the Planck 2018 cosmology, as in the CHIME Overview |
| `src/radiofisher/castorina.py` | the HI bias fit of Castorina & Villaescusa-Navarro (2017) |
| `src/radiofisher/data/nx_CHIME_800.dat`, `nx_CHIME_inter.dat` | CHIME all-pair and inter-cylinder densities |

## Package boundary

RadioFisher owns the forecasting physics and reusable numerical operations.
It accepts the fitted parameter names, residual function, and projections from
the caller. It does not select DTV channels, assign a residual strength, or
choose a tolerance policy. RFIsher supplies those study choices and records
the tables and figures. CHIME baseline selection is likewise a caller choice;
the original experiment default still uses all baselines.

This package does not cover the other telescopes and survey modes, galaxy surveys,
modified gravity, neutrino and non-Gaussianity parameters, CAMB execution, the
redshift-binning and plotting helpers, or the CHIME Overview's multi-bin combination.
These are in the full code at commit ae723f7 of this repository.

## Reference check

The tests compare the retained six-parameter block with the original output
at commit ae723f7 (`tests/data/reference_ae723f7.json`) to 1e-8 relative.
The reference names the damping variance `sigma_NL`; the current name is
`s_NL`.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Credits

Philip Bull and Pedro G. Ferreira wrote RadioFisher (2013 to 2014). Simon Foreman wrote the
CHIME configuration for the CHIME Overview. If you use this code, cite Bull, Ferreira,
Patel & Santos (2015), [doi:10.1088/0004-637X/803/1/21](https://doi.org/10.1088/0004-637X/803/1/21).
The code is released under the Academic Free License v3.0 (see `LICENSE`).

## Figure colors

Plots and diagrams use the WVU colors of the
[Beamer template](https://github.com/WVURAIL/wvu_beamer_template). A standalone copy
is kept in `tools/wvu_palette.py` and is updated with the template's
`tools/export_plot_palette.py`. Measurements are drawn in blue, and model curves and
small labels in old gold. Further series use the other WVU accents, with distinct
markers or line styles where needed. Grayscale masks keep their binary meaning. A
journal's own figure requirements take precedence. The palette changes only the
figures.
