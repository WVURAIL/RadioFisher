# RadioFisher (CHIME forecast path)

RadioFisher computes a Fisher forecast of the baryon acoustic oscillation (BAO)
dilations from a 21 cm intensity-mapping survey with a cylinder interferometer.
This repository is RadioFisher (Bull, Ferreira, Patel & Santos 2015, ApJ 803, 21,
[arXiv:1405.1452](https://arxiv.org/abs/1405.1452)) cut down to the one path used by
the CHIME Overview forecast (CHIME Collaboration 2022, ApJS 261, 29, Appendix A).
One hook is added for a power that is not part of the model.
[RFIsher](https://github.com/WVURAIL/RFIsher) uses this hook to determine how much
residual digital television (DTV) power each 6 MHz channel can tolerate.

![RadioFisher call graph](docs/radiofisher_functions.svg)

## Fisher matrix

For one redshift bin, `fisher()` returns the Fisher matrix of twelve parameters:
the BAO amplitude `A`, the HI bias `b_HI`, the brightness temperature `Tb`, the damping
variance `s_NL = sigma_nl^2` (Mpc squared), `sigma8tot`, `n_s`, the growth rate `f`, the two dilations `aperp` and
`apar`, `fs8`, `bs8`, and the power amplitude `pk`. Each element is

    F_ij = Vfac * Int Int k^2 d_i d_j dk dmu,   d_i = (d ln C_S / d theta_i) * C_S / C_tot

where `C_tot = C_S + C_N + C_FG` is the sum of the HI signal (`Csignal`), the thermal
noise of the cylinder array (`Cnoise`), and the foreground residual (`Cfg`, zero for
CHIME here).

## Residual hook

If `expt['P_res']` is set, `fisher_integrands` adds one more row, `_Pres`, equal to
`dP / C_tot`. The residual `dP` is either a multiple of the noise power or a callable
`dP(k, mu, C_N, C_S)`. It never enters `C_tot`, so the rest of the matrix is the clean
forecast. The `_Pres` column gives the first-order bias. If `B` is the `_Pres` column
and `C` is the inverse of the fitted block, the bias of the fitted parameters is `C B`.
The `_Pres` row and column must be removed before the matrix is inverted.

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

The example prints `(13, 13) _Pres`, for the twelve forecast parameters and the
residual row. The tests compare the matrices with the recorded reference.

RFIsher installs its own pinned revision of RadioFisher, and forecasts are
reproduced with that revision. The environment above is for developing
RadioFisher itself.

The P(k) table is a linear matter power spectrum at z = 0 (columns k in Mpc^-1 and
P in Mpc^3). RFIsher ships the Planck 2018 table and checks it against the cosmology.

`prepare_bin()` returns the set-up that `fisher()` integrates over, so a caller can
evaluate `fisher_integrands()` at its own quadrature nodes. RFIsher does this because
the bias row needs a more accurate quadrature than RadioFisher's Simpson grid.

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
| `src/radiofisher/chime.py` | CHIME and the Planck 2018 cosmology, as in the CHIME Overview |
| `src/radiofisher/castorina.py` | the HI bias fit of Castorina & Villaescusa-Navarro (2017) |
| `src/radiofisher/data/nx_CHIME_800.dat` | CHIME's as-built baseline density n(u) |
| `tools/generate_call_graph.py` | draws the call graph above |

## Scope

This package does not cover the other telescopes and survey modes, galaxy surveys,
modified gravity, neutrino and non-Gaussianity parameters, CAMB execution, the
redshift-binning and plotting helpers, or the CHIME Overview's multi-bin combination.
These are in the full code at commit ae723f7 of this repository. The tests compare the
matrices with the output of that commit (`tests/data/reference_ae723f7.json`) to a
relative tolerance of 1e-8. The reference file names the fourth row `sigma_NL`, which
is the same variance as `s_NL`.

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
