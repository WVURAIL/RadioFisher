"""Nominal inputs for CHORD, HIRAX and the CHIME/FRB outriggers.

These experiments show that the forecast runs for other arrays. Their values
are nominal and are not the design of record of any collaboration.

    CHORD   22 x 24 grid of 6 m dishes at 6.3 m east-west and 8.5 m
            north-south (CHORD HI-galaxy forecasts, arXiv:2607.24903), with
            every grid point filled, 300 to 1500 MHz in 6144 channels
            (arXiv:2603.03520) and an assumed T_sys of 30 K
    HIRAX   32 x 32 grid of 6 m dishes at 7 m, 400 to 800 MHz and
            T_inst = 50 K, as in the RadioFisher HIRAX experiment
            (S. Foreman's fork, radiofisher/experiments.py), with its 2 pi sr
    KKO     one 20 m cylinder with 64 feeds at 305 mm, 400 to 800 MHz
    GBO     one 20 m cylinder with 128 feeds at 305 mm
    HCRO    one 20 m cylinder with 128 feeds at 305 mm
            (CHIME/FRB Outriggers design overview, arXiv:2504.05192,
            Section III.2.3), each with the CHIME T_sys and sky area

A single cylinder has no inter-cylinder pairs, so the outriggers use every
baseline. The CHIME survey settings (one year, k_NL, no foreground residual)
apply to all five.
"""
import copy
from importlib import resources
from pathlib import Path

import numpy as np

from . import baselines, chime

X_STEP = 1 / 850     # n(x) bin width, about that of the CHIME tables

LAYOUTS = {
    "CHORD": dict(rows=22, columns=24, row_spacing_m=6.3, column_spacing_m=8.5, min_baseline_m=6.0),
    "HIRAX": dict(rows=32, columns=32, row_spacing_m=7.0, column_spacing_m=7.0, min_baseline_m=6.0),
    "KKO": dict(rows=1, columns=64, row_spacing_m=0.0, column_spacing_m=0.305),
    "GBO": dict(rows=1, columns=128, row_spacing_m=0.0, column_spacing_m=0.305),
    "HCRO": dict(rows=1, columns=128, row_spacing_m=0.0, column_spacing_m=0.305),
}


def density_file(name: str) -> Path:
    """The packaged n(x) table of an array."""
    return Path(str(resources.files("radiofisher").joinpath("data", f"nx_{name}.dat")))


def _dish(name, Tsys_K, numax, dnutot, dnu, Sarea):
    layout = LAYOUTS[name]
    expt = {"mode": "idish", "Ndish": layout["rows"] * layout["columns"], "Nbeam": 1, "Ddish": 6.0,
            "Tsys_tot(z)": lambda z: Tsys_K * 1e3, "survey_numax": numax, "survey_dnutot": dnutot,
            "dnu": dnu, "Sarea": Sarea, "n(x)": str(density_file(name))}
    expt.update(copy.deepcopy(chime.SURVEY))
    return expt


def _outrigger(name):
    feeds = LAYOUTS[name]["columns"]
    expt = copy.deepcopy(chime.CHIME)
    expt.update({"Ndish": feeds, "Ncyl": 1, "cyl_area": 20 * feeds * 0.305, "n(x)": str(density_file(name))})
    return expt


CHORD = _dish("CHORD", 30.0, 1500.0, 1200.0, 1200.0 / 6144, chime.CHIME["Sarea"])
HIRAX = _dish("HIRAX", 50.0, 800.0, 400.0, 0.4, 2 * np.pi)
KKO = _outrigger("KKO")
GBO = _outrigger("GBO")
HCRO = _outrigger("HCRO")
EXPERIMENTS = {"CHORD": CHORD, "HIRAX": HIRAX, "KKO": KKO, "GBO": GBO, "HCRO": HCRO}


def density(name: str) -> np.ndarray:
    """The n(x) table of an array from its layout, as columns x and n(x)."""
    layout = LAYOUTS[name]
    span = np.hypot((layout["rows"] - 1) * layout["row_spacing_m"],
                    (layout["columns"] - 1) * layout["column_spacing_m"]) / baselines.METERS_PER_X
    x = (np.arange(int(np.ceil(span / X_STEP)) + 2) + 0.5) * X_STEP
    return np.c_[x, baselines.grid_density(x, **layout)]


def build(directory=None) -> list[Path]:
    """Write the n(x) table of every array."""
    directory = Path(directory) if directory is not None else Path(__file__).parent / "data"
    paths = []
    for name in LAYOUTS:
        path = directory / f"nx_{name}.dat"
        np.savetxt(path, density(name))
        paths.append(path)
    return paths


if __name__ == "__main__":  # pragma: no cover
    print(*build(), sep="\n")
