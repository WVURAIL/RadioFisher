"""Draw docs/radiofisher_functions.svg: the forecast at function level, and where a residual enters."""
from pathlib import Path

import wvu_palette as wvu

INK, MUTED, SURF, BG = wvu.INK, wvu.MUTED, wvu.WHITE, wvu.NOT_QUITE_WHITE
MODEL, MODEL_BG = wvu.MODEL, wvu.tint(wvu.MODEL)

MONO = "DejaVu Sans Mono, monospace"


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, t, size=13, col=MUTED, weight="normal", mono=False, anchor="start"):
    fam = f' font-family="{MONO}"' if mono else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{col}" font-weight="{weight}"'
            f' text-anchor="{anchor}"{fam} xml:space="preserve">{esc(t)}</text>')


def box(x, y, w, h, title, lines=(), stroke=INK, sw=1.3, fill=SURF, tcol=INK):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>',
           text(x + 12, y + 23, title, 14, tcol, "600", True)]
    for i, line in enumerate(lines):
        out.append(text(x + 12, y + 44 + 18 * i, line, 12.5))
    return "\n".join(out)


def arrow(x1, y1, x2, y2, col=INK, mk="a", label=None, lx=None, ly=None):
    out = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="1.5" marker-end="url(#{mk})"/>'
    if label:
        out += text(lx if lx is not None else x1 + 8, ly if ly is not None else (y1 + y2) / 2 + 4, label, 12, col)
    return out


W, H = 1180, 1110
s = [(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="DejaVu Sans, sans-serif" role="img" '
     'aria-label="RadioFisher call graph: inputs, fisher(), the integrands and where the residual row is added">'),
     ("<defs>"
     f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
     f'<marker id="b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MODEL}"/></marker>'
     "</defs>"), f'<rect width="{W}" height="{H}" fill="{BG}"/>',
     text(24, 34, "RadioFisher: one redshift bin of a 21 cm BAO forecast", 19, INK, "600"),
     text(24, 56, "Main RadioFisher forecast path and the residual hook used by RFIsher (gold).", 13)]

# inputs
s.append(box(24, 76, 360, 118, "chime.py", ["cosmo: Planck 2018, sigma_nl = 7 Mpc", "CHIME: 4 cylinders x 256 feeds, T_sys = 55 K,",
                                             "31,000 deg^2, 0.39 MHz channels, n(u) file,", "one year, BAO shift only"]))
s.append(box(408, 76, 360, 118, "load_power_spectrum(cosmo, table)", ["linear P(k) table at z = 0", "spline_pk_nobao: split into the smooth",
                                                                    "pk_nobao(k) and the BAO wiggles fbao(k)"]))
s.append(box(792, 76, 364, 118, "background_evolution_splines(cosmo)", ["H(z), r(z), D(z), f(z) as splines", "(Ez, fgrowth)",
                                                                       "passed on as cosmo_fns"]))
for x in (204, 588, 974):
    s.append(arrow(x, 194, x, 236))

# fisher()
s.append(f'<rect x="24" y="238" width="1132" height="720" rx="9" fill="{SURF}" stroke="{INK}" stroke-width="2.2"/>')
s.append(text(42, 266, "fisher(zmin, zmax, cosmo, expt, cosmo_fns)", 16, INK, "600", True))
s.append(text(42, 286, "returns F (one row and column per parameter) and the parameter names", 12.5))

s.append(box(44, 302, 1092, 96, "1  prepare_bin(zmin, zmax, cosmo, expt, cosmo_fns)",
             ["band edges and center z; bandwidth = nu_21 / (1 + zmin) - nu_21 / (1 + zmax)",
              "load_interferom_file: baseline density n(u);  omega_HI(z), bias_HI(z) (castorina.py), Tb(z)",
              "D(z), f(z), r(z), r_nu = c (1+z)^2 / H(z);  Vfac = pi V / (2 pi)^3;  k and mu grids;  fbao_derivative"]))
s.append(arrow(590, 398, 590, 436))

# integrands
s.append(f'<rect x="44" y="438" width="1092" height="400" rx="8" fill="{BG}" stroke="{INK}" stroke-width="1.5"/>')
s.append(text(58, 464, "2  fisher_integrands(k, mu, cosmo, expt)", 14.5, INK, "600", True))
s.append(text(58, 484, "at every Fourier mode (k, mu) of the grid", 12.5))
s.append(box(62, 498, 340, 118, "Csignal (fiducial dilations)", ["C_S = T_b^2 (b + f mu^2)^2", "  x exp(-k^2 mu^2 s_NL)", "  x (1 + A fbao) D^2 pk_nobao / (r^2 r_nu)",
                                            "HI signal; s_NL = sigma_nl^2 (Mpc^2)"]))
s.append(box(420, 498, 340, 118, "Cnoise", ["T_sys^2 S_area / (n_pol t_tot nu_21)",
                                           "x lambda^4 / [N_beam (A_eff theta_b)^2]",
                                           "x interferometer_response(q, y)",
                                           "cuts: low |k_par| and high |k|"]))
s.append(box(778, 498, 340, 118, "Cfg", ["foreground power x epsilon_fg^2", "= 0 for CHIME here"]))
s.append(box(62, 632, 1056, 40, "C_tot = C_S + C_N + C_FG", []))
for x in (232, 590, 948):
    s.append(arrow(x, 616, x, 630))
s.append(box(62, 690, 640, 132, "one row per parameter", ["d_i = (d ln C_S / d theta_i) x C_S / C_tot",
                                                         "A, b_HI, Tb, s_NL, sigma8tot, n_s, f, fs8, bs8, pk",
                                                         "aperp, apar: the BAO shift term, through dfbao_dk",
                                                         "(logpk_derivative feeds the switched-off terms)"]))
s.append(box(722, 690, 396, 132, "expt['P_res']  ->  row '_Pres'", ["d_res = dP(k, mu, C_N, C_S) / C_tot",
                                                                   "Integration gives the bias projection B.",
                                                                   "Parameter bias = inverse(F_fit) B.",
                                                                   "The residual stays outside C_tot."], stroke=MODEL, sw=2.2, fill=MODEL_BG, tcol=MODEL))
s.append(arrow(400, 672, 400, 688))
s.append(arrow(920, 672, 920, 688, MODEL, "b"))
s.append(arrow(590, 838, 590, 860))

s.append(box(44, 862, 1092, 78, "3  integrate_fisher_elements(derivs, k, mu)",
             ["F_ij = Vfac x Int Int k^2 d_i d_j dk dmu   (integrate_grid: Simpson rule over the grid)",
              "RFIsher replaces the BAO derivative and integrates paired=True batches with segmented Gauss-Legendre nodes"]))
s.append(box(24, 972, 1132, 58, "Shared instrument.py helpers (RadioFisher and RFIsher)",
             ["wavelength_m, collecting_area_m2, mode_limits: common wavelength, collecting area and mode cuts"]))
s.append(text(24, 1056, "Output: F with rows A, b_HI, Tb, s_NL, sigma8tot, n_s, f, aperp, apar, fs8, bs8, pk and, with a residual, _Pres.", 12.5, INK))
s.append(text(24, 1080, "Select the fitted parameter block F_fit; remove _Pres before inversion. Its cross-column is B, so bias = inverse(F_fit) B.", 12.5, INK))
s.append("</svg>")
(Path(__file__).resolve().parents[1] / "docs" / "radiofisher_functions.svg").write_text("\n".join(s) + "\n")
