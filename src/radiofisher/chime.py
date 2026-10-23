"""
CHIME as forecast in the CHIME Overview (CHIME Collaboration 2022, ApJS 261,
29, Appendix A): the Planck 2018 fiducial cosmology, the as-built 4 x 256
cylinder geometry, T_sys = 55 K, 31,000 deg^2, one year, epsilon_fg = 0, the
BAO-shift dilations only, and the as-built baseline density n(u).

From S. Foreman's chime-update branch of RadioFisher (chime2021/
experiments_CHIME.py); keys the forecast does not read are removed.
"""
import os

from .units import DEG_TO_RAD, HOUR_IN_MHZ_INVERSE, NU21_MHZ

# Planck 2018 (CMB only) best fit, CHIME Collaboration 2022 Eq. A8
cosmo = {
    'omega_M_0':        0.3158,
    'omega_lambda_0':   1 - 0.3158,
    'omega_b_0':        0.022383/0.6732**2,
    'N_eff':            3.046,
    'h':                0.6732,
    'ns':               0.96605,
    'sigma_8':          0.8120,
    'gamma':            0.55,       # growth index, f = Omega_m(z)^gamma
    'w0':               -1.,
    'wa':               0.,
    'mnu':              0.06,       # eV
    'k_piv':            0.05,       # n_s pivot [Mpc^-1]
    'aperp':            1.,
    'apar':             1.,
    'A':                1.,         # BAO wiggle amplitude
    'sigma_nl':         7.,         # nonlinear damping scale [Mpc]
}

# Which terms enter the dilation derivatives: the BAO shift only
USE = {
    'f_rsd':            False,      # RSD constraint on f(z)
    'alpha_volume':     False,
    'alpha_rsd_angle':  False,
    'alpha_rsd_shift':  False,
    'alpha_bao_shift':  True,
    'alpha_pk_shift':   False,
}

SURVEY = {
    'ttot':             365 * 24 * HOUR_IN_MHZ_INVERSE,      # Total integration time [MHz^-1]
    'nu_line':          NU21_MHZ,                # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       0,                       # FG subtraction residual amplitude
    'k_nl0':            0.14,                    # Non-linear scale at z=0 (sets kmax)
    'use':              USE,
}

# Foreground model (Santos et al. 2005); scaled by epsilon_fg, so zero here
foregrounds = {
    'A':     [57.0, 0.014, 700., 0.088],        # FG noise amplitude [mK^2]
    'nx':    [1.1, 1.0, 2.4, 3.0],              # Angular scale powerlaw index
    'mx':    [-2.07, -2.10, -2.80, -2.15],      # Frequency powerlaw index
    'l_p':   1000.,                             # Reference angular scale
    'nu_p':  130.                               # Reference frequency [MHz]
}
cosmo['foregrounds'] = foregrounds

CHIME = {
    'mode':             'icyl',             # cylinder interferometer
    'Ndish':            1024,               # feeds (receivers)
    'Nbeam':            1,
    'Ncyl':             4,                  # cylinders
    'Ddish':            20.,                # cylinder width [m]
    'cyl_area':         20 * (256*0.3048),  # illuminated area of one cylinder [m^2]
    'Tsys_tot(z)':      lambda z: 55 * 1e3, # total system temperature [mK]
    'survey_dnutot':    400.,               # bandwidth of the whole survey [MHz]
    'survey_numax':     800.,               # top of the survey band [MHz]
    'dnu':              0.390,              # channel width [MHz]
    'Sarea':            31e3*(DEG_TO_RAD)**2.,   # survey area [rad^2]
    'n(x)': os.path.join(os.path.dirname(__file__), 'data', 'nx_CHIME_800.dat'),
    }
CHIME.update(SURVEY)
