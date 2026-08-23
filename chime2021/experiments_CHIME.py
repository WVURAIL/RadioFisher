"""
Alternative version of radiofisher/experiments.py, featuring updated CHIME parameters.
"""
import numpy as np

from radiofisher.units import D2RAD, HRS_MHZ

# Define fiducial cosmology and parameters
# Planck-only best-fit parameters, from Table 1 of 1807.06209.
# (This is consistent with the external D_V measurements compiled
# in 2110.03824.)
cosmo = {
    'omega_M_0':        0.3158,
    'omega_lambda_0':   1 - 0.3158,
    'omega_b_0':        0.022383/0.6732**2,
    'omega_HI_0':       -1e20,   # This is not expected to be used, so set to a crazy value
    'N_eff':            3.046,
    'h':                0.6732,
    'ns':               0.96605,
    'sigma_8':          0.8120,
    'gamma':            0.55,
    'w0':               -1.,
    'wa':               0.,
    'fNL':              0.,
    'mnu':              0.06, # eV
    'k_piv':            0.05, # n_s
    'aperp':            1.,
    'apar':             1.,
    'bHI0':             -1e20,   # This is not expected to be used, so set to a crazy value
    'A':                1.,
    'sigma_nl':         7.,
    'b_1':              0.,         # Scale-dependent bias (k^2 term coeff.)
    'k0_bias':          0.1,        # Scale-dependent bias pivot scale [Mpc^-1]
    'gamma0':           0.55,
    'gamma1':           0.,
    'eta0':             0.,
    'eta1':             0.,
    'A_xi':             0.00,         # Modified gravity growth amplitude
    'logkmg':           np.log10(0.05) # New modified gravity growth scale
}

# Define which measurements to include in forecasts.
# For CHIME BAO forecasts, we only want to use BAO shift information,
# to be conservative
USE = {
  'f_rsd':             False,     # RSD constraint on f(z)
  'f_growthfactor':    False,    # D(z) constraint on f(z)
  'alpha_all':         False,     # Use all constraints on alpha_{perp,par}
  'alpha_volume':      False,
  'alpha_rsd_angle':   False,
  'alpha_rsd_shift':   False,
  'alpha_bao_shift':   True,
  'alpha_pk_shift':    False
}

# CHIME survey parameters. We assume 1 year of integration time,
# no foreground residuals, and k_NL = 0.14 Mpc^-1 (corresponding
# to FoG damping scale of 7 Mpc, following Bull et al. 2015).
SURVEY = {
    'ttot':             365 * 24 * HRS_MHZ,      # Total integration time [MHz^-1]
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       0,              # FG subtraction residual amplitude
    'k_nl0':            0.14,              # Non-linear scale at z=0 (sets kmax)
    'use':              USE                # Which constraints to use/ignore
}

# Add foreground components to cosmology dict.
# (Extragal. ptsrc, extragal. free-free, gal. synch., gal. free-free).
# These aren't used in the ideal CHIME forecast.
foregrounds = {
    'A':     [57.0, 0.014, 700., 0.088],        # FG noise amplitude [mK^2]
    'nx':    [1.1, 1.0, 2.4, 3.0],              # Angular scale powerlaw index
    'mx':    [-2.07, -2.10, -2.80, -2.15],      # Frequency powerlaw index
    'l_p':   1000.,                             # Reference angular scale
    'nu_p':  130.                               # Reference frequency [MHz]
}
cosmo['foregrounds'] = foregrounds


################################################################################
# IM experiment configurations
################################################################################

CHIME = {
    'mode':             'icyl',             # Interferometer or single dish
    'Ndish':            1024,               # For an interferometer, this should be
                                            #   the total number of feeds
    'Nbeam':            1,                  # No. of beams (for multi-pixel detectors)
    'Ncyl':             4,                  # No. cylinders
    'Ddish':            20.,                # Cylinder width
    'cyl_area':         20 * (256*0.3048),  # Cylinder area, only counting illuminated
                                            #   part of NS axis [m^2]
    'Tinst':            -1e20 * 50.*(1e3),  # Instrument contribution to system temp [mK].
                                            #   Should not be used, so we set to a crazy value.
    'Tsys_tot(z)':      lambda z: 55 * 1e3, # Total system temp., as function of z [mK]
    'survey_dnutot':    400.,               # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,               # Max. freq. of survey
    'dnu':              0.390,              # Bandwidth of single channel [MHz]
    'Sarea':            31e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             102.,               # Max. interferom. baseline [m]
    'Dmin':             0.305,              # Min. interferom. baseline [m]
    'n(x)': "array_config/nx_CHIME_800.dat" # Interferometer antenna density
    }
CHIME.update(SURVEY)
