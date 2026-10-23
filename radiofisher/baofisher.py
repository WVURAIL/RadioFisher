"""
Fisher forecast for 21 cm intensity mapping with a cylinder interferometer.

This is the part of RadioFisher (Bull et al. 2015, ApJ 803, 21) that the CHIME
Overview forecast (CHIME Collaboration 2022, ApJS 261, 29, Appendix A) uses,
cut down to one path: one redshift bin, the Kaiser redshift-space signal, the
cylinder-interferometer noise, the BAO-shift dilations, and one hook for a
power that the model does not include (``expt['P_res']``).

    fisher(zmin, zmax, cosmo, expt, cosmo_fns)
      prepare_bin                  bin centre, distances, HI signal amplitude, grids
      fisher_integrands(k, mu)     one row per parameter: d ln C_S / d theta * C_S / C_tot
        Csignal                    HI signal power, C_S
        Cnoise                     thermal noise power, C_N (Tsys_tot, interferometer_response)
        Cfg                        foreground residual power, C_FG
        expt['P_res']              extra row '_Pres' = dP / C_tot (enters only here)
      integrate_fisher_elements    Simpson rule over (k, mu)

C_tot = C_S + C_N + C_FG never contains the residual, so the Fisher matrix and
its uncertainties are those of a clean survey. The '_Pres' row and column hold
the first-order bias the residual causes. Strip them before inverting.

(Phil Bull & Pedro G. Ferreira, 2013--2014; cut to the CHIME path in 2026)
"""
import copy

import numpy as np
import scipy.integrate
import scipy.interpolate

from .castorina import castorinaBias
from .units import C

# Samples in k and mu for the Simpson integration in fisher().
NSAMP_K = 500
NSAMP_U = 1500

INF_NOISE = 1e200        # finite stand-in for infinite noise (masks a mode)
EXP_OVERFLOW_VAL = 250.  # cap on exponents before np.exp() overflows


################################################################################
# Inputs: baseline density and linear power spectrum
################################################################################

def load_interferom_file(fname):
    """
    Load n(u) file for interferometer and return linear interpolation function.
    """
    x, _nx = np.genfromtxt(fname).T
    interp_nx = scipy.interpolate.interp1d( x, _nx, kind='linear',
                                            bounds_error=False, fill_value=1./INF_NOISE )
    return interp_nx

def load_power_spectrum(cosmo, cachefile):
    """
    Attach the smooth power spectrum and the BAO wiggle template to ``cosmo``.

    ``cachefile`` is a linear matter power spectrum table at z = 0: columns
    k [Mpc^-1] and P(k) [Mpc^3], header lines starting with '#'. Its k range
    must extend past the largest k of the forecast.
    """
    k_in, pk_in = np.genfromtxt(cachefile).T
    cosmo['pk_nobao'], cosmo['fbao'] = spline_pk_nobao(k_in, pk_in)
    cosmo['k_in_max'] = np.max(k_in)
    cosmo['k_in_min'] = np.min(k_in)
    return cosmo

def spline_pk_nobao(k_in, pk_in, kref=[2.4e-2, 4.5e-1]):
    """
    Construct a smooth power spectrum with BAOs removed, and a corresponding
    BAO template function, by using a two-stage splining process.
    """
    # Spline all (log-log) points except those in user-defined "wiggle region",
    # and then get derivatives of result
    idxs = np.where(np.logical_or(k_in <= kref[0], k_in >= kref[1]))
    _pk_smooth = scipy.interpolate.UnivariateSpline( np.log(k_in[idxs]),
                                                    np.log(pk_in[idxs]), k=3, s=0 )
    pk_smooth = lambda x: np.exp(_pk_smooth(np.log(x)))

    # Construct "wiggles" function using spline as a reference, then spline it
    # and find its 2nd derivative
    fwiggle = scipy.interpolate.UnivariateSpline(k_in, pk_in / pk_smooth(k_in), k=3, s=0)
    derivs = np.array([fwiggle.derivatives(_k) for _k in k_in]).T
    d2 = scipy.interpolate.UnivariateSpline(k_in, derivs[2], k=3, s=1.0) #s=1.
    # (s=1 to get sensible smoothing)

    # Find maxima and minima of the gradient (zeros of 2nd deriv.), then put a
    # low-order spline through zeros to subtract smooth trend from wiggles fn.
    wzeros = d2.roots()
    wzeros = wzeros[np.where(np.logical_and(wzeros >= kref[0], wzeros <= kref[1]))]
    wzeros = np.concatenate((wzeros, [kref[1],]))
    wtrend = scipy.interpolate.UnivariateSpline(wzeros, fwiggle(wzeros), k=3, s=0)

    # Construct smooth "no-bao" function by summing the original splined function and
    # the wiggles trend function
    idxs = np.where(np.logical_and(k_in > kref[0], k_in < kref[1]))
    pk_nobao = pk_smooth(k_in)
    pk_nobao[idxs] *= wtrend(k_in[idxs])
    fk = (pk_in - pk_nobao)/pk_nobao

    # Construct interpolating functions
    ipk = scipy.interpolate.interp1d( k_in, pk_nobao, kind='linear',
                                      bounds_error=False, fill_value=0. )
    ifk = scipy.interpolate.interp1d( k_in, fk, kind='linear',
                                      bounds_error=False, fill_value=0. )
    return ipk, ifk


################################################################################
# Background cosmology and the HI signal amplitude
################################################################################

def fgrowth(cosmo, z):
    """
    Linear growth rate f(z) = Omega_m(z)^gamma.
    """
    c = cosmo
    Oma = c['omega_M_0'] * (1.+z)**3. / Ez(cosmo, z)**2.
    return Oma**c['gamma']

def Ez(cosmo, z):
    """
    Dimensionless Hubble rate.
    """
    a = 1. / (1. + z)
    w0 = cosmo['w0']; wa = cosmo['wa']
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol
    omegaDE = ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    return np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE )

def background_evolution_splines(cosmo, zmax=10., nsamples=500):
    """
    Get interpolation functions for background functions of redshift:
      * H(z), Hubble rate in km/s/Mpc
      * r(z), comoving distance in Mpc
      * D(z), linear growth factor
      * f(z), linear growth rate
    """
    _z = np.linspace(0., zmax, nsamples)
    a = 1. / (1. + _z)
    H0 = (100.*cosmo['h']); w0 = cosmo['w0']; wa = cosmo['wa']
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol

    # Sample Hubble rate H(z) and comoving dist. r(z) at discrete points
    omegaDE = ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    E = np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE )
    _H = H0 * E

    r_c = np.concatenate( ([0.], scipy.integrate.cumulative_trapezoid(1./E, x=_z)) )
    if ok > 0.:
        _r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
    elif ok < 0.:
        _r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
    else:
        _r = (C/H0) * r_c

    # Integrate linear growth rate to find linear growth factor, D(z)
    # N.B. D(z=0) = 1.
    a = 1. / (1. + _z)
    _f = fgrowth(cosmo, _z)
    _D = np.concatenate( ([0.,], scipy.integrate.cumulative_trapezoid(_f, x=np.log(a))) )
    _D = np.exp(_D)

    # Construct interpolating functions and return
    r = scipy.interpolate.interp1d(_z, _r, kind='linear', bounds_error=False)
    H = scipy.interpolate.interp1d(_z, _H, kind='linear', bounds_error=False)
    D = scipy.interpolate.interp1d(_z, _D, kind='linear', bounds_error=False)
    f = scipy.interpolate.interp1d(_z, _f, kind='linear', bounds_error=False)
    return H, r, D, f


def omega_HI(z):
    """
    Omega_HI(z), the fit of Crighton et al. (2015).
    """
    return 4e-4 * (1 + z)**0.6

def bias_HI(z):
    """
    b_HI(z), the halo-model fit of Castorina & Villaescusa-Navarro (2017),
    tabulated in castorina.py.
    """
    return castorinaBias(z)

def Tb(z, cosmo):
    """
    Mean HI brightness temperature Tb(z) in mK (Hall, Bonvin & Challinor 2013).
    """
    omegaHI = omega_HI(z)
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol
    E = np.sqrt(om*(1.+z)**3. + ok*(1.+z)**2. + ol)
    return 188. * cosmo['h'] * omegaHI * (1.+z)**2. / E


################################################################################
# Numerical helpers
################################################################################

def inverse_interpfn(f):
    """
    Invert an array, replacing all 1/0 with 0. This is for use on arrays that
    have been produced by interpolation, where some of the elements were
    outside the interpolation range.
    """
    values = np.asarray(f, dtype=float)
    return np.divide(1., values, out=np.zeros_like(values), where=values != 0)

def logpk_derivative(pk, kgrid):
    """
    Calculate the first derivative of the (log) power spectrum,
    d log P(k) / d k. Sets the derivative to zero wherever P(k) is not defined.

    Parameters
    ----------

    pk : function
        Callable function (usually an interpolation fn.) for P(k)

    kgrid : array_like
        Array of k values on which the integral will be computed.
    """
    # Calculate dlog(P(k))/dk using central difference technique
    # (Sets lowest-k values to zero since P(k) not defined there)
    # Suppress expected boundary divisions locally without changing NumPy's
    # process-global floating-point error policy.
    dk = 1e-7
    with np.errstate(divide='ignore', invalid='ignore'):
        dP = pk(kgrid + 0.5*dk) / pk(kgrid - 0.5*dk)
    # Set invalid and non-positive ratios to 1 (sets deriv. to zero). These
    # values arise when a sample straddles the edge of the P(k) input range
    # (pk=0 on one side); logging them would otherwise produce infinities that
    # poison every Fisher row that adds an alpha-shift contribution.
    dP[(~np.isfinite(dP)) | (dP <= 0.)] = 1.
    dlogpk_dk = np.log(dP) / dk
    return dlogpk_dk

def fbao_derivative(fbao, kgrid, kref=[3e-2, 4.5e-1]):
    """
    Calculate the first derivative of the fbao(k) function.

    Parameters
    ----------

    fbao : function
        Callable function (usually an interpolation fn.) for fbao(k).

    kgrid : array_like
        Array of k values on which the integral will be computed.

    kref : array_like, size 2
        k range for which f_bao(k) was defined.
    """
    # Calculate dfbao/dk using central difference technique
    # (Sets lowest-k values to zero since P(k) not defined there)
    dk = 1e-7
    dfbao_dk = (fbao(kgrid + 0.5*dk) - fbao(kgrid - 0.5*dk)) / dk

    # Deal with sharp edge effect because of bracket over which fbao(k) was
    # defined (zero derivative near to edge)
    idxs = np.where(kgrid <= kref[0])
    if len(idxs[0]) > 0:
        dfbao_dk[idxs] = 0.
        dfbao_dk[np.max(idxs) + 1] = 0.

    # Interpolate
    idfbao_dk = scipy.interpolate.interp1d( kgrid, dfbao_dk, kind='linear',
                                            bounds_error=False, fill_value=0. )
    return idfbao_dk

def integrate_grid(integrand, kgrid, ugrid):
    """
    Integrate over a 2D grid of sample points using the Simpson rule.
    """
    Ik = [scipy.integrate.simpson(integrand.T[i], x=ugrid) for i in range(kgrid.size)]
    return scipy.integrate.simpson(Ik, x=kgrid)

def integrate_fisher_elements(derivs, kgrid, ugrid):
    """
    Construct a Fisher matrix by performing the integrals for each element and
    populating a matrix with the results.
    """
    Nparams = len(derivs)
    K, U = np.meshgrid(kgrid, ugrid)

    # Loop through matrix elements, performing the appropriate integral for each
    F = np.zeros((Nparams, Nparams))
    for i in range(Nparams):
      for j in range(Nparams):
        if j >= i:
          F[i,j] = integrate_grid(K**2. * derivs[i]*derivs[j], kgrid, ugrid)
          F[j,i] = F[i,j]
    return F


################################################################################
# Covariances: signal, noise, foregrounds
################################################################################

def Tsys_tot(z, expt):
    """
    Total system temperature T_sys(z) in mK, from ``expt['Tsys_tot(z)']``.
    """
    return expt['Tsys_tot(z)'](z)

def interferometer_response(q, y, cosmo, expt):
    """
    Inverse baseline density and the channel response along the line of sight,
    1 / n(u) * exp[(k_par / sigma_kpar)^2], for the noise of a cylinder
    interferometer.
    """
    c = cosmo
    kperp = q / (c['aperp']*c['r'])

    # Baseline density n(u) from n(x) = n(u) nu^2, with x = u / nu [MHz]
    u = kperp * c['r'] / (2. * np.pi) # UV plane: |u| = d / lambda
    nu = expt['nu_line'] / (1. + c['z'])
    x = u / nu
    n_u = expt['n(x)'](x) / nu**2.
    n_u[np.where(n_u == 0.)] = 1. / INF_NOISE

    # Gaussian channel response in the parallel direction
    sigma_kpar = np.sqrt(16.*np.log(2)) * expt['nu_line'] / (expt['dnu'] * c['rnu'])
    B_par = (y/(c['rnu']*sigma_kpar))**2.
    B_par[np.where(B_par > EXP_OVERFLOW_VAL)] = EXP_OVERFLOW_VAL
    invbeam2 = np.exp(B_par)
    return invbeam2 / n_u

def Cnoise(q, y, cosmo, expt):
    """
    Thermal noise power C_N of a cylinder interferometer [mK^2], with modes
    below the foreground cut k_fg and above the nonlinear scale k_NL set to
    infinite noise.
    """
    if expt['mode'] != 'icyl':
        raise ValueError("only the cylinder interferometer ('icyl') is supported")
    c = cosmo
    kperp = q / (c['aperp']*c['r'])
    kpar = y / (c['apar']*c['rnu'])

    nu = expt['nu_line'] / (1. + c['z'])
    l = 3e8 / (nu * 1e6) # Wavelength (m)

    # Number of receiver polarisation channels (default is two) and efficiency
    npol = expt['npol'] if 'npol' in list(expt.keys()) else 2.
    effic = expt['effic'] if 'effic' in list(expt.keys()) else 0.7

    # Radiometer noise per unit survey volume
    Vsurvey = expt['Sarea'] * expt['dnutot'] / expt['nu_line']
    Tsys = Tsys_tot(c['z'], expt)
    noise = Tsys**2. * Vsurvey / (npol * expt['ttot'] * expt['dnutot'])

    # Cylinder: collecting area per receiver and a ~90 deg x l/D field of view
    Aeff = effic * expt['Ncyl'] * expt['cyl_area'] / expt['Ndish']
    theta_b = np.sqrt( 0.5 * np.pi * l / expt['Ddish'] )
    noise *= interferometer_response(q, y, cosmo, expt)
    noise *= l**4. / (expt['Nbeam'] * (Aeff * theta_b)**2.)

    # Cut-off in parallel direction due to (freq.-dep.) foreground subtraction
    kfg = 2.*np.pi * expt['nu_line'] / (expt['survey_dnutot'] * c['rnu'])
    kfg *= expt['kfg_fac'] if 'kfg_fac' in list(expt.keys()) else 1.
    noise[np.where(np.abs(kpar) < kfg)] = INF_NOISE

    # Cut off nonlinear scales (Smith et al. 2003, Eq. 20)
    knl0 = expt['k_nl0']
    kmax = knl0 * (1.+c['z'])**(2./(2. + c['ns']))
    noise[np.where(np.sqrt(kpar**2. + kperp**2.) > kmax)] = INF_NOISE

    return noise

def Csignal(q, y, cosmo, expt):
    """
    HI signal power C_S in redshift space (Kaiser) [mK^2].
    """
    c = cosmo

    # Wavenumber and mu = cos(theta)
    # aperp = D_M(fid)/D_M, apar = H/H(fid), consistent with the
    # volume Jacobian below and with fisher_integrands.
    kperp = c['aperp'] * q / c['r']
    kpar = c['apar'] * y / c['rnu']
    k = np.sqrt(kpar**2. + kperp**2.)
    u2 = (kpar / k)**2.
    f, D = c['f'], c['D']

    # Kaiser redshift-space distortion with Gaussian nonlinear damping
    Frsd = (c['btot'] + f*u2)**2. * np.exp(-u2*(k*c['sigma_nl'])**2.)

    # Construct signal covariance and return
    cs = Frsd * (1. + c['A'] * c['fbao'](k)) * D**2. * c['pk_nobao'](k)
    cs *= c['aperp']**2. * c['apar']
    cs = cs * c['Tb']**2. / (c['r']**2. * c['rnu'])

    return cs

def Cfg(q, y, cosmo, expt):
    """
    Foreground removal noise covariance, from p13 (and Table 1) of Pedro's
    notes. [mK^2]
    """
    fg = cosmo['foregrounds']
    n_fg = len(fg['A']) # No. of foreground components
    nu = expt['nu_line'] / (1. + cosmo['z']) # Centre freq. of redshift bin

    # Replace zeros in q to prevent div/0
    _q = q.copy()
    _q[np.where(q == 0.)] = 1e-100 # Set to a very small (but non-zero) value

    # Sum foreground noise contributions
    Cfg = 0
    for i in range(n_fg):
        Cx = fg['A'][i] * (fg['l_p'] / (2.*np.pi*_q))**fg['nx'][i] \
                        * (nu / fg['nu_p'])**fg['mx'][i]
        Cx[np.where(q == 0.)] = INF_NOISE # Mask values where q is zero
        Cfg += Cx

    # Scale by FG subtraction residual amplitude and return
    return expt['epsilon_fg']**2. * Cfg


################################################################################
# Fisher integrands
################################################################################

def fisher_integrands(kgrid, ugrid, cosmo, expt):
    """
    Integrands over (k, mu) for the Fisher matrix, one per parameter, in the
    order ( A, b_HI, Tb, sigma_NL, sigma8tot, n_s, f, aperp, apar, fs8, bs8,
    pk [, _Pres] ). Each is d ln C_S / d theta * C_S / C_tot. Which dilation
    terms enter aperp and apar is set by ``expt['use']``.
    The historical label sigma_NL denotes damping variance s_NL = sigma_nl**2;
    its log-derivative is -k**2 * mu**2, not the damping-length derivative.
    """
    c = cosmo
    use = expt['use']
    r = c['r']; rnu = c['rnu']
    aperp = c['aperp']; apar = c['apar']

    # Convert (k, u) into (q, y)
    K, U = np.meshgrid(kgrid, ugrid)
    y = rnu * K * U
    q = r * K * np.sqrt(1. - U**2.)

    # Calculate k, mu
    k = np.sqrt( (aperp*q/r)**2. + (apar*y/rnu)**2. )
    u2 = y**2. / ( y**2. + (q * rnu/r * aperp/apar)**2. )

    # Linear growth rate and growth factor
    f, D = c['f'], c['D']
    Dinv = inverse_interpfn(D)

    # Linear HI bias
    b = c['btot'] = c['bHI']

    # Signal, noise and foreground covariances
    cs = Csignal(q, y, cosmo, expt)
    cn = Cnoise(q, y, cosmo, expt)
    cf = Cfg(q, y, cosmo, expt)
    ctot = cs + cn + cf

    # Derivatives of the Kaiser RSD function
    drsd_df = 2. * u2 / (b + f*u2)
    drsd_dfsig8 = 2. * u2 * Dinv / (b*c['sigma_8'] + f*c['sigma_8']*u2)
    drsd_dsig2 = -k**2. * u2
    drsd_du2 = 2.*f / (b + f*u2) - (k * c['sigma_nl'])**2.
    drsd_dk = -2. * k * u2 * c['sigma_nl']**2.

    # Analytic log-derivatives
    deriv_A   = c['fbao'](k) / (1. + c['A']*c['fbao'](k)) * cs / ctot
    deriv_bHI = 2. / (b + f*u2) * cs / ctot
    deriv_f   = use['f_rsd'] * drsd_df * cs / ctot
    deriv_bsig8 = 2.*Dinv/(b*c['sigma_8'] + f*c['sigma_8']*u2) * cs / ctot
    deriv_fsig8 = drsd_dfsig8 * cs / ctot
    deriv_sig2 = drsd_dsig2 * cs / ctot
    deriv_pk = cs / ctot
    deriv_sigma8 = (2. / c['sigma_8']) * cs / ctot
    deriv_ns = np.log(k / c['k_piv']) * cs / ctot
    deriv_Tb = (2. / c['Tb']) * cs / ctot

    # Dilation derivatives
    dlogpk_dk = logpk_derivative(c['pk_nobao'], k) # Numerical deriv.

    # Algebraically equivalent to (q/y * ... * u2)**2, but finite at
    # transverse modes (y=0). Odd-sized mu grids include that endpoint.
    angular_factor = 2. * u2 * (1. - u2)
    daperp_u2 = -angular_factor / aperp
    dapar_u2 = angular_factor / apar
    daperp_k = (aperp*q/r)**2. / (k*aperp)
    dapar_k  = (apar*y/rnu)**2. / (k*apar)

    # Split-out alpha terms so that they can be switched on and off
    dfbao_dk = c['dfbao_dk'](k)
    t_term = [0,] * 5; r_term = [0,] * 5
    use_term = [ use['alpha_volume'], use['alpha_rsd_angle'],
                 use['alpha_rsd_shift'], use['alpha_bao_shift'],
                 use['alpha_pk_shift'] ]

    # (t = transverse)
    t_term[0] = (2./aperp)
    t_term[1] = drsd_du2 * daperp_u2
    t_term[2] = drsd_dk * daperp_k
    t_term[3] = c['A'] * dfbao_dk / (1. + c['A'] * c['fbao'](k)) * daperp_k
    t_term[4] = dlogpk_dk * daperp_k

    # (r = radial)
    r_term[0] = (1./apar)
    r_term[1] = drsd_du2 * dapar_u2
    r_term[2] = drsd_dk * dapar_k
    r_term[3] = c['A'] * dfbao_dk / (1. + c['A'] * c['fbao'](k)) * dapar_k
    r_term[4] = dlogpk_dk * dapar_k

    # Sum-up all terms
    deriv_aperp = 0; deriv_apar = 0
    for i in range(5):
        deriv_aperp += use_term[i] * t_term[i]
        deriv_apar  += use_term[i] * r_term[i]
    deriv_aperp *= cs / ctot
    deriv_apar  *= cs / ctot

    deriv_list = [ deriv_A, deriv_bHI, deriv_Tb, deriv_sig2, deriv_sigma8,
                   deriv_ns, deriv_f, deriv_aperp, deriv_apar,
                   deriv_fsig8, deriv_bsig8, deriv_pk ]
    paramnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8tot', 'n_s', 'f',
                  'aperp', 'apar', 'fs8', 'bs8', 'pk']

    # Unmodelled additive residual power (e.g. RFI surviving a mask).
    #
    # An additive dP that the model does not include biases the recovered
    # parameters at first order by  dtheta_i = sum_j (F^-1)_ij B_j  with
    #   B_j = Vfac * Int K^2 (dlnP/dtheta_j)(cs/ctot) (dP/ctot) dk du.
    # The first two factors are exactly deriv_list[j], so B_j is the integral
    # of deriv_list[j] against (dP/ctot) -- the same integral the Fisher
    # elements already perform. Appending dP/ctot as a pseudo-derivative
    # therefore puts the whole bias vector in the last row/column of F, with
    # no change to the integration machinery.
    #
    # expt['P_res'] is either a scalar multiple of the noise power, or a
    # callable f(k, mu, cn, cs) returning dP on the (k, mu) grid.
    # The caller must strip the '_Pres' row/column before inverting F; the
    # entry is a bias, not a parameter, and treating it as one would
    # marginalise over the very contamination being tested.
    if 'P_res' in expt:
        pres = expt['P_res']
        c_res = pres(k, u2**0.5, cn, cs) if callable(pres) else float(pres) * cn
        deriv_list.append(c_res / ctot)
        paramnames.append('_Pres')

    return deriv_list, paramnames


################################################################################
# Forecast for one redshift bin
################################################################################

def prepare_bin(zmin, zmax, cosmo, expt, cosmo_fns, kmin=1e-7, kmax=1.0):
    """
    Set up one redshift bin: copies of ``cosmo`` and ``expt`` filled in at the
    bin centre, the volume factor that multiplies the integrals, and the
    (k, mu) grids that fisher() integrates over.

    Returns (cosmo, expt, Vfac, kgrid, ugrid).
    """
    # Copy, to make sure we don't modify input expt or cosmo
    cosmo = copy.deepcopy(cosmo)
    expt = copy.deepcopy(expt)
    HH, rr, DD, ff = cosmo_fns

    # Survey redshift bounds, central redshift, and total bandwidth
    numin = expt['nu_line'] / (1. + zmax)
    numax = expt['nu_line'] / (1. + zmin)
    expt['dnutot'] = numax - numin
    z = 0.5 * (zmax + zmin)

    # Baseline density n(u)
    expt['n(x)_file'] = expt['n(x)']
    expt['n(x)'] = load_interferom_file(expt['n(x)'])

    # Bin-centre values
    cosmo['omega_HI'] = omega_HI(z)
    cosmo['bHI'] = bias_HI(z)
    cosmo['Tb'] = Tb(z, cosmo)
    cosmo['z'] = z; cosmo['D'] = DD(z)
    cosmo['f'] = ff(z)
    cosmo['r'] = rr(z); cosmo['rnu'] = C*(1.+z)**2. / HH(z) # Perp/par. dist. scales

    # Physical volume (in rad^2 Mpc^3) (note factor of nu_line in here)
    Vphys = expt['Sarea'] * (expt['dnutot']/expt['nu_line']) \
          * cosmo['r']**2. * cosmo['rnu']
    Vfac = np.pi * Vphys / (2. * np.pi)**3.

    # Integration sample points in (k, u)-space
    ugrid = np.linspace(-1., 1., NSAMP_U) # N.B. Order of integ. limits is correct
    kgrid = np.logspace(np.log10(kmin), np.log10(kmax), NSAMP_K)

    # fbao(k) derivative
    cosmo['dfbao_dk'] = fbao_derivative(cosmo['fbao'], kgrid)
    return cosmo, expt, Vfac, kgrid, ugrid

def fisher(zmin, zmax, cosmo, expt, cosmo_fns, kmin=1e-7, kmax=1.0):
    """
    Fisher matrix for one redshift bin [zmin, zmax].

    ``cosmo`` holds the fiducial parameters with ``pk_nobao`` and ``fbao``
    attached (load_power_spectrum); ``expt`` holds the telescope and survey
    (see chime.py); ``cosmo_fns`` is (H, r, D, f) from
    background_evolution_splines.

    Returns (F, paramnames).
    """
    cosmo, expt, Vfac, kgrid, ugrid = prepare_bin(zmin, zmax, cosmo, expt,
                                                  cosmo_fns, kmin, kmax)
    derivs, paramnames = fisher_integrands(kgrid, ugrid, cosmo, expt)
    F = Vfac * integrate_fisher_elements(derivs, kgrid, ugrid)
    return F, paramnames
