#!/usr/bin/env python
"""
Calculate Fisher matrix and P(k) constraints for all redshift bins for a given
experiment.
"""
import numpy as np
# import radiofisher as rf
from rfwrapper import rf
import experiments_CHIME as experiments
from radiofisher.units import *
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()

################################################################################
# Set-up experiment parameters
################################################################################

# Load cosmology and experimental settings
cosmo = experiments.cosmo

# Label experiments with different settings
EXPT_LABEL = ""

# Set survey name and load survey and redshift bins
survey_name = 'yCHIME'
expt = experiments.CHIME
mode = 'icyl'
Sarea = None
cv_limited = False

# Set root for output filenames
root = "forecast_outputs/" + survey_name

# Set redshift bins
zs = np.arange(0.8, 2.51, 0.1)
zs = np.array([
    0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.04, 2.20,
    2.355, 2.51
])
# zs[0] = 0.775
zc = 0.5 * (zs[1:] + zs[:-1])

# Define kbins (used for output)
kbins = np.logspace(np.log10(0.001), np.log10(1.), 91)

# Neutrino mass
# cosmo['mnu'] = 0.

# Precompute cosmological functions, P(k), massive neutrinos, and T(k) for f_NL
cosmo_fns = rf.background_evolution_splines(cosmo)
if cosmo['mnu'] != 0.:
    # Massive neutrinos
    mnu_str = "mnu%03d" % (cosmo['mnu']*100.)
    fname_pk = "cache_pk_%s.dat" % mnu_str
    fname_nu = "cache_%s" % mnu_str
    survey_name += mnu_str; root += mnu_str
    cosmo = rf.load_power_spectrum(cosmo, fname_pk, comm=comm)
    mnu_fn = rf.deriv_neutrinos(cosmo, fname_nu, mnu=cosmo['mnu'], comm=comm)
else:
    # Normal operation (no massive neutrinos or non-Gaussianity)
    cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", comm=comm)
    mnu_fn = None

# Non-Gaussianity
#transfer_fn = rf.deriv_transfer(cosmo, "cache_transfer.dat", comm=comm)
transfer_fn = None

# Effective no. neutrinos, N_eff
#Neff_fn =  rf.deriv_neutrinos(cosmo, "cache_Neff", Neff=cosmo['N_eff'], comm=comm)
Neff_fn = None

switches = []

H, r, D, f = cosmo_fns
# H(z), Hubble rate in km/s/Mpc
# r(z), comoving distance in Mpc
# D(z), linear growth factor
# f(z), linear growth rate


################################################################################
# Store cosmological functions
################################################################################

# Store values of cosmological functions
if myid == 0:
    # Calculate cosmo fns. at redshift bin centroids and save
    _H = H(zc)
    _dA = r(zc) / (1. + np.array(zc))
    _D = D(zc)
    _f = f(zc)
    np.savetxt(root+"-cosmofns-zc.dat", np.column_stack((zc, _H, _dA, _D, _f)))

    # Calculate cosmo fns. as smooth fns. of z and save
    zz = np.linspace(0., 1.05*np.max(zc), 1000)
    _H = H(zz)
    _dA = r(zz) / (1. + zz)
    _D = D(zz)
    _f = f(zz)
    np.savetxt(root+"-cosmofns-smooth.dat", np.column_stack((zz, _H, _dA, _D, _f)) )

# Precompute derivs for all processes
eos_derivs = rf.eos_fisher_matrix_derivs(cosmo, cosmo_fns, fsigma8=True)

################################################################################
# Loop through redshift bins, assigning them to each process
################################################################################

for i in range(zs.size-1):
    if i % size != myid:
      continue
    print(">>> Task %2d working on redshift bin %d / %d -- z = %3.3f" \
          % (myid, i, zs.size, zc[i]))

    # Calculate effective experimental params. in the case of overlapping expts.
    Sarea_rad = Sarea*(D2RAD)**2. if Sarea is not None else None
    expt_eff = rf.overlapping_expts(expt, zs[i], zs[i+1], Sarea=Sarea_rad)

    # Calculate basic Fisher matrix
    # (A, bHI, Tb, sigma_NL, sigma8, n_s, f, aperp, apar, [Mnu], [fNL], [pk]*Nkbins)
    F_pk, kc, binning_info, paramnames = rf.fisher(
                                         zs[i], zs[i+1], cosmo, expt_eff,
                                         cosmo_fns=cosmo_fns,
                                         transfer_fn=transfer_fn,
                                         massive_nu_fn=mnu_fn,
                                         Neff_fn=Neff_fn,
                                         return_pk=True,
                                         cv_limited=cv_limited,
                                         switches=switches,
                                         kbins=kbins )

    # Expand Fisher matrix with EOS parameters
    ##F_eos =  rf.fisher_with_excluded_params(F, [10, 11, 12]) # Exclude P(k)
    F_eos, paramnames = rf.expand_fisher_matrix(zc[i], eos_derivs, F_pk,
                                                names=paramnames, exclude=[],
                                                fsigma8=True)

    # Expand Fisher matrix for H(z), dA(z)
    # Replace aperp with dA(zi), using product rule. aperp(z) = dA(fid,z) / dA(z)
    # (And convert dA to Gpc, to help with the numerics)
    paramnames[paramnames.index('aperp')] = 'DA'
    da = r(zc[i]) / (1. + zc[i]) / 1000. # Gpc
    F_eos[7,:] *= -1. / da
    F_eos[:,7] *= -1. / da

    # Replace apar with H(zi)/100, using product rule. apar(z) = H(z) / H(fid,z)
    paramnames[paramnames.index('apar')] = 'H'
    F_eos[8,:] *= 1. / H(zc[i]) * 100.
    F_eos[:,8] *= 1. / H(zc[i]) * 100.

    # Save Fisher matrix and k bins
    np.savetxt(root+"-fisher-full-%d.dat" % i, F_eos, header=" ".join(paramnames))
    if myid == 0: np.savetxt(root+"-fisher-kc.dat", kc)

    # Save P(k) rebinning info
    np.savetxt(root+"-rebin-Fbase-%d.dat" % i, np.array(binning_info['F_base']) )
    np.savetxt(root+"-rebin-cumul-%d.dat" % i, np.array(binning_info['cumul']) )
    np.savetxt(root+"-rebin-kgrid-%d.dat" % i, np.array(binning_info['kgrid']) )
    np.savetxt(root+"-rebin-Vfac-%d.dat" % i, np.array([binning_info['Vfac'],]) )

comm.barrier()
if myid == 0:
    print("Finished.")
