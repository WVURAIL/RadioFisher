#!/usr/bin/python
"""
Plot fractional errorbars on D_V(z), along with a compilation of existing
D_V(z) measurements.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.patches import Rectangle
import matplotlib.cm

from rfwrapper import rf
from radiofisher import euclid
from radiofisher.units import *

# Get a colorblind-friendly matplotlib color cycle, so that we can repeat colors easily in plots
plt.style.use('tableau-colorblind10')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Load cosmology
cosmo = rf.experiments.cosmo

# Set experiment names and plotting styles
names_fisher = ['gDESI_combinedmnu006', 'yCHIMEmnu006']
names = ['gDESI_combinedmnu006', 'DESI_Lyalpha', 'yCHIMEmnu006']
plot_colors = [colors[3], colors[0], colors[1]]
plot_labels = [
    'DESI LRG+ELG+QSO', 
    r'DESI Lyman-$\alpha$ Forest', 
    'Ideal CHIME (no foregrounds/systematics)'
]
plot_ls = ['.', '.', '.']
plot_offset = [0, 0.0, 0.03]

# Define plot size
fig = plt.figure(figsize=(4.7, 3.2 * 2))
ax = fig.subplots(2, 1)


#############
# Compute fractional D_V(z) errorbars
#############

# Compute and save fractional D_V(z) errorbars for surveys we have Fisher
# matrices for
zc_survey = {}
err_survey = {}
for name in names_fisher:
    root = "forecast_outputs/" + name

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T

    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]

    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    #pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar',
    #         'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    #pnames += ["pk%d" % i for i in range(kc.size)]
    #zfns = [0,1,6,7,8]
    #excl = [2,4,5,  9,10,11,12,13,14] # Exclude all cosmo params
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")

    # Transform from D_A and H to D_V and F
    F_list_lss = []
    for i in range(Nbins):
        Fnew, pnames_new = rf.transform_to_lss_distances(
                              zc[i], F_list[i], pnames, DA=dAc[i], H=Hc[i],
                              rescale_da=1e3, rescale_h=1e2)
        F_list_lss.append(Fnew)
    pnames = pnames_new
    F_list = F_list_lss

    #zfns = ['A', 'b_HI', 'f', 'DV', 'F']
    zfns = ['A', 'bs8', 'fs8', 'DV', 'F']
    excl = ['Tb', 'sigma8', 'n_s', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI'] #'fs8', 'bs8']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )

    # Compute C = F^-1, and compute marginalized errorbars from sqrt(diag(C))
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))

    # Identify functions of z
    pDV = rf.indices_for_param_names(lbls, 'DV*')
    pFF = rf.indices_for_param_names(lbls, 'F*')

    # Compute fiducial D_V(z) and F(z) from D_A(z) and H(z)
    DV = ((1.+zc)**2. * dAc**2. * C*zc / Hc)**(1./3.)
    Fz = (1.+zc) * dAc * Hc / C
    indexes = [pFF, pDV]
    fn_vals = [Fz, DV]

    # Save fractional D_V(z) errorbars and z values
    err_survey[name] = errs[indexes[1]] / fn_vals[1]
    zc_survey[name] = zc


# Now, load saved fractional D_A and H errorbars for DESI Lyman-alpha
# forecasts. We don't compute these ourselves, but simply take them from
# Table 2.7 of Aghamousa et al. 2016
zc_survey["DESI_Lyalpha"], desi_lyalpha_fracsigmaDA, desi_lyalpha_fracsigmaH = (
    np.loadtxt("forecast_inputs/desi_lyalpha_BAO_errorbars_table2.7.txt").T
)
# Assuming that the D_A and H errorbars are uncorrelated (which is all we can
# do without more information), we translate them to fractional D_V errorbars
# via simple error propagation in the expression for D_V.
err_survey["DESI_Lyalpha"] = (
    (4 / 9) * desi_lyalpha_fracsigmaDA ** 2
    + (1 / 9) * desi_lyalpha_fracsigmaH ** 2
) ** 0.5 / 100


#############
# Plot persistent RFI bands for CHIME
#############

# Ranges of bad frequencies given by their start and end frequencies (in MHz).
# Taken from ch_util.rfi
bad_frequencies = np.array(
    [
        [449.41, 450.98],
        [454.88, 456.05],
        [457.62, 459.18],
        [483.01, 485.35],
        [487.70, 494.34],
        [497.85, 506.05],
        [529.10, 536.52],
        [541.60, 554.49],
        [564.65, 585.35],
        [693.16, 693.55],
        [694.34, 696.68],
        [729.88, 745.12],
        [746.29, 756.45],
    ]
)
bad_frequencies = bad_frequencies[:, ::-1]
bad_redshifts = 1420.4 / bad_frequencies - 1

# Plot each one as a shaded rectangle
for i in range(bad_redshifts.shape[0]):
    ax[0].add_patch(Rectangle((bad_redshifts[i, 0], 0.95), bad_redshifts[i, 1]-bad_redshifts[i, 0], 0.1,
             edgecolor = 'none',
             facecolor = colors[2],
             fill=True,
             lw=0,
             alpha=0.4))

## Also shade z regions that are beyond CHIME band
# ax.add_patch(Rectangle((0.5, 0.96), (1420.4/800 - 1) - 0.5, 0.08,
#          edgecolor = 'none',
#          facecolor = colors[0],
#          fill=True,
#          lw=0,
#          alpha=0.3))
#
# ax.add_patch(Rectangle(((1420.4/400 - 1), 0.96), 2.6 - (1420.4/400 - 1), 0.08,
#          edgecolor = 'none',
#          facecolor = colors[0],
#          fill=True,
#          lw=0,
#          alpha=0.3))

#############
# Set aesthetics for upper panel
#############

ax[0].set_xlim(0.5, 2.55)
ax[0].set_ylim(0.95, 1.05)
# ax[0].set_yticks(np.arange(0.95, 1.051, 0.01))
ax[0].set_ylabel(r"$D_V \,/\, D_V^{\rm fid}$")
ax[0].set_xlabel(r"$z$")
# ax[0].legend(ncol=2, loc='upper center')
ax[0].grid(ls=':')
ax[0].tick_params(axis='x')
ax[0].set_xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
# ax[0].set_xticklabels([r'$%.2f$' % x  for x in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]] )

# Make upper frequency axis
ax2 = ax[0].twiny()
ax2.set_xlim(0.5, 2.55)
# ax2.set_ylim(0.96, 1.04)
ax2.xaxis.tick_top()
ax2.set_xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
freq_labels_numbers = np.round(1420.4 / (1 + np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]))).astype(int)
freq_labels = [r'$' + str(x) + '$' for x in freq_labels_numbers]
ax2.tick_params(axis='x')
ax2.set_xticklabels(freq_labels)
ax2.set_xlabel(r'$\nu\;[{\rm MHz}]$')

#############
# Plot fractional D_V(z) errorbars
#############

# Plot DESI forecasts in normal axis
for k in range(len(names)-1):
    ax[0].errorbar(
        zc_survey[names[k]] + plot_offset[k],
        np.ones_like(err_survey[names[k]]),
        yerr=err_survey[names[k]],
        fmt='none',
        c=plot_colors[k],
        label=plot_labels[k],
        capsize=3
    )

# Plot CHIME forecasts in twinned axis, so we can show a separate legend
k = -1
ax2.errorbar(
    zc_survey[names[k]] + plot_offset[k],
    np.ones_like(err_survey[names[k]]),
    yerr=err_survey[names[k]],
    fmt='none',
    c=plot_colors[k],
    label=plot_labels[k],
    capsize=3
)

# Show two legends: one with DESI, one with CHIME
ax[0].legend()
ax2.legend(loc="lower left")


#############
# Plot external D_V(z) measurements in lower panel
#############

# Load measurements
ext_points = np.loadtxt(
    "forecast_inputs/previous_dv_measurements.txt",
    dtype={
        "names": ("Name", "val_mean", "val_low", "val_high", "z_mean", "z_low", "z_high", "arXiv_num"),
        "formats": ("|S15", float, float, float, float, float, float, float)
    }
)

# Plot measurements, with x errorbar denoting redshift range
for mi, (meas, c) in enumerate(zip(ext_points, [colors[5], colors[3], colors[4], colors[2]])):
    ax[1].errorbar(
        [meas[4]],
        [1+meas[1]],
        xerr=[[meas[4]-meas[5]], [meas[6]-meas[4]]],
        yerr=[[meas[1]-meas[2]], [meas[3]-meas[1]]],
        c=c,
        capsize=3
    )

ax[1].text(1.75, 0.975, r"eBOSS Ly$\alpha$", c=colors[2], fontsize=12.)
ax[1].text(1.55, 1.01, r"eBOSS QSO", c=colors[4], fontsize=12.)
ax[1].text(0.9, 0.971, r"eBOSS ELG", c=colors[3], fontsize=12.)
ax[1].text(0.9, 1.001, r"eBOSS LRG", c=colors[5], fontsize=12.)

#############
# Set aesthetics for lower panel
#############

ax[1].set_xlim(0.5, 2.55)
ax[1].set_ylim(0.95, 1.05)
ax[1].set_xlabel(r"$z$")
ax[1].set_ylabel(r"$D_V \,/\, D_V^{\rm fid}$")
ax[1].grid(ls=':')
ax[1].set_xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])

#############
# Save plot to PDF
#############

plt.tight_layout()
plt.savefig('chime_forecast.pdf')
