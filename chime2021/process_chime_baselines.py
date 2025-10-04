#!/usr/bin/python
"""
Convert the CHIME baseline distribution into its binned n(u) representation.
"""
import numpy as np
import pylab as P
import scipy.integrate

root = "CHIME"
Ddish = 20.
Ndish = 256 * 4
nu = 800. # MHz
l = 3e8 / (nu * 1e6) # Lambda [m]

####################################

outfile = "array_config/nx_CHIME_%d.dat" % nu
# Optionally, can cut baselines d < d_fov by setting this to True
AVG_SMALL_BASELINES = False
Dcut = 0 #Ddish # Cut baselines below this separation
datfile = "array_config/CHIME_baselines.npy"
####################################

def fov(nu, D):
    """
    Field of view, in rad^2, as a fn. of frequency.
    """
    l = 3e8 / (nu*1e6)
    return 180. * 1.22 * (l/D) * (np.pi/180.)**2.

def ubin_width(nu, D):
    """
    Bin width, corresponding to du at a given frequency (u = d / lambda).
    """
    return (1./30.) / np.sqrt(fov(nu, D)) # 1/30!

dat = np.load(datfile).T

# Remove D < Ddish baselines
if Dcut > 0:
    dat = dat[np.where(dat > Dcut)] # Cut sub-FOV baselines
dat /= l # Rescale into u = d / lambda

# Calculate bin edges
du = ubin_width(nu, Ddish)
imax = int(np.max(dat) / du) + 1
edges = np.linspace(0., imax * du, imax+1)

# Calculate histogram (no. baselines in each ring of width du)
bins, edges = np.histogram(dat, edges)
u = np.array([0.5*(edges[i+1] + edges[i]) for i in range(edges.size-1)]) # Centroids
#idxs = np.where(u < Dmin/l)

# Convert to a density, n(u)
nn = bins / (2. * np.pi * u * du)

# Integrate n(u) to find normalisation (should give unity if no baseline cuts applied)
norm = scipy.integrate.simpson(2.*np.pi*nn*u, x=u)
print("n(u) renorm. factor (should be close to unity):", 0.5 * Ndish * (Ndish - 1) / norm, "(not applied)")
#n *= 0.5 * Ndish * (Ndish - 1) / norm

# Optionally, average over all baselines below the cylinder width
idxs = np.where(u < Ddish/l)
n_mean = scipy.integrate.simpson(2.*np.pi*nn[idxs]*u[idxs], x=u[idxs]) \
          / scipy.integrate.simpson(2.*np.pi*u[idxs], x=u[idxs])
if AVG_SMALL_BASELINES: nn[idxs] = n_mean

# Convert to freq.-independent expression, n(x) = n(u) * nu^2,
# where nu is in MHz.
n_x = nn * nu**2.
x = u / nu
np.savetxt(outfile, np.column_stack((x, n_x)))
print("Saved to %s" % outfile)

#P.plot(u, nn)                   
#P.axvline(Dmin/l, color='r')
#P.axvline(Dcut/l, color='c')
#P.axhline(n_mean, color='g')
#P.show()


#exit()
#fname = "%s_%3.2fe9_dec00_60sec.MS_bin%7.4f_du%7.5f.txt" % (root, nu/1e3, umin, du)
#np.savetxt( fname, np.column_stack((edges[idxs[0][-1]+1:-1], bins[idxs[0][-1]+1:])) )
#print "Saved to: %s" % fname

#exit()
# Plot results
#P.hist(dat, edges, edgecolor='none', log=True, alpha=0.3)
#P.show()
