#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from scipy.stats import linregress
from crpropa import *


SEED = 100
NSIM = 10000
# NSIM = 1

OUTFILENAME = 'output/deflections_brms-%.0e_r-%.0e.txt'
SAVEFILENAME = 'plotdata/deflections_%s_brms-%.0e_r-%.0e.csv'
# OUTFILENAME = 'output/deflections_brms-%.0e_r-%.0e_one-particle_full-traj_ObserverSurface_%s.txt'
LABEL = r'$B_{rms}=%.0e$, $r=%.0e$'


A, Z = 1, 1
EMIN, EMAX = 1e17*eV, 1e20*eV
LMIN, LMAX = 150*kpc, 2000*kpc
SPECTRAL_INDEX_BFIELD = -11/3
SPECTRAL_INDEX_FERMI = -2
SPECTRAL_INDEX_FLAT = -1
CENTER = np.array([128, 128, 128]) * 75*kpc
B_RMS = np.array([50, 20, 10, 1]) * nG
# B_RMS = np.array([10]) * nG


L_c = turbulentCorrelationLength(LMIN, LMAX, SPECTRAL_INDEX_BFIELD)
RADII = L_c * np.array([.05, .5, 1., 5., 50., 100.])
# RADII = L_c * np.array([5., 50.])
DMAX = RADII[-1] * 1e2



def run(B_rms):
    vgrid = VectorGrid(Vector3d(0), 256, 75*kpc)
    initTurbulence(vgrid, B_rms, LMIN, LMAX, SPECTRAL_INDEX_BFIELD, SEED)
    Bfield = MagneticFieldGrid(vgrid)

    m = ModuleList()
    m.add(PropagationCK(Bfield))

    for radius in RADII:
        obs = Observer()
        obs.add(ObserverSurface(Sphere(Vector3d(*CENTER), radius)))
        obs.setDeactivateOnDetection(False)
        output = TextOutput(OUTFILENAME % (B_rms, radius), Output.Event3D)
        # output = TextOutput(OUTFILENAME % (B_rms, radius, 'obs'), Output.Event3D)
        output.enable(Output.SourceDirectionColumn)
        obs.onDetection(output)
        m.add(obs)

    # output = TextOutput(OUTFILENAME % (B_rms, radius, 'full'), Output.Event3D)
    # m.add(output)

    m.add(MaximumTrajectoryLength(DMAX))

    source = Source()
    source.add(SourceParticleType(nucleusId(A, Z)))
    source.add(SourcePowerLawSpectrum(EMIN, EMAX, SPECTRAL_INDEX_FLAT))
    source.add(SourcePosition(Vector3d(*CENTER)))
    source.add(SourceIsotropicEmission())

    m.setShowProgress(True)
    m.run(source, NSIM, True)



def plot_spectrum(B_rms, radius, domain_borders, save=False):
    data = np.genfromtxt(OUTFILENAME % (B_rms, radius), names=True)
    E, E0 = data['E'], data['E0']
    weights = E0**(-SPECTRAL_INDEX_FLAT + SPECTRAL_INDEX_FERMI)
    N_noweights, _ = np.histogram(np.log10(E), bins=20)
    N, bins = np.histogram(np.log10(E), weights=weights, bins=20, density=True)
    dE = (10**bins[1:] + 10**bins[:-1]) / 2.
    dR_L = dE / (sc.c * sc.e * B_rms)
    drho = dR_L / L_c
    yerr = dE**2 * N / dE / np.sqrt(N_noweights)
    plt.errorbar(dE / eV, dE**2 * N / dE, yerr, label=LABEL % (B_rms, radius))
    # yerr = drho**2 * N / drho / np.sqrt(N_noweights)
    # plt.errorbar(drho, drho**2 * N / drho, yerr, label=LABEL % (B_rms, radius))
    # yerr = N / drho / np.sqrt(N_noweights)
    # plt.errorbar(drho, N / drho, yerr, label=LABEL % (B_rms, radius))

    if save:
        modificator = .1 if B_rms < B_RMS[-2] else 1
        np.savetxt(SAVEFILENAME % ('error', B_rms, radius),
                   np.vstack((drho, drho**2*N/drho * modificator, yerr * modificator)).T,
                   header='dE N yerr', comments='')

    domain = (drho > domain_borders[0]) & (drho < domain_borders[1])
    domain = domain if np.count_nonzero(domain) > 1 else True
    lr = linregress(np.log10(drho)[domain], np.log10(N/drho)[domain])
    return lr.slope, lr.rvalue**2



def plot_spectra(save=False):
    domains = [
        (1e-2, 2e-1),
        (1e-2, 5e0),
        (1e-2, 5e0),
        (8e-1, 1e1),
        (2e0, 3e1),
        (4e0, 5e1),
    ]
    plt.figure()
    # ax_idx = 221
    ax_idx = 231
    # for b in B_RMS:
    for r, d in zip(RADII, domains):
        plt.subplot(ax_idx)
        plt.loglog(nonposx='clip', nonposy='clip')
        ax_idx += 1
        print('R = %.2f Mpc' % (r / Mpc))
        # for r in RADII:
        for b in B_RMS:
            slope, r_sq = plot_spectrum(b, r, d, save)
            print('\tBrms = %.0e nG:\talpha = %f, r_sq = %f'
                    % (b, slope, r_sq))
        plt.legend()


def plot_traj(B_rms, radius, save, plot=True):
    data = np.genfromtxt(OUTFILENAME % (B_rms, radius), names=True)
    traj = data['D']

    if plot:
        N, bins, _ = plt.hist(np.log10(traj), bins=50, density=True,
                              histtype='step', label='Brms = %.0e nG' % B_rms)

    if save:
        N_save = np.hstack((0, N, 0))
        N_save[N_save == 0] = 1e-4
        dbin = bins[1] - bins[0]
        bins_save = np.hstack((bins[0]-dbin, bins))
        np.savetxt(SAVEFILENAME % ('traj', B_rms, radius),
                   # np.vstack((np.hstack((N, N[-1])), bins)).T,
                   np.vstack((N_save, bins_save)).T,
                   header='N bins', comments='')

    return traj.mean(), traj.std()


def plot_defl(B_rms, radius, save, plot):
    def dot(x, y):
        return np.sum(x * y, axis=0)
    def norm(x):
        return np.linalg.norm(x, ord=2, axis=0)

    data = np.genfromtxt(OUTFILENAME % (B_rms, radius), names=True)
    p, p0 = np.array([data['Px'], data['Py'], data['Pz']]), \
            np.array([data['P0x'], data['P0y'], data['P0z']])
    angle = np.rad2deg(np.arccos(dot(p, p0) / (norm(p) * norm(p0))))

    if plot:
        N, bins, _ = plt.hist(angle, bins=50, density=True, histtype='step',
                              label='Brms = %.0e nG' % B_rms)

    if save:
        N_save = np.hstack((0, N, 0))
        N_save[N_save == 0] = 1e-6
        dbin = bins[1] - bins[0]
        bins_save = np.hstack((bins[0]-dbin, bins))
        np.savetxt(SAVEFILENAME % ('defl', B_rms, radius),
                   # np.vstack((np.hstack((N, N[-1])), bins)).T,
                   np.vstack((N_save, bins_save)).T,
                   header='N bins', comments='')

    return angle.mean(), angle.std()


def plot_defl_or_traj(which='traj', save=False):
    f, unit = {
        'traj'  :   (plot_traj, 'Mpc'),
        'defl'  :   (plot_defl, 'deg'),
    }.get(which)

    plt.figure()
    ax_idx = 231
    for r in RADII:
        plt.subplot(ax_idx,
                    title='r = %.2f Mpc' % (r / Mpc),
                    yscale='log')
        ax_idx += 1
        print('R = %.2f Mpc' % (r / Mpc))
        for b in B_RMS:
            mean, std = f(b, r, save)
            print('\tB_rms = %.0e nG:\t(%.2f +- %.2f) %s' % (b, mean, std, unit))
        plt.legend()


def plot_mean_defl_or_traj(which='traj'):
    f = {'traj' : plot_traj, 'defl' : plot_defl}.get(which)

    res = np.zeros((RADII.size, B_RMS.size, 2))

    plt.figure()
    plt.subplot(111)

    for j, b in enumerate(B_RMS):
        for i, r in enumerate(RADII):
            res[i,j] = f(b, r, False, False)
        plt.errorbar(RADII, res[:,j,0], res[:,j,1])


def plot_trajectories():
    import glob
    from mpl_toolkits.mplot3d import Axes3D
    data_files = glob.glob('output/deflections*one-particle_full-traj*.txt')
    plt.figure()
    ax = plt.subplot(111, projection='3d', aspect='equal')
    for df in data_files:
        data = np.genfromtxt(df, names=True)
        x, y, z = data['X'], data['Y'], data['Z']
        ax.plot(x, y, z)
    x0, y0, z0 = CENTER / Mpc
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    X = 24 * np.outer(np.cos(u), np.sin(v))
    Y = 24 * np.outer(np.sin(u), np.sin(v))
    Z = 24 * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(X+x0, Y+y0, Z+z0, rstride=10, cstride=10, color='tab:red')


if __name__ == '__main__':
    SAVE = False
    # print('turbulent correlation length: %e kpc\n' % L_c)
    # for b in B_RMS:
    #     run(b)
    plot_spectra(SAVE)
    # plot_defl_or_traj('traj', SAVE)
    # plot_defl_or_traj('defl', SAVE)
    # plot_trajectories()
    # plot_mean_defl_or_traj('traj')
    # plot_mean_defl_or_traj('defl')
    plt.show()



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
