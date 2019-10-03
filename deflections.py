#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from crpropa import *


SEED = 100
NSIM = 10000
# NSIM = 1

OUTFILENAME = 'output/deflections_brms-%.0e_r-%.0e.txt'
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



def plot_spectrum(B_rms, radius):
    data = np.genfromtxt(OUTFILENAME % (B_rms, radius), names=True)
    E, E0 = data['E'], data['E0']
    weights = E0**(-SPECTRAL_INDEX_FLAT + SPECTRAL_INDEX_FERMI)
    N_noweights, _ = np.histogram(np.log10(E), bins=20)
    N, bins = np.histogram(np.log10(E), weights=weights, bins=20)
    dE = (10**bins[1:] + 10**bins[:-1]) / 2.
    dR_L = dE / (sc.c * sc.e * B_rms)
    drho = dR_L / L_c
    yerr = dE**2 * N / drho / np.sqrt(N_noweights)
    plt.errorbar(dE / eV, dE**2 * N / drho, yerr, label=LABEL % (B_rms, radius))



def plot_spectra():
    plt.figure()
    ax_idx = 221
    for b in B_RMS:
        plt.subplot(ax_idx)
        plt.loglog(nonposx='clip', nonposy='clip')
        ax_idx += 1
        for r in RADII:
            plot_spectrum(b, r)
        plt.legend()



def compute_deflection_and_traj(B_rms, radius):
    def dot(x, y):
        return np.sum(x * y, axis=0)
    def norm(x):
        return np.linalg.norm(x, ord=2, axis=0)

    data = np.genfromtxt(OUTFILENAME % (B_rms, radius), names=True)
    p, p0 = np.array([data['Px'], data['Py'], data['Pz']]), \
            np.array([data['P0x'], data['P0y'], data['P0z']])
    angle = np.rad2deg(np.arccos(dot(p, p0) / (norm(p) * norm(p0))))
    traj_len = data['D']

    return np.array([
        angle.mean(), angle.std(), traj_len.mean(), traj_len.std()
    ])



def plot_deflection_and_traj():
    for b in B_RMS:
        print('B_rms = %.0e nG' % b)
        for i, r in enumerate(RADII):
            res = compute_deflection_and_traj(b, r)
            print('\tR = %.0e:\tdefl = (%.2f +- %.2f) Degree\ttraj = (%.2f +- %.2f) Mpc' \
                    % (r, res[0], res[1], res[2], res[3]))
        print()


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
    pass
    # print('turbulent correlation length: %e kpc\n' % L_c)
    # for b in B_RMS:
    #     run(b)
    # plot_spectra()
    # plot_deflection_and_traj()
    plot_trajectories()
    plt.show()



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
