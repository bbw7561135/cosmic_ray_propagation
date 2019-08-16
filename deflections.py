#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from crpropa import *


SEED = 100
NSIM = 10000

OUTFILENAME = 'output/deflections_brms-%.0e_r-%.0e.txt'
LABEL = r'$B_{rms}=%.0e$, $r=%.0e$'


A, Z = 1, 1
EMIN, EMAX = 1e18*eV, 1e21*eV
LMIN, LMAX = 150*kpc, 2000*kpc
SPECTRAL_INDEX_BFIELD = -11/3
SPECTRAL_INDEX_FERMI = -2
SPECTRAL_INDEX_FLAT = -1
CENTER = (128, 128, 128)
B_RMS = np.array([50, 10, 1]) * nG


L_c = turbulentCorrelationLength(LMIN, LMAX, SPECTRAL_INDEX_BFIELD)
RADII = L_c * np.array([.05, .5, 1., 5., 50.])
DMAX = RADII[-1] * 1e2



def run(B_rms):
    vgrid = VectorGrid(Vector3d(0), 256, 75*kpc)
    initTurbulence(vgrid, B_rms, LMIN, LMAX, SPECTRAL_INDEX_BFIELD, SEED)
    Bfield = MagneticFieldGrid(vgrid)

    m = ModuleList()
    m.add(PropagationCK(Bfield))

    for radius in RADII:
        obs = Observer()
        obs.add(ObserverLargeSphere(Vector3d(*CENTER), radius))
        obs.setDeactivateOnDetection(False)
        output = TextOutput(OUTFILENAME % (B_rms, radius), Output.Event3D)
        output.enable(Output.SourceDirectionColumn)
        obs.onDetection(output)
        m.add(obs)

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
    dE = 10**bins[1:] - 10**bins[:-1]
    dR_L = dE / (sc.c * sc.e * B_rms)
    drho = dR_L / L_c
    yerr = N / drho / np.sqrt(N_noweights)
    plt.errorbar(drho, N / drho, yerr, label=LABEL % (B_rms, radius))



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
    angle = np.arccos(dot(p, p0) / (norm(p) * norm(p0)))
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
                    % (r, np.rad2deg(res[0]), np.rad2deg(res[1]), res[2], res[3]))
        print()


# TODO: find and mark area of resonance, interpolate


if __name__ == '__main__':
    # print('turbulent correlation length: %e kpc\n' % L_c)
    for b in B_RMS:
        run(b)
    plot_spectra()
    plot_deflection_and_traj()
    plt.show()



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
