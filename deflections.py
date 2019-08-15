#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from crpropa import *


SEED = 100
NSIM = 1000

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
DMAX = RADII[-1] * 1e3



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
    N_noweights, _ = np.histogram(np.log10(E), bins=10)
    N, bins = np.histogram(np.log10(E), weights=weights, bins=10)
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


def compute_deflection(B_rms, radius):
    data = np.genfromtxt(OUTFILENAME % (B_rms, radius), names=True)
    p, p0 = np.array([data['Px'], data['Py'], data['Pz']]), \
            np.array([data['P0x'], data['P0y'], data['P0z']])

    # TODO: mean, std, plotting


def compute_traj_len(B_rms, radius):
    data = np.genfromtxt(OUTFILENAME % (B_rms, radius), names=True)
    traj_len = data['D']
    # TODO: mean, std, plotting



if __name__ == '__main__':
    # print('turbulent correlation length: %e kpc\n' % L_c)
    # for b in B_RMS:
    #     run(b)
    plot_spectra()
    plt.show()



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
