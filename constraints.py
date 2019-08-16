#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from crpropa import *


SEED = 100
NSIM = 10000

OUTFILENAME = 'output/constraints_brms-%.0e_r-%.0e.txt'

OBSERVER_RADIUS = 25 * kpc
OBSERVER_AREA = 4. * np.pi * OBSERVER_RADIUS**2

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

INTERACTIONS = [
    Redshift(),
    PhotoPionProduction(CMB),
    PhotoPionProduction(IRB_Gilmore12),
    PhotoDisintegration(CMB),
    PhotoDisintegration(IRB_Gilmore12),
    ElectronPairProduction(CMB),
    ElectronPairProduction(IRB_Gilmore12),
]


def run(B_rms):
    vgrid = VectorGrid(Vector3d(0), 256, 75*kpc)
    initTurbulence(vgrid, B_rms, LMIN, LMAX, SPECTRAL_INDEX_BFIELD, SEED)
    Bfield = MagneticFieldGrid(vgrid)

    m = ModuleList()
    m.add(PropagationCK(Bfield))
    m.add(MinimumEnergy(EMIN))
    m.add(MaximumTrajectoryLength(DMAX))

    for interaction in INTERACTIONS:
        m.add(interaction)

    for radius in RADII:
        obs = Observer()
        obs.add(ObserverLargeSphere(Vector3d(*CENTER), radius))
        obs.setDeactivateOnDetection(False)
        output = TextOutput(OUTFILENAME % (B_rms, radius), Output.Event3D)
        output.enable(Output.SourceDirectionColumn)
        obs.onDetection(output)
        m.add(obs)

    source = Source()
    source.add(SourceParticleType(nucleusId(A, Z)))
    source.add(SourcePowerLawSpectrum(EMIN, EMAX, SPECTRAL_INDEX_FLAT))
    source.add(SourcePosition(Vector3d(*CENTER)))
    source.add(SourceIsotropicEmission())

    m.setShowProgress(True)
    m.run(source, NSIM, True)


def plot_spectrum(B_rms, radius):
    def dot(x, y):
        return np.sum(x * y, axis=0)
    def norm(x):
        return np.linalg.norm(x, ord=2, axis=0)

    data = np.genfromtxt(OUTFILENAME % (B_rms, radius), names=True)
    E, E0 = data['E'], data['E0']
    r = np.array([data['X'], data['Y'], data['Z']])
    p = np.array([data['Px'], data['Py'], data['Pz']])
    theta = np.arccos(dot(r, p) / (norm(r) * norm(p)))

    weights = E0**(-SPECTRAL_INDEX_FLAT + SPECTRAL_INDEX_FERMI) \
            * 1. / np.abs(np.cos(theta)) * 1. / np.sin(theta)
    N_noweights, _ = np.histogram(np.log10(E), bins=20)
    N, bins = np.histogram(np.log10(E), weights=weights, bins=20)

    dE = 10**bins[1:] - 10**bins[:-1]
    dF = N * radius**2 / OBSERVER_AREA**2
    yerr = dF / np.sqrt(N_noweights)
    plt.errorbar(dE / eV, dF, yerr, label=LABEL % (B_rms, radius))


def plot_spectra():
    plt.figure()
    ax_idx = 221
    for b in B_RMS:
        plt.subplot(ax_idx)
        plt.loglog(nonposx='clip', nonposy='clip')
        ax_idx += 1
        for r in RADII:
            plot_spectrum(b, r)
        # plt.legend()


# for b in B_RMS:
#     run(b)
plot_spectra()


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
