#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import scipy.odr as odr
from scipy.stats import linregress
from scipy.optimize import curve_fit
from crpropa import *


SEED = 100
NSIM = 10000

OUTFILENAME = 'output/constraints_brms-%.0e_r-%.0e_high-energy.txt'
LABEL = r'$B_{rms}=%.0e$, $r=%.0e$'

OBSERVER_RADIUS = 25 * kpc
OBSERVER_AREA = 4. * np.pi * OBSERVER_RADIUS**2

A, Z = 1, 1
EMIN, EMAX = 1e18*eV, 1e21*eV
LMIN, LMAX = 150*kpc, 2000*kpc
SPECTRAL_INDEX_BFIELD = -11/3
SPECTRAL_INDEX_FERMI = -2.
SPECTRAL_INDEX_FLAT = -1
CENTER = np.array([128, 128, 128]) * 75*kpc
B_RMS = np.array([50, 20, 10, 1]) * nG

L_c = turbulentCorrelationLength(LMIN, LMAX, SPECTRAL_INDEX_BFIELD)
 RADII = L_c * np.array([.05, .5, 1., 5., 50., 100.])
DMAX = RADII[-1] * 1e2
DC = 13.8e9 * 60 * 60 * 24 * 365.25 * sc.c / Mpc

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
    vgrid.setReflective(True)
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


def compute_spectral_index(dE, dF, xerr, yerr):
    def f(B, x):
        return -B[0]*x + B[1]

    domain = (dE > 5e18) & (dE < 30e18)
    log_E = np.log10(dE[domain])
    log_F = np.log10(dF[domain])
    data = odr.RealData(log_E, log_F, xerr, yerr)
    model = odr.Model(f)
    odr_obj = odr.ODR(data, model, beta0=[2., 0.])
    # odr_obj.set_iprint(final=2)
    out = odr_obj.run()

    return out.beta[0], np.sqrt(np.diag(out.cov_beta))[0]


def plot_spectrum(B_rms, radius, alpha0=SPECTRAL_INDEX_FERMI):
    def dot(x, y):
        return np.sum(x * y, axis=0)
    def norm(x):
        return np.linalg.norm(x, ord=2, axis=0)

    data = np.genfromtxt(OUTFILENAME % (B_rms, radius), names=True)
    E, E0 = data['E'], data['E0']
    d = data['D']
    r = np.array([data['X'], data['Y'], data['Z']]) \
            - np.array([data['X0'], data['Y0'], data['Z0']])
    p = np.array([data['Px'], data['Py'], data['Pz']])
    theta = np.arccos(dot(r, p) / (norm(r) * norm(p)))
    print(np.rad2deg(theta.max()))

    weights = E0**(-SPECTRAL_INDEX_FLAT + alpha0) \
            * 1. / np.abs(np.cos(theta)) * 1. / np.sin(theta)
    N_noweights, _ = np.histogram(np.log10(E), bins=20)
    N, bins = np.histogram(np.log10(E), weights=weights, bins=20, density=True)

    dE = (10**bins[1:] + 10**bins[:-1]) / 2.
    dF = N * radius**2 / OBSERVER_AREA**2
    yerr = dF / np.sqrt(N_noweights)

    alpha, alpha_std = compute_spectral_index(dE / eV, dF,
                                              bins[1:]-bins[:-1], yerr)
    domain = (E/eV > 5e18) & (E/eV < 3e19)
    theta_ = np.rad2deg(theta[domain])

    th_mean, th_std = theta_.mean(), theta_.std()
    d_mean, d_std = d.mean(), d.std()

    # if ((alpha-alpha_std) < 2.5 < (alpha+alpha_std)) \
    #         and ((th_mean-th_std) < 45. < (th_mean+th_std)):
    print('B_rms = %.0f nG\tR = %.0f kpc:\talpha = (%.3f +- %.3f)'
              '\tdefl = (%.2f +- %.2f) Degree\ttraj = (%.2f +- %.2f) Mpc'
              % (B_rms / nG, radius / kpc, alpha, alpha_std,
                 th_mean, th_std, d_mean, d_std))
        # if d_mean + d_std > DC:
        #     print('\t[WARN] trajectory larger than age of universe!')

    plt.errorbar(dE / eV, dF, yerr, # label=LABEL % (B_rms, radius))
                 label=r'$\alpha$ = (%.3f $\pm$ %.3f)' % (alpha, alpha_std))


def plot_spectra(alpha0=SPECTRAL_INDEX_FERMI):
    plt.figure()
    ax_idx = 221
    for b in B_RMS:
        # print('\nB_rms = %.0f nG' % (b / nG))
        plt.subplot(ax_idx)
        plt.loglog(nonposx='clip', nonposy='clip')
        ax_idx += 1
        for r in RADII:
            plot_spectrum(b, r, alpha0)
        plt.legend()


# for b in B_RMS:
#     run(b)
# for alpha0 in -np.arange(1.5, 2.6, .1):
#     print('\n\nenergy spectral index = %.1f' % alpha0)
#     plot_spectra(alpha0)
if __name__ == '__main__':
    plot_spectra()


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
