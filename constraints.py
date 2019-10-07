#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import scipy.odr as odr
from scipy.stats import linregress
from scipy.optimize import curve_fit
from collections import OrderedDict
from crpropa import *

plt.rcParams['axes.grid'] = True


SEED = 100
NSIM = 10000

# OUTFILENAME = 'output/constraints_brms-%.0e_r-%.0e_high-energy.txt'
OUTFILENAME = 'output/constraints_brms-%.0e_r-%.0e.txt'
SAVEFILENAME = 'plotdata/constraints_%s_brms-%.0e_r-%.0e.csv'
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
# B_RMS = np.array([50, 20, 10]) * nG

L_c = turbulentCorrelationLength(LMIN, LMAX, SPECTRAL_INDEX_BFIELD)
RADII = L_c * np.array([.05, .5, 1., 5., 50., 100.])
# RADII = L_c * np.array([5., 50., 100.])
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



def compute_spectral_index(dE, dF, xerr, yerr, limits):
    def red_chi_sq_test(x, y, f, B, yerr):
        return np.sum((y - f(B, x))**2 / yerr**2) / (len(y) - len(B))

    def f(B, x):
        return -B[0]*x + B[1]

    domain = (dE > limits[0]*1e18) & (dE < limits[1]*1e18)
    log_E = np.log10(dE[domain])
    log_F = np.log10(dF[domain])
    yerr = (yerr/dF)[domain]
    xerr = xerr[domain]

    data = odr.RealData(log_E, log_F, xerr, yerr)
    model = odr.Model(f)
    odr_obj = odr.ODR(data, model, beta0=[2., 0.])
    # odr_obj.set_iprint(final=2)
    out = odr_obj.run()

    return (
        out.beta,
        out.sd_beta[0],
        np.sqrt(np.diag(out.cov_beta))[0],
        red_chi_sq_test(log_E, log_F, f, out.beta, yerr)
    )


def plot_spectrum(B_rms, radius, axes, alpha0=SPECTRAL_INDEX_FERMI):
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
    # print(np.rad2deg(theta.max()))

    weights = E0**(-SPECTRAL_INDEX_FLAT + alpha0) \
            * 1. / np.abs(np.cos(theta)) * 1. / np.sin(theta)
    N_noweights, _ = np.histogram(np.log10(E), bins=20)
    N, bins = np.histogram(np.log10(E), weights=weights, bins=20, density=True)

    dE = (10**bins[1:] + 10**bins[:-1]) / 2.
    dF = N * radius**2 / OBSERVER_AREA**2
    yerr = dF / np.sqrt(N_noweights)

    # alpha, alpha_std = compute_spectral_index(dE / eV, dF,
    #                                           bins[1:]-bins[:-1], yerr)
    # domain = (E/eV > 5e18) & (E/eV < 3e19)
    # theta_ = np.rad2deg(theta[domain])

    # th_mean, th_std = theta_.mean(), theta_.std()
    # d_mean, d_std = d.mean(), d.std()

    # if ((alpha-alpha_std) < 2.5 < (alpha+alpha_std)) \
    #         and ((th_mean-th_std) < 45. < (th_mean+th_std)):
    # print('B_rms = %.0f nG\tR = %.0f kpc:\talpha = (%.3f +- %.3f)'
    #           '\tdefl = (%.2f +- %.2f) Degree\ttraj = (%.2f +- %.2f) Mpc'
    #           % (B_rms / nG, radius / kpc, alpha, alpha_std,
    #              th_mean, th_std, d_mean, d_std))

    axes[0].errorbar(dE / eV, dE**2.5*dF, dE**2.5*yerr,
    # axes[0].errorbar(dE / eV, dF, yerr,
            label='Brms=%.0f nG' % (B_rms/nG))
    # axes[1].hist(np.log10(d), bins=50, density=True, histtype='step',
    #         label='Brms=%.0e nG' % (B_rms/nG))
    # axes[2].hist(np.rad2deg(theta), bins=50, density=True, histtype='step',
    #         label='Brms=%.0e nG' % (B_rms/nG))


def plot_spectra(alpha0=SPECTRAL_INDEX_FERMI):
    spec_fig, _ = plt.subplots(1, 3)
    # traj_fig, _ = plt.subplots(2, 3)
    # defl_fig, _ = plt.subplots(2, 3)

    for i, r in enumerate(RADII[3:]):
        # print('\nB_rms = %.0f nG' % (b / nG))
        # axes = [spec_fig.axes[i], traj_fig.axes[i], defl_fig.axes[i]]
        axes = [spec_fig.axes[i]]
        for ax in axes:
            ax.set_title('R = %.2f Mpc' % (r / Mpc))
        axes[0].loglog()
        # axes[1].semilogy()
        # axes[2].semilogy()
        for b in B_RMS:
            plot_spectrum(b, r, axes, alpha0)
    plt.legend()



DOMAIN_DICT = OrderedDict((
    (5.*L_c, {
        1*nG  : ((-np.inf, 3.5), (2.5, +np.inf)),
        10*nG : ((3, 40), (30, 300)),
        50*nG : ((8, 150), (150, +np.inf)),
    }),
    (50.*L_c, {
        1*nG  : ((2, 10), (10, 200)),
        # 10*nG : ((7, 70), (170, 400)),
        10*nG : ((7, 100), (170, 400)),
        50*nG : ((30, 300), (250, 400)),
    }),
    (100.*L_c, {
        # 1*nG  : ((2, 20), (15, 200)),
        1*nG  : ((2, 15), (15, 200)),
        10*nG : ((15, 100), (180, 400)),
        50*nG : ((20, 300), (200, 400)),
    }),
))
GLOBAL_LIMIT = (1, 500) # EeV


def analyse_spectrum(radius, B_rms, ax, lim1, lim2, alpha0,
        plot_defl=False, save=False):
    def dot(x, y):
        return np.sum(x * y, axis=0)
    def norm(x):
        return np.linalg.norm(x, ord=2, axis=0)

    data = np.genfromtxt(OUTFILENAME % (B_rms, radius), names=True)
    E, E0 = data['E'], data['E0']
    r = np.array([data['X'], data['Y'], data['Z']]) \
            - np.array([data['X0'], data['Y0'], data['Z0']])
    p = np.array([data['Px'], data['Py'], data['Pz']])
    theta = np.arccos(dot(r, p) / (norm(r) * norm(p)))

    whole_domain = (E > GLOBAL_LIMIT[0]*1e18*eV) & (E < GLOBAL_LIMIT[1]*1e18*eV)
    E = E[whole_domain] / eV
    E0 = E0[whole_domain] / eV
    theta = theta[whole_domain]
    domain1 = (E > lim1[0]*1e18) & (E < lim1[1]*1e18)
    # domain2 = (E > lim2[0]*1e18*eV) & (E < lim2[1]*1e18*eV)


    if plot_defl:
        theta_ = np.rad2deg(theta[domain1])
        N, bins, _ = ax.hist(theta_, bins=50, density=True, histtype='step',
                label='Brms=%.0f nG' % (B_rms/nG))
        print('\tBrms = %.0f nG: theta = %.2f +- %.2f' % (B_rms/nG,
            theta_.mean(), theta_.std()))

        if save:
            N_save = np.hstack((0, N, 0))
            N_save[N_save == 0] = 1e-6
            dbin = bins[1] - bins[0]
            bins_save = np.hstack((bins[0]-dbin, bins))
            np.savetxt(SAVEFILENAME % ('defl', B_rms, radius),
                    np.vstack((N_save, bins_save)).T,
                    header='N bins', comments='')

        return None


    weights = E0**(-SPECTRAL_INDEX_FLAT + alpha0) \
            * 1. / np.abs(np.cos(theta)) * 1. / np.sin(theta)
    N_noweights, _ = np.histogram(np.log10(E), bins=20)
    N, bins = np.histogram(np.log10(E), weights=weights, bins=20, density=True)

    dE = (10**bins[1:] + 10**bins[:-1]) / 2.
    dF = N * radius**2 / OBSERVER_AREA**2
    yerr = dF / np.sqrt(N_noweights)

    (a1_f, b1_f), a1_sd1, a1_sd2, red_chi1  = \
            compute_spectral_index(dE, dF, (bins[1:]-bins[:-1])/np.sqrt(12),
                    yerr, lim1)
    # (a2_f, b2_f), a2_sd1, a2_sd2, red_chi2  = \
    #         compute_spectral_index(dE, dF, (bins[1:]-bins[:-1])/np.sqrt(12),
    #                 yerr, lim2)
    # x = np.array(GLOBAL_LIMIT) * 1e18
    x = np.logspace(18, 20.65, 20)

    print('\tBrms = %.0f nG' % (B_rms / nG))
    print('\t1st:\t%f\t%f\t%f\t%f' % (a1_f, b1_f, a1_sd1, red_chi1))
    # print('\t2nd:\t%f\t%f\t%f\t%f' % (a2_f, a2_sd1, a2_sd2, red_chi2))
    print()

    ax.errorbar(dE, dE**2.5*dF, dE**2.5*yerr, label='Brms=%.0f nG' % (B_rms/nG))
    ax.plot(x, dE**2.5*x**(-a1_f)*10**b1_f, c='gray')
    # ax.plot(x, dE**2.5*x**(-a2_f)*10**b2_f, c='gray')

    if save:
        np.savetxt(SAVEFILENAME % ('spectrum-%.1f' % alpha0, B_rms, radius),
                np.vstack((dE, dE**2.5*dF, dE**2.5*yerr)).T,
                header='dE N yerr', comments='')



def analyse_spectra(alpha0=SPECTRAL_INDEX_FERMI, plot_defl=False, save=False):
    fig, axes = plt.subplots(1, 3)

    for i, ((r, b_dict), ax) in enumerate(zip(DOMAIN_DICT.items(), axes)):
        print('R = %.2f Mpc' % (r / Mpc))
        ax.set_title('R = %.2f Mpc' % (r / Mpc))
        if plot_defl:
            ax.semilogy()
        else:
            ax.loglog()
        for b, (l1, l2) in b_dict.items():
            analyse_spectrum(r, b, ax, l1, l2, alpha0, plot_defl, save)
        print('\n===\n')
    plt.legend()


# for b in B_RMS:
#     run(b)
if __name__ == '__main__':
    SAVE = True
    analyse_spectra(-2., True, SAVE)
    # for a in -np.arange(2.0, 2.6, .1):
    for a in (-2., -2.6):
        print('\n+++\nprocessing alpha = %.1f ...\n' % a)
        analyse_spectra(a, False, SAVE)
        # plot_spectra(a)
        plt.gcf().suptitle('alpha = %.1f' % a)
        print()


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
