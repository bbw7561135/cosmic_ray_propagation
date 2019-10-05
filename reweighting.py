#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from crpropa import *


OUTFILENAME = 'output/reweighting-alpha-%+.2f.txt'
SAVEFILENAME = 'plotdata/reweighting-%s-alpha-%+.2f-alpha_new-%s.csv'

A, Z = 1, 1     # protons
ZMIN, ZMAX = 0, 2
EMIN, EMAX = 1e18, 1e21
NSIM = 1000

Dmin = redshift2ComovingDistance(ZMIN)
Dmax = redshift2ComovingDistance(ZMAX)


def run(spectral_index):
    m = ModuleList()
    m.add(
        SimplePropagation(
            1*kpc,      # min step
            10*Mpc      # max step
        )
    )

    obs = Observer()
    obs.add(ObserverPoint())    # observer at x = 0
    output = TextOutput(OUTFILENAME % spectral_index, Output.Event1D)
    obs.onDetection(output)
    m.add(obs)

    source = Source()
    source.add(SourceParticleType(nucleusId(A, Z)))
    source.add(SourcePowerLawSpectrum(EMIN, EMAX, spectral_index))
    source.add(SourceUniform1D(Dmin, Dmax))
    source.add(SourceRedshift1D())

    m.setShowProgress(True)
    m.run(source, NSIM, True)


def plot_hist(spectral_index, new_spectral_index=None, save=False):
    data = np.genfromtxt(OUTFILENAME % spectral_index, names=True)
    E, E0 = data['E'], data['E0']

    title = r'$\alpha=%+.1f$' % spectral_index
    if new_spectral_index is not None:
        weights = E0**(-spectral_index + new_spectral_index)
        title += r' - reweighted with $\alpha_{new}=%+.1f$' % new_spectral_index
    else:
        weights = None

    plt.figure()
    plt.suptitle(title)
    plt.subplot(211)
    plt.semilogy(nonposy='clip')
    N, bins, _ = plt.hist(np.log10(E), weights=weights, density=True)
    plt.subplot(212)
    plt.semilogy(nonposy='clip')
    plt.hist(np.log10(E0), weights=weights, density=True)

    if save:
        np.savetxt(SAVEFILENAME % ('hist', spectral_index,
                                   '%+.2f' % new_spectral_index
                                        if new_spectral_index is not None
                                        else None),
                   np.vstack((np.hstack((N, N[-1])), bins)).T,
                   header='N bins', comments='')


def plot_diff(spectral_index, new_spectral_index=None, save=False):
    data = np.genfromtxt(OUTFILENAME % spectral_index, names=True)
    E, E0 = data['E'], data['E0']

    title = r'$\alpha=%+.1f$' % spectral_index
    if new_spectral_index is not None:
        weights = E0**(-spectral_index + new_spectral_index)
        title += r' - reweighted with $\alpha_{new}=%+.1f$' % new_spectral_index
    else:
        weights = None

    N_noweights, _ = np.histogram(np.log10(E), bins=10)
    N, bins = np.histogram(np.log10(E), weights=weights, density=True, bins=10)
    dE = (10**bins[1:] + 10**bins[:-1]) / 2.
    yerr = N / dE / np.sqrt(N_noweights)

    plt.figure()
    plt.suptitle(title)
    plt.loglog(nonposx='clip', nonposy='clip')
    plt.errorbar(dE, N / dE, yerr)

    if save:
        np.savetxt(SAVEFILENAME % ('errbar', spectral_index,
                                   '%+.2f' % new_spectral_index
                                        if new_spectral_index is not None
                                        else None),
                   np.vstack((np.log10(dE), N / dE, yerr)).T,
                   header='dE dNdE yerr', comments='')



if __name__ == '__main__':
    SAVE = False

    # run(-1)
    # run(-2)

    # plot_hist(-1, save=SAVE)
    # plot_hist(-2, save=SAVE)
    # plot_hist(-1, -2, save=SAVE)

    plot_diff(-1, save=SAVE)
    plot_diff(-2, save=SAVE)
    plot_diff(-1, -2, save=SAVE)

    # plt.show()


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
