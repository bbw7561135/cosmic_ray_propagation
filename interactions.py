#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, Counter
from crpropa import *


NSIM = 10000

OUTFILENAME = 'output/interactions-%s.txt'
SAVEFILENAME = 'plotdata/interactions-%s.csv'


SPECTRAL_INDICES = {
    'flat' : -1,        
    'fermi' : -2,
}

EMIN, EMAX = 1e17 * eV, 1e21 * eV

CANDIDATES = OrderedDict([
    ('H', {'A' : 1, 'Z' : 1}),
    ('Fe56', {'A' : 56, 'Z' : 26}),
])

INTERACTIONS = OrderedDict([
    ('Redshift', Redshift()),
    ('PhotoPionCMB', PhotoPionProduction(CMB)),
    ('PhotoPionIRB', PhotoPionProduction(IRB_Gilmore12)),
    ('PhotoDisCMB', PhotoDisintegration(CMB)),
    ('PhotoDisIRB', PhotoDisintegration(IRB_Gilmore12)),
    ('ElectronPairCMB', ElectronPairProduction(CMB)),
    ('ElectronPairIRB', ElectronPairProduction(IRB_Gilmore12)),
])

DISTANCES = OrderedDict([
    ('close', (0, 10*Mpc)),
    ('far', (100*Mpc, 1000*Mpc)),
])



def run_all():
    # iterate the candidate types
    for c_name, c_specs in CANDIDATES.items():

        # iterate the distance regimes
        for dist_name, dist_spec in DISTANCES.items():

            # build id string
            descr = '%s-%s-%%s' % (c_name, dist_name)

            # perform one run without interaction
            run(c_specs, None, dist_spec, descr % 'free',
                with_nuclear_decay=False if c_name == 'H' else True)

            # iterate interactions
            for i_name, i in INTERACTIONS.items():
                run(c_specs, i, dist_spec, descr % i_name,
                    with_nuclear_decay=False if c_name == 'H' else True)



def run(c_specs, interaction, distances, description, with_nuclear_decay=True):
    print(description, '\n')

    m = ModuleList()
    m.add(SimplePropagation(1*kpc, 10*Mpc))
    m.add(MinimumEnergy(EMIN))
    if interaction is not None:
        print('adding interaction %s' % interaction)
        m.add(interaction)
        if with_nuclear_decay:
            m.add(NuclearDecay())

    obs = Observer()
    obs.add(ObserverPoint())    # observer at x = 0
    output = TextOutput(OUTFILENAME % description, Output.Event1D)
    obs.onDetection(output)
    m.add(obs)

    source = Source()
    candidate = nucleusId(c_specs['A'], c_specs['Z'])
    source.add(SourceParticleType(candidate))
    source.add(SourcePowerLawSpectrum(EMIN, c_specs['Z']*EMAX, SPECTRAL_INDICES['flat']))
    source.add(SourceUniform1D(*distances))
    source.add(SourceRedshift1D())

    m.setShowProgress(True)
    m.run(source, NSIM, True)



def plot_all(save=False):
    plt.figure()
    ax_idx = 221

    for c_name in CANDIDATES.keys():
        # if c_name == 'H':
        #     continue
        for dist_name in DISTANCES.keys():
        # for i_name in INTERACTIONS.keys():
            plt.subplot(ax_idx)
            # plt.loglog(nonposx='clip', nonposy='clip')
            # plt.semilogy()
            plt.semilogx()
            ax_idx += 1
            descr = '%s-%s-%%s' % (c_name, dist_name)

            plot_interaction(descr % 'free', save)
            for i_name in INTERACTIONS.keys():
            # for dist_name in DISTANCES.keys():
                plot_interaction(descr % i_name, save)
                # plot_traj_length(descr % i_name)
                # loss_time_scale(descr % i_name, dist_name)
                # rel_energy_loss(descr % i_name)
                # descr = '%s-%s-%%s' % (c_name, dist_name)
                # decay_products(descr % i_name)

            plt.legend()
            # print()

    if not save:
        plt.show()
        # pass



def plot_interaction(descr, save):
    data = np.genfromtxt(OUTFILENAME % descr, names=True)
    E, E0 = data['E'], data['E0']
    weights = E0**(-SPECTRAL_INDICES['flat'] + SPECTRAL_INDICES['fermi'])
    N_noweights, _ = np.histogram(np.log10(E))
    N, bins = np.histogram(np.log10(E), weights=weights, density=True)
    dE = (10**bins[1:] + 10**bins[:-1]) / 2.
    yerr = N * dE / np.sqrt(N_noweights)
    plt.errorbar(dE / eV, N * dE, yerr, label=descr)

    if save:
        np.savetxt(SAVEFILENAME % descr,
                   np.vstack((dE/eV, N*dE, yerr)).T,
                   header='dE N yerr', comments='')


def loss_time_scale(descr, dist):
    data = np.genfromtxt(OUTFILENAME % descr, names=True)
    D, E, E0 = data['D'], data['E'], data['E0']
    m = np.array([nuclearMass(int(i)) for i in data['ID']])
    weights = E0**(-SPECTRAL_INDICES['flat'] + SPECTRAL_INDICES['fermi'])
    # t = np.sqrt(m / np.abs(E0 - E)) * DISTANCES[dist][1] / 60 / 60 / 24 / 365.25
    # print('%s:\tmin = %e' % (descr, t.min()))
    max_idx = np.argmax(np.abs(E0 - E))
    # t = np.sqrt(m[max_idx] / np.abs(E0 - E)[max_idx]) * D[max_idx] * Mpc / 60 / 60 / 24 / 365.25
    t = np.sqrt(m[max_idx] / np.abs(E0 - E)[max_idx]) * 100*Mpc / 60 / 60 / 24 / 365.25
    print('%s:\tmin = %f' % (descr, t))


def decay_products(descr):
    from pprint import pprint
    elements = np.array([
        'n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
        'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
        'Mn', 'Fe',
    ])
    def id_to_nucleus(i):
        i = str(i)
        A = i[-3:-1]
        Z = i[-6:-4]
        n = elements[int(Z)]
        return n + A
    data = np.genfromtxt(OUTFILENAME % descr, names=True, dtype=int)
    ids = data['ID']
    stats_raw  = Counter(ids)
    total = ids.size
    Fe56 = stats_raw.pop(nucleusId(56, 26))
    print('\n' + descr)
    print('decayed nuclei: %f %%' % ((1 - Fe56 / NSIM) * 100))
    if len(stats_raw) != 0:
        stats_arr = np.array(sorted(zip([id_to_nucleus(i) for i in
            stats_raw.keys()], stats_raw.values()), key=lambda t: t[1]))
        percent = stats_arr[:,1].astype(int) / (total - Fe56) * 100
        pprint(list(zip(stats_arr[:,0][percent>1], percent[percent>1])))
        print('traces: %f %%' % np.sum(percent[percent<1]))


# run_all()
plot_all(False)



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
