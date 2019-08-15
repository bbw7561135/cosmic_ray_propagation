#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from crpropa import *


NSIM = 1000

OUTFILENAME = 'output/interactions-%s.txt'


SPECTRAL_INDICES = {
    'flat' : -1,        
    'fermi' : -2,
}

EMIN, EMAX = 1e18 * eV, 1e21 * eV

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
    ('far', (100*Mpc, 5000*Mpc)),
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



def plot_all():
    plt.figure()
    ax_idx = 221

    for c_name in CANDIDATES.keys():
        for dist_name in DISTANCES.keys():
            plt.subplot(ax_idx)
            plt.loglog(nonposx='clip', nonposy='clip')
            ax_idx += 1
            descr = '%s-%s-%%s' % (c_name, dist_name)

            plot_interaction(descr % 'free')
            for i_name in INTERACTIONS.keys():
                plot_interaction(descr % i_name)

            plt.legend()

    plt.show()



def plot_interaction(descr):
    data = np.genfromtxt(OUTFILENAME % descr, names=True)
    E, E0 = data['E'], data['E0']
    weights = E0**(-SPECTRAL_INDICES['flat'] + SPECTRAL_INDICES['fermi'])
    N_noweights, _ = np.histogram(np.log10(E))
    N, bins = np.histogram(np.log10(E), weights=weights)
    dE = 10**bins[1:] - 10**bins[:-1]
    yerr = N / dE / np.sqrt(N_noweights)
    plt.errorbar(dE, N / dE, yerr, label=descr)



# run_all()
plot_all()



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
