#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from crpropa import *


SEED = 100
NSIM = 10000

OUTFILENAME = 'output/constraints_brms-%.0e_r-%.0e.txt'

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


for b in B_RMS:
    run(b)


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
