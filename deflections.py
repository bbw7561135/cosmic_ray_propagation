#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from crpropa import *


SEED = 100
NSIM = 1000

OUTFILENAME = 'output/deflections-%.0e.txt'


A, Z = 1, 1
EMIN, EMAX = 1e18, 1e21
LMIN, LMAX = 150*kpc, 2000*kpc
SPECTRAL_INDEX_BFIELD = -11/3
SPECTRAL_INDEX_FERMI = -2
CENTER = (128, 128, 128)
B_RMS = np.array([50, 10, 1]) * nG



def run(B_rms):
    vgrid = VectorGrid(Vector3d(0), 256, 75*kpc)
    initTurbulence(vgrid, B_rms, LMIN, LMAX, SPECTRAL_INDEX_BFIELD, SEED)
    Bfield = MagneticFieldGrid(vgrid)
    l_c = turbulentCorrelationLength(LMIN, LMAX, SPECTRAL_INDEX_BFIELD)
    radii = l_c * np.array([.05, .5, 1., 5., 50.])
    Dmax = radii[-1] * 1e3

    print('turbulent correlation length: %e kpc\n' % l_c)

    m = ModuleList()
    m.add(PropagationCK(Bfield))

    output = TextOutput(OUTFILENAME % B_rms, Output.Event3D)
    output.enable(Output.SourceDirectionColumn)

    for radius in radii:
        obs = Observer()
        obs.add(ObserverLargeSphere(Vector3d(*CENTER), radius))
        obs.setDeactivateOnDetection(False)
        obs.onDetection(output)
        m.add(obs)

    m.add(MaximumTrajectoryLength(Dmax))

    source = Source()
    source.add(SourceParticleType(nucleusId(A, Z)))
    source.add(SourcePowerLawSpectrum(EMIN, EMAX, SPECTRAL_INDEX_FERMI))
    source.add(SourcePosition(Vector3d(*CENTER)))
    source.add(SourceIsotropicEmission())

    m.setShowProgress(True)
    m.run(source, NSIM, True)


for b in B_RMS:
    run(b)



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
