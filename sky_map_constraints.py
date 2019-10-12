#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


PI = np.pi
PI_HALF = PI / 2
TWO_PI = 2*PI

OUTFILENAME = 'data/constraints_brms-%.0e_r-%.0e.txt'
RADII = np.array([7e20, 7e21, 1e22, 7e22, 7e23, 1e24])
BRMS = np.array([1e-13, 1e-12, 5e-12])


def plot_one_map(Brms, radius, map_ax, hist_ax, save=False):
    def cross(a, b):
        return np.cross(a, b, axis=0)
    def dot(a, b):
        return np.sum(a*b, axis=0)
    def norm(a):
        return np.linalg.norm(a, ord=2, axis=0)

    def rodrigues_rot(v, k, a):
        return v*np.cos(a) + cross(k, v)*np.sin(a) + k*dot(k, v)*(1-np.cos(a))

    try:
        data = np.genfromtxt(OUTFILENAME % (Brms, radius), names=True)
    except ValueError:
        print('[%s] - corrupted data, skipping...' % (OUTFILENAME % (Brms, radius)))
        return

    r = np.array([data['X'], data['Y'], data['Z']]) \
            - np.array([data['X0'], data['Y0'], data['Z0']])
    p = np.array([data['Px'], data['Py'], data['Pz']])
    # rot axis k = r cross e_x
    k = np.array([np.zeros_like(r[0]), r[2], -r[1]])
    k /= norm(k)
    # angle of rotation alpha = arccos(r dot e_x / abs(r) times abs(e_x))
    a = np.arccos(r[0] / norm(r))
    #  r_rot = rodrigues_rot(r, k, a)
    p_rot = rodrigues_rot(p, k, a)

    # CHEATING: `ObserverSphereLarge` only detects leaving particles causing an
    # angular distribution in [0, 90].
    # Multiply angles by 2 to mock complete isotropy.
    theta_p = 2*np.arctan2(p_rot[2], np.sqrt(p_rot[0]**2+p_rot[1]**2))
    phi_p = 2*np.arctan2(p_rot[1], p_rot[0])

    if map_ax is not None:
        h, xe, ye = np.histogram2d(
                phi_p, theta_p,
                #  bins=[360, 180],
                bins=[180, 90],
                range=[[-PI, PI], [-PI_HALF, PI_HALF]],
                density=True)
        h[h==0] = h[h!=0].min() / 10
        map_ax.pcolormesh(xe, ye, h.T,
                norm=colors.LogNorm())

    if hist_ax is not None:
        defl = 2*np.rad2deg(np.arccos(dot(r, p) / (norm(r)*norm(p))))
        N, bins, _ = hist_ax.hist(defl, bins=50, histtype='step')
        if save:
            N = np.hstack((0, N, 0))
            N[N==0] = N[N!=0].min() / 10
            dbin = bins[1] - bins[0]
            bins = np.hstack((bins[0]-dbin, bins))
            np.savetxt(
                'sky_map_hist_brms-%.0e_r-%.0e.csv' % (Brms, radius),
                np.vstack((N, bins)).T,
                header='N bins', comments='',
            )


def plot_all_maps(Brms):
    map_fig = plt.figure()
    hist_fig = plt.figure()
    plt.set_cmap('plasma')
    ax_idx = 321

    for r in RADII:
        map_ax = map_fig.add_subplot(ax_idx,
                                     projection='hammer')
                                     #  projection='mollweide')
        map_ax.axis('off')
        hist_ax = hist_fig.add_subplot(ax_idx)
        hist_ax.set_yscale('log')
        ax_idx += 1
        plot_one_map(Brms, r, map_ax, hist_ax)

    #  x = np.linspace(-PI/2, PI/2, 200)
    #  y = np.sqrt((PI/2)**2-x**2)
    #  map_ax.plot(x, y, c='tab:red', ls='--', lw=2)
    #  map_ax.plot(x, -y, c='tab:red', ls='--', lw=2)


def save_maps(Brms, radii):
    plt.set_cmap('plasma'); plt.close();

    for r in radii:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111, projection='hammer')
        ax.axis('off')
        hist = plt.figure()
        ha = hist.add_subplot(111)
        ha.set_yscale('log')
        plot_one_map(Brms, r, ax, ha, True)
        fig.tight_layout()
        fig.savefig('sky_map_brms-%.0e_r-%.0e.pdf' % (Brms, r))
        #  fig.savefig('sky_map_brms-%.0e_r-%.0e.png' % (Brms, r),
        #              dpi=300, transparent=True)
        plt.close()
        plt.close()


#  plot_all_maps(1e-13)
#  plot_all_maps(1e-12)
#  plot_all_maps(5e-12)

save_maps(1e-12, [7e21, 7e22, 7e23])


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
