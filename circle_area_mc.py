#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import sqrt, pi
import numpy as np
import matplotlib.pyplot as plt

# seed from device entropy
np.random.seed()

# sample size per MC run
N = 10_000
# number of MC runs
M = 1_000


# single MC
a = np.random.uniform(size=(2, N))
in_circle = a[0]**2 + a[1]**2 < 1
area_one = 4 * np.count_nonzero(in_circle) / N
print(f'Area of an unit circle from 1 MC run with {N:_d} samples: '
      f'{area_one:.3f}')

# plotting
phi = np.linspace(0, np.pi/2, 100)
x = np.cos(phi)
y = np.sin(phi)

plt.figure(figsize=(4, 4), dpi=300)
plt.scatter(*a[:,in_circle], c='r', s=4, marker='.')
plt.scatter(*a[:,~in_circle], c='k', s=4, marker='.')
plt.plot(x, y, lw='2')
plt.gca().set(xlim=(0, 1), ylim=(0, 1), xlabel='x', ylabel='y')
plt.tick_params(which='both', length=0)
plt.grid(False, which='both')


# averaging multiple MC runs
b = np.random.uniform(size=(2, N, M))
area_many = 4 * np.count_nonzero(b[0]**2 + b[1]**2 < 1, axis=0) / N
mean, std = area_many.mean(), area_many.std()
print(f'Mean and std of {M:_d} MC runs with {N:_d} samples each: '
      f'{mean:.3f} +/- {std:.3f}')

# normal limit
x = np.linspace(3.08, 3.20, 200)
y = np.exp(-(x-mean)**2 / (2*std**2)) / (std * sqrt(2*pi))

plt.figure()
plt.hist(area_many, density=True, label='data')
plt.plot(x, y, label='limit normal distribution')
plt.xlabel('area of unit circle')
plt.ylabel('normalized number of occurrences')
plt.legend()
plt.show()


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
