# Protocol

## Setup
### Circle Area MC
The task is to compute the area of the unit circle stochastically. This is
achieved by randomly sampling 10,000 coordinates `(x, y)` from the
continuous uniform distribution over the interval `[0, 1]` `\mathcal{U}(0,
1)` and counting those, which have a euclidean distance `< 1` from `(0, 0)`.

> TODO: add scatter plot

Because the points are distributed uniformly, the ratio of points in the
square (i.e. all of them) and points in the quadrant of the circle equals
the ratio of the squares and the quadrants surface area:
```
N_quadrant / N_total = A_quadrant / A_square
```
With `A_square` known to be one, the circles surface are can be calculated:
```
A_circle = 4 * N_quad / N_tot
```

In the next step, the simulation is performed 1,000 times in order to
investigate the distribution and accuracy of the obtained results.

## Evaluation
### Circle Area MC
The exact value of the unit circles surface area is
```
A_circle,exact = \eval{\pi r^2}_{r=1}=\pi \approx 3.141\dotso
```

Performing and averaging the Monte-Carlo simulations can be easily done with
`numpy`:
```
r = np.random.uniform(size=(2, NUMBER_OF_SAMPLES, NUMBER_OF_MC_RUNS))
area = 4 * np.count_nonzero(r[0]**2 + r[1]**2 < 1, axis=0) / NUMBER_OF_SAMPLES
mean, std = area.mean(), area.std()
```
which typically gives results like
```
A_circle,MC = 3.142 \pm 0.016
```

When plotting a histogram, one finds that the results distribute around some
mean value close to \pi, resembling a normal distribution (-> central limit
theorem).

MC makes use of the Law of large numbers.


### Reweighting

### Interactions

### Deflections

### Constraints


>  vim: set spell ff=unix tw=79 sw=4 ts=4 et ic ai :
