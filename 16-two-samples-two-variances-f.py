import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import numpy as np
import pandas as pd
import math

ALPHA = 0.05

NUMBER_OF_TESTS = 1000

POP_1_MEAN = 10
POP_1_STD = 2
SAMPLE_1_SIZE = 100

POP_2_MEAN = 15
POP_2_STD = 2
SAMPLE_2_SIZE = 100

POP_1_PROB_DENS_X = np.linspace(POP_1_MEAN - 4*POP_1_STD, POP_1_MEAN + 4*POP_1_STD, 1000)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
POP_1_PROB_DENS_Y = stats.norm.pdf(POP_1_PROB_DENS_X, loc = POP_1_MEAN, scale = POP_1_STD)
POP_1_PROB_MAX = stats.norm.pdf(POP_1_MEAN, loc = POP_1_MEAN, scale = POP_1_STD)

POP_2_PROB_DENS_X = np.linspace(POP_2_MEAN - 4*POP_2_STD, POP_2_MEAN + 4*POP_2_STD, 1000)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
POP_2_PROB_DENS_Y = stats.norm.pdf(POP_2_PROB_DENS_X, loc = POP_2_MEAN, scale = POP_2_STD)
POP_2_PROB_MAX = stats.norm.pdf(POP_2_MEAN, loc = POP_2_MEAN, scale = POP_2_STD)

fig, ax = plt.subplots(1, 1)
ax.grid(axis='both', linestyle='--', color='0.95')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.set_xlim(POP_1_MEAN - 4.2*POP_1_STD, POP_2_MEAN + 4.2*POP_2_STD)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

text0 = ax.text(POP_1_MEAN - 4*POP_1_STD, POP_1_PROB_MAX * 0.1, f'', fontsize=12)
dots1, = ax.plot([], [], 'bo', alpha=1.0)
dots2, = ax.plot([], [], 'go', alpha=1.0)
vlines0 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines1 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines2 = ax.vlines([], [], [], color='r', alpha=1.0)
fill = ax.fill([], [], 'lightblue', [], [], 'lightblue', alpha=0.4, hatch="X")

# Distribution
ax.plot(POP_1_PROB_DENS_X, POP_1_PROB_DENS_Y, color='b')
ax.plot(POP_2_PROB_DENS_X, POP_2_PROB_DENS_Y, color='g')
# Population Mean
ax.vlines([POP_1_MEAN], [0], [POP_1_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
ax.vlines([POP_2_MEAN], [0], [POP_2_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
# Single Standard Deviation spread
ax.vlines([
    POP_1_MEAN - POP_1_STD,
    POP_1_MEAN + POP_1_STD, 
    POP_2_MEAN - POP_2_STD,
    POP_2_MEAN + POP_2_STD, 
], [
    0, 
    0,
    0, 
    0,
], [
    POP_1_PROB_MAX, 
    POP_1_PROB_MAX,
    POP_2_PROB_MAX, 
    POP_2_PROB_MAX,
], color='blue', linestyle='dashed', linewidth=2)
# Single Standard Deviation area
ax.fill([
    POP_1_MEAN - POP_1_STD, 
    POP_1_MEAN - POP_1_STD,
    POP_1_MEAN + POP_1_STD,
    POP_1_MEAN + POP_1_STD
], [
    0, 
    POP_1_PROB_MAX,
    POP_1_PROB_MAX,
    0
], 'lightblue', [
    POP_2_MEAN - POP_2_STD, 
    POP_2_MEAN - POP_2_STD,
    POP_2_MEAN + POP_2_STD,
    POP_2_MEAN + POP_2_STD
], [
    0, 
    POP_2_PROB_MAX,
    POP_2_PROB_MAX,
    0
], 'lightblue', alpha=0.4, hatch="\\\\", edgecolor='lightblue')

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html
# alpha divide by 2 - two-tailed test LEFT
f_alpha_left = stats.f.ppf(ALPHA/2, SAMPLE_1_SIZE - 1, SAMPLE_2_SIZE - 1)
# alpha divide by 2 - two-tailed test RIGHT
f_alpha_rght = stats.f.ppf(1 - ALPHA/2, SAMPLE_1_SIZE - 1, SAMPLE_2_SIZE - 1)

h0_counter = 0
h1_counter = 0
for x in range(NUMBER_OF_TESTS):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    sample1 = stats.norm.rvs(size=SAMPLE_1_SIZE, loc = POP_1_MEAN, scale = POP_1_STD)
    s1_mean = np.mean(sample1)
    s1_std = np.std(sample1)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    sample2 = stats.norm.rvs(size=SAMPLE_2_SIZE, loc = POP_2_MEAN, scale = POP_2_STD)
    s2_mean = np.mean(sample2)
    s2_std = np.std(sample2)

    ratio = s1_std**2/s2_std**2 if s2_std != 0 else 1

    if f_alpha_left <= ratio and ratio <= f_alpha_rght:
        h0_counter += 1
    else:
        h1_counter += 1

    if x < 50 or x == NUMBER_OF_TESTS - 1:
        text0.set_text(
            f'Significance Level (α): {ALPHA * 100:.2f} % \n'
            + f'F(α={ALPHA/2:.4f}, d1={SAMPLE_1_SIZE - 1}, d2={SAMPLE_2_SIZE - 1}) Two-Tailed Left: {f_alpha_left:.6f} \n'
            + f'F(α={1-ALPHA/2:.4f}, d1={SAMPLE_1_SIZE - 1}, d2={SAMPLE_2_SIZE - 1}) Two-Tailed Right: {f_alpha_rght:.6f} \n\n'
            + f'Population Mean (μ1): {POP_1_MEAN:.4f} \n' 
            + f'Population Mean (μ2): {POP_2_MEAN:.4f} \n' 
            + f'Population Standard Deviation (σ1): {POP_1_STD:.4f}\n'
            + f'Population Standard Deviation (σ2): {POP_2_STD:.4f}\n\n'
            + f'Sample Size (n1): {SAMPLE_1_SIZE}\n'
            + f'Sample Size (n2): {SAMPLE_2_SIZE}\n'
            + f'Sample Mean (x̄1): {s1_mean:.4f} \n'
            + f'Sample Mean (x̄2): {s2_mean:.4f} \n'
            + f'Sample Standard Deviation (s1): {s1_std:.4f}\n'
            + f'Sample Standard Deviation (s2): {s2_std:.4f}\n\n'
            + f'Ratio: {ratio:.6f}\n\n'
            + f'H0 (σ1^2~σ2^2) is TRUE: {h0_counter} (Correct)\n'
            + f'H1 (σ1^2≠σ2^2) is TRUE: {h1_counter} (False positive)\n\n'
            + f'Actuall Type I Error Percent: {100 * h1_counter / (h0_counter + h1_counter):.2f} %'
        )

        # # Sample Mean
        # vlines0.remove()
        # vlines0 = ax.vlines([s_mean], [0], [POP_PROB_MAX], color='green', linestyle='dashed', linewidth=3)

        # Dots
        dots1.set_data(sample1, stats.norm.pdf(sample1, loc = POP_1_MEAN, scale = POP_1_STD))
        dots2.set_data(sample2, stats.norm.pdf(sample2, loc = POP_2_MEAN, scale = POP_2_STD))

        # Confidence Interval
        vlines2.remove()
        vlines2 = ax.vlines([
                s1_mean - s1_std, 
                s1_mean + s1_std, 
                s2_mean - s2_std, 
                s2_mean + s2_std, 
            ], [
                0, 
                0,
                0,
                0
            ], [
                POP_1_PROB_MAX, 
                POP_1_PROB_MAX,
                POP_2_PROB_MAX, 
                POP_2_PROB_MAX
            ], color='blue', linestyle='dashed', linewidth=1)

        fill[0].remove()
        fill[1].remove()
        fill = ax.fill([
                s1_mean - s1_std,
                s1_mean - s1_std,
                s1_mean + s1_std,
                s1_mean + s1_std
            ], [
                0, 
                POP_1_PROB_MAX,
                POP_1_PROB_MAX,
                0
            ], 'lightgreen', [
                s2_mean - s2_std,
                s2_mean - s2_std,
                s2_mean + s2_std,
                s2_mean + s2_std
            ], [
                0, 
                POP_2_PROB_MAX,
                POP_2_PROB_MAX,
                0
            ], 'lightgreen', alpha=0.4, hatch="//", edgecolor='lightgreen'
        )

        plt.tight_layout()
        plt.pause(0.5)

plt.tight_layout()
plt.show()