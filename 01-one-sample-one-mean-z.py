import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import numpy as np
import pandas as pd
import math

ALPHA = 0.05
SAMPLE_SIZE = 100
NUMBER_OF_TESTS = 1000

POP_MEAN = 10
POP_STD = 3
POP_PROB_DENS_X = np.linspace(POP_MEAN - 4*POP_STD, POP_MEAN + 4*POP_STD, 1000)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
POP_PROB_DENS_Y = stats.norm.pdf(POP_PROB_DENS_X, loc = POP_MEAN, scale = POP_STD)
POP_PROB_MAX = stats.norm.pdf(POP_MEAN, loc = POP_MEAN, scale = POP_STD)

fig, ax = plt.subplots(1, 1)
ax.grid(axis='both', linestyle='--', color='0.95')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.set_xlim(POP_MEAN - 4.2*POP_STD, POP_MEAN + 4.2*POP_STD)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

text0 = ax.text(POP_MEAN - 4*POP_STD, POP_PROB_MAX * 0.5, f'', fontsize=12)
dots, = ax.plot([], [], 'bo', alpha=1.0)
vlines0 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines1 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines2 = ax.vlines([], [], [], color='r', alpha=1.0)
fill = ax.fill([], [], alpha=0.4, hatch="X", color='lightblue')

# Distribution
ax.plot(POP_PROB_DENS_X, POP_PROB_DENS_Y)
# Population Mean
ax.vlines([POP_MEAN], [0], [POP_PROB_MAX], color='red', linestyle='dashed', linewidth=3)

h0_counter = 0
h1_counter = 0
for x in range(NUMBER_OF_TESTS):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    sample = stats.norm.rvs(size=SAMPLE_SIZE, loc = POP_MEAN, scale = POP_STD)
    s_mean = np.mean(sample)
    s_std = np.std(sample)

    # https://en.wikipedia.org/wiki/97.5th_percentile_point
    # Significance Level (α): 0.05
    # Z-Value (two-tailed): +/- 1.96
    # z_005 = 1.96

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    z_alpha = stats.norm.ppf(1 - ALPHA/2) # alpha divide by 2 - two-tailed test

    # Using Population Std Deviation
    moe = z_alpha * POP_STD / math.sqrt(len(sample))

    if (s_mean - moe) <= POP_MEAN and POP_MEAN <= (s_mean + moe):
        h0_counter += 1
    else:
        h1_counter += 1

    text0.set_text(
        f'Significance Level (α): {ALPHA * 100:.2f} % \n'
        + f'Z({ALPHA:.2f}) Two-Tailed: {z_alpha:.6f} \n\n'
        + f'Population Mean (μ): {POP_MEAN:.4f} \n'
        + f'Population Standard Deviation (σ): {POP_STD:.4f}\n\n'
        + f'Sample Size (n): {SAMPLE_SIZE}\n'
        + f'Sample Mean (x̄): {s_mean:.4f} \n'
        + f'Sample Standard Deviation (s): {s_std:.4f}\n\n'
        + f'H0 (μ=μ0) is TRUE: {h0_counter} (Correct)\n'
        + f'H1 (μ≠μ0) is TRUE: {h1_counter} (False positive)\n\n'
        + f'Actuall Type I Error Percent: {100 * h1_counter / (h0_counter + h1_counter):.2f} %'
    )

    # Sample Mean
    vlines0.remove()
    vlines0 = ax.vlines([s_mean], [0], [POP_PROB_MAX], color='green', linestyle='dashed', linewidth=3)

    # Dashed Lines
    # vlines1.remove()
    # vlines1 = ax.vlines(sample, [0] * len(sample), stats.norm.pdf(sample, loc = p_mean, scale = p_std), color='black', linestyle='dashed', linewidth=0.5)
    # Dots
    dots.set_data(sample, stats.norm.pdf(sample, loc = POP_MEAN, scale = POP_STD))

    # Confidence Interval
    vlines2.remove()
    vlines2 = ax.vlines([
            s_mean - moe, 
            s_mean + moe
        ], [
            0, 
            0
        ], [
            POP_PROB_MAX, 
            POP_PROB_MAX
        ], color='blue', linestyle='dashed', linewidth=1)

    fill[0].remove()
    fill = ax.fill([
            s_mean - moe, 
            s_mean - moe,
            s_mean + moe,
            s_mean + moe
        ], [
            0, 
            POP_PROB_MAX,
            POP_PROB_MAX,
            0
        ], alpha=0.4, hatch="//", color='lightblue')

    (x < 50 or x == NUMBER_OF_TESTS - 1) and (plt.tight_layout() or plt.pause(0.5))

plt.tight_layout()
plt.show()