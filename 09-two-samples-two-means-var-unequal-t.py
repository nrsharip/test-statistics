import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import numpy as np
import pandas as pd
import math

ALPHA = 0.05

NUMBER_OF_TESTS = 1000

POP_1_MEAN = 20
POP_1_STD = 2
SAMPLE_1_SIZE = 100

POP_2_MEAN = 15
POP_2_STD = 4
SAMPLE_2_SIZE = 200

POP_1_PROB_DENS_X = np.linspace(POP_1_MEAN - 4*POP_1_STD, POP_1_MEAN + 4*POP_1_STD, 1000)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
POP_1_PROB_DENS_Y = stats.t.pdf(POP_1_PROB_DENS_X, df=SAMPLE_1_SIZE - 1, loc = POP_1_MEAN, scale = POP_1_STD)
POP_1_PROB_MAX = stats.t.pdf(POP_1_MEAN, df=SAMPLE_1_SIZE - 1, loc = POP_1_MEAN, scale = POP_1_STD)

POP_2_PROB_DENS_X = np.linspace(POP_2_MEAN - 4*POP_2_STD, POP_2_MEAN + 4*POP_2_STD, 1000)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
POP_2_PROB_DENS_Y = stats.t.pdf(POP_2_PROB_DENS_X, df=SAMPLE_2_SIZE - 1, loc = POP_2_MEAN, scale = POP_2_STD)
POP_2_PROB_MAX = stats.t.pdf(POP_2_MEAN, df=SAMPLE_2_SIZE - 1, loc = POP_2_MEAN, scale = POP_2_STD)

fig, ax = plt.subplots(1, 1)
ax.grid(axis='both', linestyle='--', color='0.95')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.set_xlim(POP_2_MEAN - 4.0*POP_2_STD, POP_1_MEAN + 6.0*POP_1_STD)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

text0 = ax.text(POP_1_MEAN + 2*POP_1_STD, POP_1_PROB_MAX * 0.3, f'', fontsize=12)
dots0, = ax.plot([], [], 'bo', alpha=1.0)
dots1, = ax.plot([], [], 'go', alpha=1.0)
vlines0 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines1 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines2 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines3 = ax.vlines([], [], [], color='r', alpha=1.0)
fill = ax.fill([], [], alpha=0.4, hatch="X", color='lightblue')

# Distributions
ax.plot(POP_1_PROB_DENS_X, POP_1_PROB_DENS_Y, color='blue')
ax.plot(POP_2_PROB_DENS_X, POP_2_PROB_DENS_Y, color='green')
# Population Mean
ax.vlines([POP_1_MEAN], [0], [POP_1_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
ax.vlines([POP_2_MEAN], [0], [POP_2_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
ax.vlines([POP_1_MEAN - POP_2_MEAN], [0], [min(POP_1_PROB_MAX, POP_2_PROB_MAX)], color='red', linestyle='dashed', linewidth=3)

h0_counter = 0
h1_counter = 0
for x in range(NUMBER_OF_TESTS):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    sample1 = stats.t.rvs(df=SAMPLE_1_SIZE - 1, size=SAMPLE_1_SIZE, loc = POP_1_MEAN, scale = POP_1_STD)
    s1_mean = np.mean(sample1)
    s1_std = np.std(sample1)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    sample2 = stats.t.rvs(df=SAMPLE_2_SIZE - 1, size=SAMPLE_2_SIZE, loc = POP_2_MEAN, scale = POP_2_STD)
    s2_mean = np.mean(sample2)
    s2_std = np.std(sample2)

    # https://en.wikipedia.org/wiki/97.5th_percentile_point
    # Significance Level (α): 0.05
    # Z-Value (two-tailed): +/- 1.96
    # z_005 = 1.96

    df = (s1_std**2/SAMPLE_1_SIZE + s2_std**2/SAMPLE_2_SIZE)**2
    df /= (s1_std**2/SAMPLE_1_SIZE)**2/(SAMPLE_1_SIZE - 1) + (s2_std**2/SAMPLE_2_SIZE)**2/(SAMPLE_2_SIZE - 1)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    t_alpha = stats.t.ppf(1 - ALPHA/2, df=df) # alpha divide by 2 - two-tailed test
    
    moe = t_alpha * math.sqrt(s1_std**2/SAMPLE_1_SIZE + s2_std**2/SAMPLE_2_SIZE)
    
    if ((s1_mean - s2_mean) - moe) <= (POP_1_MEAN - POP_2_MEAN) \
        and (POP_1_MEAN - POP_2_MEAN) <= ((s1_mean - s2_mean) + moe):
        h0_counter += 1
    else:
        h1_counter += 1

    if x < 50 or x == NUMBER_OF_TESTS - 1:
        text0.set_text(
            f'Significance Level (α): {ALPHA * 100:.2f} % \n'
            + f'T({ALPHA:.2f},df={df:.2f}) Two-Tailed: {t_alpha:.6f}\n\n'
            + f'Population Mean (μ1): {POP_1_MEAN:.4f} \n'
            + f'Population Mean (μ2): {POP_2_MEAN:.4f} \n'
            + f'Population Standard Deviation (σ1): {POP_1_STD:.4f}\n\n'
            + f'Population Standard Deviation (σ2): {POP_2_STD:.4f}\n\n'
            + f'Sample Size 1 (n1): {SAMPLE_1_SIZE}\n'
            + f'Sample Size 2 (n2): {SAMPLE_2_SIZE}\n'
            + f'Sample Mean 1 (x̄1): {s1_mean:.4f} \n'
            + f'Sample Mean 2 (x̄2): {s2_mean:.4f} \n'
            + f'Sample Standard Deviation 1 (s1): {s1_std:.4f}\n'
            + f'Sample Standard Deviation 2 (s2): {s2_std:.4f}\n\n'
            + f'H0 (μ1-μ2={POP_1_MEAN - POP_2_MEAN:.2f}) is TRUE: {h0_counter} (Correct)\n'
            + f'H1 (μ1-μ2≠{POP_1_MEAN - POP_2_MEAN:.2f}) is TRUE: {h1_counter} (False positive)\n\n'
            + f'Actuall Type I Error Percent: {100 * h1_counter / (h0_counter + h1_counter):.2f} %'
        )

        # Sample Means
        vlines0.remove()
        vlines0 = ax.vlines([s1_mean], [0], [POP_1_PROB_MAX], color='blue', linestyle='dashed', linewidth=3)
        vlines1.remove()
        vlines1 = ax.vlines([s2_mean], [0], [POP_2_PROB_MAX], color='green', linestyle='dashed', linewidth=3)
        vlines2.remove()
        vlines2 = ax.vlines([s1_mean - s2_mean], [0], [min(POP_1_PROB_MAX, POP_2_PROB_MAX)], color='black', linestyle='dashed', linewidth=3)

        # Dots
        dots0.set_data(sample1, stats.t.pdf(sample1, df=SAMPLE_1_SIZE - 1, loc = POP_1_MEAN, scale = POP_1_STD))
        dots1.set_data(sample2, stats.t.pdf(sample2, df=SAMPLE_2_SIZE - 1, loc = POP_2_MEAN, scale = POP_2_STD))

        # Confidence Interval
        vlines3.remove()
        vlines3 = ax.vlines([
                (s1_mean - s2_mean) - moe, 
                (s1_mean - s2_mean) + moe
            ], [
                0, 
                0
            ], [
                min(POP_1_PROB_MAX, POP_2_PROB_MAX), 
                min(POP_1_PROB_MAX, POP_2_PROB_MAX)
            ], color='blue', linestyle='dashed', linewidth=1)

        fill[0].remove()
        fill = ax.fill([
                (s1_mean - s2_mean) - moe, 
                (s1_mean - s2_mean) - moe,
                (s1_mean - s2_mean) + moe,
                (s1_mean - s2_mean) + moe
            ], [
                0, 
                min(POP_1_PROB_MAX, POP_2_PROB_MAX),
                min(POP_1_PROB_MAX, POP_2_PROB_MAX),
                0
            ], alpha=0.4, hatch="//", color='lightblue')

        plt.tight_layout()
        plt.pause(0.5)

plt.tight_layout()
plt.show()