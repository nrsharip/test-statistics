import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import numpy as np
import pandas as pd
import math

# np.random.seed(seed=3)

ALPHA = 0.05

P_1_SIZE = 10000
P_2_SIZE = 10000

S_1_SIZE = 40 # 40 200 1000
S_2_SIZE = 200 # 40 200 1000

P_1_PROPORTION = 0.7 # 0.7 0.5
P_2_PROPORTION = 0.5

NUMBER_OF_TESTS = 1000

POP_PROB_DENS_X = np.arange(0, P_1_SIZE + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
POP_PROB_DENS_Y = stats.binom.pmf(POP_PROB_DENS_X, n = P_1_SIZE, p = P_1_PROPORTION)
POP_PROB_DENS_X = np.array(POP_PROB_DENS_X) / P_1_SIZE
POP_PROB_MAX = stats.binom.pmf(P_1_SIZE * P_1_PROPORTION, n = P_1_SIZE, p = P_1_PROPORTION)

SAM_1_PROB_DENS_X = np.arange(0, S_1_SIZE + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
SAM_1_PROB_DENS_Y = stats.binom.pmf(SAM_1_PROB_DENS_X, n = S_1_SIZE, p = P_1_PROPORTION)
SAM_1_PROB_DENS_X = np.array(SAM_1_PROB_DENS_X) / S_1_SIZE
SAM_1_PROB_MAX = stats.binom.pmf(S_1_SIZE * P_1_PROPORTION, n = S_1_SIZE, p = P_1_PROPORTION)

SAM_2_PROB_DENS_X = np.arange(0, S_2_SIZE + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
SAM_2_PROB_DENS_Y = stats.binom.pmf(SAM_2_PROB_DENS_X, n = S_2_SIZE, p = P_2_PROPORTION)
SAM_2_PROB_DENS_X = np.array(SAM_2_PROB_DENS_X) / S_2_SIZE
SAM_2_PROB_MAX = stats.binom.pmf(S_2_SIZE * P_2_PROPORTION, n = S_2_SIZE, p = P_2_PROPORTION)

# PROB_MAX = max(POP_PROB_MAX, SAM_PROB_MAX)
# PROB_MAX = POP_PROB_MAX
PROB_MAX = max(SAM_1_PROB_MAX, SAM_2_PROB_MAX)
PROB_MIN = min(SAM_1_PROB_MAX, SAM_2_PROB_MAX)

fig, ax = plt.subplots(1, 1)
ax.grid(axis='both', linestyle='--', color='0.95')
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.set_xlim(-0.1, 1.1)
ax.yaxis.set_major_locator(ticker.MultipleLocator(PROB_MAX * 0.1))
# ax.set_ylim(-0.2, 0.2)
# ax.set_yscale("log")

text0 = ax.text(0.05, PROB_MAX * 0.5, f'', fontsize=12)
vlines0 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines1 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines2 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines3 = ax.vlines([], [], [], color='r', alpha=1.0)
fill = ax.fill([], [], alpha=0.4, hatch="X", color='lightblue')

# Distribution
# ax.plot(POP_PROB_DENS_X, POP_PROB_DENS_Y, marker='o', linestyle='dashed', alpha=1.0, linewidth=2.0)
ax.plot(SAM_1_PROB_DENS_X, SAM_1_PROB_DENS_Y, color='b', marker='o', linestyle='dashed', alpha=1.0, linewidth=2.0)
ax.plot(SAM_2_PROB_DENS_X, SAM_2_PROB_DENS_Y, color='g', marker='o', linestyle='dashed', alpha=1.0, linewidth=2.0)
# Population Mean
ax.vlines([P_1_PROPORTION], [0], [SAM_1_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
ax.vlines([P_2_PROPORTION], [0], [SAM_2_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
ax.vlines([P_1_PROPORTION - P_2_PROPORTION], [0], [PROB_MIN], color='red', linestyle='dashed', linewidth=3)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
binom_randoms = np.array([
    stats.binom.rvs(size=NUMBER_OF_TESTS, n = S_1_SIZE, p = P_1_PROPORTION),
    stats.binom.rvs(size=NUMBER_OF_TESTS, n = S_2_SIZE, p = P_2_PROPORTION)
]).T

h0_counter = 0
h1_counter = 0
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
for i, x in enumerate(binom_randoms):

    s1_proportion = x[0] / S_1_SIZE
    s2_proportion = x[1] / S_2_SIZE

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    z_alpha = stats.norm.ppf(1 - ALPHA/2) # alpha divide by 2 - two-tailed test

    moe = z_alpha
    moe *= math.sqrt(s1_proportion*(1 - s1_proportion)/S_1_SIZE + s2_proportion*(1 - s2_proportion)/S_2_SIZE)

    if ((s1_proportion - s2_proportion) - moe) <= P_1_PROPORTION - P_2_PROPORTION \
        and P_1_PROPORTION - P_2_PROPORTION <= ((s1_proportion - s2_proportion) + moe):
        h0_counter += 1
    else:
        h1_counter += 1

    if i < 50 or i == NUMBER_OF_TESTS - 1:
        text0.set_text(
            f'Significance Level (α): {ALPHA * 100:.2f} % \n' 
            + f'Z({ALPHA:.2f}) Two-Tailed: {z_alpha:.6f}\n\n'

            + f'Population Proportion (p1): {P_1_PROPORTION:.4f} \n'
            + f'Population Proportion (p2): {P_2_PROPORTION:.4f} \n\n'

            + f'Sample Size 1: {S_1_SIZE} \n'
            + f'Sample Size 2: {S_2_SIZE} \n'
            + f'Sample Proportion 1 (p̄1): {s1_proportion:.4f} \n'
            + f'Sample Proportion 2 (p̄2): {s2_proportion:.4f} \n\n'

            + f'Margin of Error (MOE): {moe:.4f} \n\n'

            + f'H0 (p1-p2={P_1_PROPORTION - P_2_PROPORTION:.2f}) is TRUE: {h0_counter} (Correct)\n'
            + f'H1 (p1-p2≠{P_1_PROPORTION - P_2_PROPORTION:.2f}) is TRUE: {h1_counter} (False positive)\n\n'

            + f'Actuall Type I Error Percent: {100 * h1_counter / (h0_counter + h1_counter):.2f} %'
        )

        # Sample Mean
        vlines0.remove()
        vlines0 = ax.vlines([s1_proportion], [0], [SAM_1_PROB_MAX], color='b', linestyle='dashed', linewidth=3)
        vlines1.remove()
        vlines1 = ax.vlines([s2_proportion], [0], [SAM_2_PROB_MAX], color='g', linestyle='dashed', linewidth=3)
        vlines2.remove()
        vlines2 = ax.vlines([s1_proportion-s2_proportion], [0], [PROB_MIN], color='k', linestyle='dashed', linewidth=3)

        # Confidence Interval
        vlines3.remove()
        vlines3 = ax.vlines([
            (s1_proportion - s2_proportion) - moe, 
            (s1_proportion - s2_proportion) + moe
        ], [
            0, 
            0
        ], [
            PROB_MIN, 
            PROB_MIN
        ], color='blue', linestyle='dashed', linewidth=1)

        fill[0].remove()
        fill = ax.fill([
            (s1_proportion - s2_proportion) - moe, 
            (s1_proportion - s2_proportion) - moe,
            (s1_proportion - s2_proportion) + moe,
            (s1_proportion - s2_proportion) + moe
        ], [
            0, 
            PROB_MIN,
            PROB_MIN,
            0
        ], alpha=0.4, hatch="//", color='lightblue')

        plt.tight_layout() 
        plt.pause(0.5)

plt.tight_layout()
plt.show()