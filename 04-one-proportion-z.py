import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import numpy as np
import pandas as pd
import math

# np.random.seed(seed=3)

ALPHA = 0.05
P_SIZE = 10000
S_SIZE = 40 # 40 200 1000
P_PROPORTION = 0.2
NUMBER_OF_TESTS = 1000

POP_PROB_DENS_X = np.arange(0, P_SIZE + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
POP_PROB_DENS_Y = stats.binom.pmf(POP_PROB_DENS_X, n = P_SIZE, p = P_PROPORTION)
POP_PROB_DENS_X = np.array(POP_PROB_DENS_X) / P_SIZE
POP_PROB_MAX = stats.binom.pmf(P_SIZE * P_PROPORTION, n = P_SIZE, p = P_PROPORTION)

SAM_PROB_DENS_X = np.arange(0, S_SIZE + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
SAM_PROB_DENS_Y = stats.binom.pmf(SAM_PROB_DENS_X, n = S_SIZE, p = P_PROPORTION)
SAM_PROB_DENS_X = np.array(SAM_PROB_DENS_X) / S_SIZE
SAM_PROB_MAX = stats.binom.pmf(S_SIZE * P_PROPORTION, n = S_SIZE, p = P_PROPORTION)

# PROB_MAX = max(POP_PROB_MAX, SAM_PROB_MAX)
# PROB_MAX = POP_PROB_MAX
PROB_MAX = SAM_PROB_MAX

fig, ax = plt.subplots(1, 1)
ax.grid(axis='both', linestyle='--', color='0.95')
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.set_xlim(-0.1, 1.1)
ax.yaxis.set_major_locator(ticker.MultipleLocator(PROB_MAX * 0.1))
# ax.set_ylim(-0.2, 0.2)
# ax.set_yscale("log")

text0 = ax.text(1.75 * P_PROPORTION, PROB_MAX * 0.5, f'', fontsize=12)
vlines0 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines2 = ax.vlines([], [], [], color='r', alpha=1.0)
fill = ax.fill([], [], alpha=0.4, hatch="X", color='lightblue')

# Distribution
# ax.plot(POP_PROB_DENS_X, POP_PROB_DENS_Y, marker='o', linestyle='dashed', alpha=1.0, linewidth=2.0)
ax.plot(SAM_PROB_DENS_X, SAM_PROB_DENS_Y, marker='o', linestyle='dashed', alpha=1.0, linewidth=2.0)
# Population Mean
ax.vlines([P_PROPORTION], [0], [PROB_MAX], color='red', linestyle='dashed', linewidth=3)

h0_counter = 0
h1_counter = 0
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
for i, x in enumerate(stats.binom.rvs(size=NUMBER_OF_TESTS, n = S_SIZE, p = P_PROPORTION)):

    s_proportion = x / S_SIZE

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    z_alpha = stats.norm.ppf(1 - ALPHA/2) # alpha divide by 2 - two-tailed test

    moe = z_alpha * math.sqrt(s_proportion * (1 - s_proportion) / S_SIZE)

    if (s_proportion - moe) <= P_PROPORTION and P_PROPORTION <= (s_proportion + moe):
        h0_counter += 1
    else:
        h1_counter += 1

    text0.set_text(
        f'Significance Level (α): {ALPHA * 100:.2f} % \n' 
        + f'Z({ALPHA:.2f}) Two-Tailed: {z_alpha:.6f}\n\n'
        + f'Population Size : {P_SIZE} \n'
        + f'Population Proportion (p0): {P_PROPORTION:.4f} \n\n'
        + f'Sample Size : {S_SIZE} \n'
        + f'Sample Proportion (p̄): {s_proportion:.4f} \n\n'
        + f'Margin of Error (MOE): {moe:.4f} \n\n'
        + f'H0 (p̄=p0) is TRUE: {h0_counter} (Correct)\n'
        + f'H1 (p̄≠p0) is TRUE: {h1_counter} (False positive)\n\n'
        + f'Actuall Type I Error Percent: {100 * h1_counter / (h0_counter + h1_counter):.2f} %'
    )

    # Sample Mean
    vlines0.remove()
    vlines0 = ax.vlines([s_proportion], [0], [PROB_MAX], color='green', linestyle='dashed', linewidth=3)

    # Confidence Interval
    vlines2.remove()
    vlines2 = ax.vlines([
            s_proportion - moe, 
            s_proportion + moe
        ], [
            0, 
            0
        ], [
            PROB_MAX, 
            PROB_MAX
        ], color='blue', linestyle='dashed', linewidth=1)

    fill[0].remove()
    fill = ax.fill([
            s_proportion - moe, 
            s_proportion - moe,
            s_proportion + moe,
            s_proportion + moe
        ], [
            0, 
            PROB_MAX,
            PROB_MAX,
            0
        ], alpha=0.4, hatch="//", color='lightblue')


    (i < 50 or i == NUMBER_OF_TESTS - 1) and (plt.tight_layout() or plt.pause(0.5))

plt.tight_layout()
plt.show()