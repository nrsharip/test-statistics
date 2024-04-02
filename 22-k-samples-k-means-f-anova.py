import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import numpy as np
import pandas as pd
import math

ALPHA = 0.05

NUMBER_OF_TESTS = 1000

POP_1_MEAN = 10 # 10 25
POP_1_STD = 2
SAMPLE_1_SIZE = 25

POP_2_MEAN = 10 # 10 20
POP_2_STD = 3
SAMPLE_2_SIZE = 50

POP_3_MEAN = 10 # 10 15
POP_3_STD = 4
SAMPLE_3_SIZE = 75

POP_4_MEAN = 10
POP_4_STD = 5
SAMPLE_4_SIZE = 100

N = SAMPLE_1_SIZE + SAMPLE_2_SIZE + SAMPLE_3_SIZE + SAMPLE_4_SIZE
K = 4

POP_1_PROB_DENS_X = np.linspace(POP_1_MEAN - 4*POP_1_STD, POP_1_MEAN + 4*POP_1_STD, 1000)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
POP_1_PROB_DENS_Y = stats.norm.pdf(POP_1_PROB_DENS_X, loc = POP_1_MEAN, scale = POP_1_STD)
POP_1_PROB_MAX = stats.norm.pdf(POP_1_MEAN, loc = POP_1_MEAN, scale = POP_1_STD)

POP_2_PROB_DENS_X = np.linspace(POP_2_MEAN - 4*POP_2_STD, POP_2_MEAN + 4*POP_2_STD, 1000)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
POP_2_PROB_DENS_Y = stats.norm.pdf(POP_2_PROB_DENS_X, loc = POP_2_MEAN, scale = POP_2_STD)
POP_2_PROB_MAX = stats.norm.pdf(POP_2_MEAN, loc = POP_2_MEAN, scale = POP_2_STD)

POP_3_PROB_DENS_X = np.linspace(POP_3_MEAN - 4*POP_3_STD, POP_3_MEAN + 4*POP_3_STD, 1000)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
POP_3_PROB_DENS_Y = stats.norm.pdf(POP_3_PROB_DENS_X, loc = POP_3_MEAN, scale = POP_3_STD)
POP_3_PROB_MAX = stats.norm.pdf(POP_3_MEAN, loc = POP_3_MEAN, scale = POP_3_STD)

POP_4_PROB_DENS_X = np.linspace(POP_4_MEAN - 4*POP_4_STD, POP_4_MEAN + 4*POP_4_STD, 1000)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
POP_4_PROB_DENS_Y = stats.norm.pdf(POP_4_PROB_DENS_X, loc = POP_4_MEAN, scale = POP_4_STD)
POP_4_PROB_MAX = stats.norm.pdf(POP_4_MEAN, loc = POP_4_MEAN, scale = POP_4_STD)

fig, ax = plt.subplots(1, 1)
ax.grid(axis='both', linestyle='--', color='0.95')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.set_xlim(POP_4_MEAN - 4.2*POP_4_STD, max(POP_1_MEAN + 4.2*POP_1_STD, POP_4_MEAN + 4.2*POP_4_STD))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

text0 = ax.text(POP_4_MEAN - 4.0*POP_4_STD, 0.5*POP_1_PROB_MAX, f'', fontsize=12)
dots1, = ax.plot([], [], 'bo', alpha=1.0)
dots2, = ax.plot([], [], 'go', alpha=1.0)
dots2, = ax.plot([], [], 'mo', alpha=1.0)
dots4, = ax.plot([], [], 'yo', alpha=1.0)

vlines1 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines2 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines3 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines4 = ax.vlines([], [], [], color='r', alpha=1.0)
vlines5 = ax.vlines([], [], [], color='r', alpha=1.0)
fill1 = ax.fill(
    [], [], 'b', 
    [], [], 'g', 
    [], [], 'm', 
    [], [], 'y', 
    alpha=0.4, hatch="X"
)
fill2 = ax.fill(
    [], [], 'b', 
    [], [], 'g', 
    [], [], 'm', 
    [], [], 'y', 
    alpha=0.4, hatch="X"
)
fill3 = ax.fill_between(
    [], []
)

# Distributions
ax.plot(POP_1_PROB_DENS_X, POP_1_PROB_DENS_Y, color='b')
ax.plot(POP_2_PROB_DENS_X, POP_2_PROB_DENS_Y, color='g')
ax.plot(POP_3_PROB_DENS_X, POP_3_PROB_DENS_Y, color='m')
ax.plot(POP_4_PROB_DENS_X, POP_4_PROB_DENS_Y, color='y')
# Population Mean
ax.vlines([POP_1_MEAN], [0], [POP_1_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
ax.vlines([POP_2_MEAN], [0], [POP_2_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
ax.vlines([POP_3_MEAN], [0], [POP_3_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
ax.vlines([POP_4_MEAN], [0], [POP_4_PROB_MAX], color='red', linestyle='dashed', linewidth=3)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html
f_alpha_left = stats.f.ppf(ALPHA/2, K - 1, N - K) # alpha divide by 2 - two-tailed test LEFT
f_alpha_rght = stats.f.ppf(1 - ALPHA/2, K - 1, N - K) # alpha divide by 2 - two-tailed test RIGHT

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

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    sample3 = stats.norm.rvs(size=SAMPLE_3_SIZE, loc = POP_3_MEAN, scale = POP_3_STD)
    s3_mean = np.mean(sample3)
    s3_std = np.std(sample3)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    sample4 = stats.norm.rvs(size=SAMPLE_4_SIZE, loc = POP_4_MEAN, scale = POP_4_STD)
    s4_mean = np.mean(sample4)
    s4_std = np.std(sample4)

    T = np.concatenate((sample1, sample2, sample3, sample4), axis=None)

    S_mean = T.sum()/N
    
    S_I_squared = np.power(np.array(sample1) - s1_mean, 2).sum()
    S_I_squared += np.power(np.array(sample2) - s2_mean, 2).sum()
    S_I_squared += np.power(np.array(sample3) - s3_mean, 2).sum()
    S_I_squared += np.power(np.array(sample4) - s4_mean, 2).sum()
    S_I_squared /= N - K

    S_I = math.sqrt(S_I_squared)

    S_II_squared = SAMPLE_1_SIZE * (s1_mean - S_mean)**2
    S_II_squared += SAMPLE_2_SIZE * (s2_mean - S_mean)**2
    S_II_squared += SAMPLE_3_SIZE * (s3_mean - S_mean)**2
    S_II_squared += SAMPLE_4_SIZE * (s4_mean - S_mean)**2
    S_II_squared /= K - 1

    S_II = math.sqrt(S_II_squared)

    ratio = S_II_squared / S_I_squared

    if f_alpha_left <= ratio and ratio <= f_alpha_rght:
        h0_counter += 1
    else:
        h1_counter += 1

    text0.set_text(
        f'Significance Level (α): {ALPHA * 100:.2f} % \n'
        + f'F({ALPHA/2:.4f}, d1={K - 1}, d2={N - K}) Two-Tailed Left: {f_alpha_left:.6f} \n'
        + f'F({1-ALPHA/2:.4f}, d1={K - 1}, d2={N - K}) Two-Tailed Right: {f_alpha_rght:.6f} \n\n'
        + f'S I - squared: {S_I_squared:.4f}\n'
        + f'S II - squared: {S_II_squared:.4f}\n\n'
        + f'Ratio: {ratio:.6f}\n\n'
        + f'H0 (μ1=μ2=μ3=μ4) is TRUE: {h0_counter} (Correct)\n'
        + f'H1 (∃i,j,i≠j: μi≠μj) is TRUE: {h1_counter} (False positive)\n\n'
        + f'Actuall Type I Error Percent: {100 * h1_counter / (h0_counter + h1_counter):.2f} %'
    )

    # Sample Means
    vlines1.remove()
    vlines1 = ax.vlines([s1_mean], [0], [POP_1_PROB_MAX], color='b', linestyle='dashed', linewidth=2)
    vlines2.remove()
    vlines2 = ax.vlines([s2_mean], [0], [POP_2_PROB_MAX], color='g', linestyle='dashed', linewidth=2)
    vlines3.remove()
    vlines3 = ax.vlines([s3_mean], [0], [POP_3_PROB_MAX], color='m', linestyle='dashed', linewidth=2)
    vlines4.remove()
    vlines4 = ax.vlines([s4_mean], [0], [POP_4_PROB_MAX], color='y', linestyle='dashed', linewidth=2)
    vlines5.remove()
    vlines5 = ax.vlines([S_mean], [0], [POP_4_PROB_MAX], color='k', linestyle='dashed', linewidth=3)

    # Dots
    dots1.set_data(sample1, stats.norm.pdf(sample1, loc = POP_1_MEAN, scale = POP_1_STD))
    dots2.set_data(sample2, stats.norm.pdf(sample2, loc = POP_2_MEAN, scale = POP_2_STD))
    dots2.set_data(sample3, stats.norm.pdf(sample3, loc = POP_3_MEAN, scale = POP_3_STD))
    dots4.set_data(sample4, stats.norm.pdf(sample4, loc = POP_4_MEAN, scale = POP_4_STD))

    # Confidence Interval
    fill1[0].remove()
    fill1[1].remove()
    fill1[2].remove()
    fill1[3].remove()
    fill1 = ax.fill([
            s1_mean - s1_std, s1_mean - s1_std, s1_mean + s1_std, s1_mean + s1_std, 
        ], [
            0, POP_1_PROB_MAX, POP_1_PROB_MAX, 0
        ], 'b', [
            s2_mean - s2_std, s2_mean - s2_std, s2_mean + s2_std, s2_mean + s2_std, 
        ], [
            0, POP_2_PROB_MAX, POP_2_PROB_MAX, 0
        ], 'g', [
            s3_mean - s3_std, s3_mean - s3_std, s3_mean + s3_std, s3_mean + s3_std, 
        ], [
            0, POP_3_PROB_MAX, POP_3_PROB_MAX, 0
        ], 'm', [
            s4_mean - s4_std, s4_mean - s4_std, s4_mean + s4_std, s4_mean + s4_std, 
        ], [
            0, POP_4_PROB_MAX, POP_4_PROB_MAX, 0
        ], 'y', alpha=0.1, hatch="//", edgecolor='grey')

    fill2[0].remove()
    fill2[1].remove()
    fill2[2].remove()
    fill2[3].remove()
    fill2 = ax.fill([
            s1_mean - S_I, s1_mean - S_I, s1_mean + S_I, s1_mean + S_I, 
        ], [
            0, POP_1_PROB_MAX, POP_1_PROB_MAX, 0
        ], 'b', [
            s2_mean - S_I, s2_mean - S_I, s2_mean + S_I, s2_mean + S_I, 
        ], [
            0, POP_2_PROB_MAX, POP_2_PROB_MAX, 0
        ], 'g', [
            s3_mean - S_I, s3_mean - S_I, s3_mean + S_I, s3_mean + S_I, 
        ], [
            0, POP_3_PROB_MAX, POP_3_PROB_MAX, 0
        ], 'm', [
            s4_mean - S_I, s4_mean - S_I, s4_mean + S_I, s4_mean + S_I, 
        ], [
            0, POP_4_PROB_MAX, POP_4_PROB_MAX, 0
        ], 'y', alpha=0.1, hatch="\\\\", edgecolor='grey')
    
    fill3.remove()
    fill3 = ax.fill_between([
            S_mean - S_II, S_mean - S_II, S_mean + S_II, S_mean + S_II
        ], [
            0, POP_4_PROB_MAX, POP_4_PROB_MAX, 0
        ], facecolor="none", alpha=1.0, hatch="-", edgecolor='red')

    (x < 50 or x == NUMBER_OF_TESTS - 1) and (plt.tight_layout() or plt.pause(0.5))

plt.tight_layout()
plt.show()