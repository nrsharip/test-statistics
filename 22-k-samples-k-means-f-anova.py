import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import numpy as np
import pandas as pd
import math

import statsmodels.api as sm
import statsmodels.formula.api as smf

ALPHA = 0.05

NUMBER_OF_TESTS = 1000

POP_1_MEAN = 10 # 10 25
POP_1_STD = 2
SAMPLE_1_SIZE = 5

POP_2_MEAN = 10 # 10 20
POP_2_STD = 3
SAMPLE_2_SIZE = 10

POP_3_MEAN = 10 # 10 15
POP_3_STD = 4
SAMPLE_3_SIZE = 15

POP_4_MEAN = 10
POP_4_STD = 6
SAMPLE_4_SIZE = 20

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

fig, ax = plt.subplots(2, 1)
ax[0].grid(axis='both', linestyle='--', color='0.95')
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax[0].set_xlim(POP_4_MEAN - 4.2*POP_4_STD, max(POP_1_MEAN + 4.2*POP_1_STD, POP_4_MEAN + 4.2*POP_4_STD))
ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.01))

text0 = ax[0].text(POP_4_MEAN - 4.0*POP_4_STD, 0.02*POP_1_PROB_MAX, f'', fontsize=7)
text_mu_1 = ax[0].text(POP_1_MEAN, -0.004, f'μ1=μ2=μ3=μ4')
text_mu_2 = ax[0].text(POP_2_MEAN, -0.004, f'μ1=μ2=μ3=μ4')
text_mu_3 = ax[0].text(POP_3_MEAN, -0.004, f'μ1=μ2=μ3=μ4')
text_mu_4 = ax[0].text(POP_4_MEAN, -0.004, f'μ1=μ2=μ3=μ4')

text_x_dash_1 = ax[0].text(0, 0, f'x̄1')
text_x_dash_2 = ax[0].text(0, 0, f'x̄2')
text_x_dash_3 = ax[0].text(0, 0, f'x̄3')
text_x_dash_4 = ax[0].text(0, 0, f'x̄4')

text_s_1_left = ax[0].text(0, 0, f'x̄1-s1')
text_s_1_rght = ax[0].text(0, 0, f'x̄1+s1')
text_s_2_left = ax[0].text(0, 0, f'x̄2-s2')
text_s_2_rght = ax[0].text(0, 0, f'x̄2+s2')
text_s_3_left = ax[0].text(0, 0, f'x̄3-s3')
text_s_3_rght = ax[0].text(0, 0, f'x̄3+s3')
text_s_4_left = ax[0].text(0, 0, f'x̄4-s4')
text_s_4_rght = ax[0].text(0, 0, f'x̄4+s4')

dots1, = ax[0].plot([], [], 'bo', alpha=1.0)
dots2, = ax[0].plot([], [], 'go', alpha=1.0)
dots3, = ax[0].plot([], [], 'mo', alpha=1.0)
dots4, = ax[0].plot([], [], 'yo', alpha=1.0)

vlines1 = ax[0].vlines([], [], [], color='r', alpha=1.0)
vlines2 = ax[0].vlines([], [], [], color='r', alpha=1.0)
vlines3 = ax[0].vlines([], [], [], color='r', alpha=1.0)
vlines4 = ax[0].vlines([], [], [], color='r', alpha=1.0)
vlines5 = ax[0].vlines([], [], [], color='r', alpha=1.0)
fill1 = ax[0].fill(
    [], [], 'b', 
    [], [], 'g', 
    [], [], 'm', 
    [], [], 'y', 
    alpha=0.4, hatch="X"
)
fill2 = ax[0].fill(
    [], [], 'b', 
    [], [], 'g', 
    [], [], 'm', 
    [], [], 'y', 
    alpha=0.4, hatch="X"
)
fill3 = ax[0].fill_between(
    [], []
)

# Distributions
ax[0].plot(POP_1_PROB_DENS_X, POP_1_PROB_DENS_Y, color='b')
ax[0].plot(POP_2_PROB_DENS_X, POP_2_PROB_DENS_Y, color='g')
ax[0].plot(POP_3_PROB_DENS_X, POP_3_PROB_DENS_Y, color='m')
ax[0].plot(POP_4_PROB_DENS_X, POP_4_PROB_DENS_Y, color='y')
# Population Mean
ax[0].vlines([POP_1_MEAN], [0], [POP_1_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
ax[0].vlines([POP_2_MEAN], [0], [POP_2_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
ax[0].vlines([POP_3_MEAN], [0], [POP_3_PROB_MAX], color='red', linestyle='dashed', linewidth=3)
ax[0].vlines([POP_4_MEAN], [0], [POP_4_PROB_MAX], color='red', linestyle='dashed', linewidth=3)

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

    if x < 50 or x == NUMBER_OF_TESTS - 1:
        # Uncomment for more details
        # df = pd.DataFrame(
        #     data = [{'Sample': 'sample 1', 'Value': value} for value in sample1] 
        #     + [{'Sample': 'sample 2', 'Value': value} for value in sample2]
        #     + [{'Sample': 'sample 3', 'Value': value} for value in sample3]
        #     + [{'Sample': 'sample 4', 'Value': value} for value in sample4],
        #     columns=['Sample', 'Value']
        # )
        # model = smf.ols('Value ~ Sample', data=df).fit()
        # table = sm.stats.anova_lm(model)
        # print(table)

        scipy_res = stats.f_oneway(sample1, sample2, sample3, sample4)

        text0.set_text(
            f'Significance Level (α): {ALPHA * 100:.2f} % \n'
            + f'F({ALPHA/2:.4f}, d1={K - 1}, d2={N - K}) Two-Tailed Left: {f_alpha_left:.6f} \n'
            + f'F({1-ALPHA/2:.4f}, d1={K - 1}, d2={N - K}) Two-Tailed Right: {f_alpha_rght:.6f} \n\n'

            + f'Population Mean (μ1): {POP_1_MEAN:.4f} \n'
            + f'Population Mean (μ2): {POP_2_MEAN:.4f} \n'
            + f'Population Mean (μ3): {POP_3_MEAN:.4f} \n'
            + f'Population Mean (μ4): {POP_4_MEAN:.4f} \n'
            + f'Population Standard Deviation (σ1): {POP_1_STD:.4f}\n'
            + f'Population Standard Deviation (σ2): {POP_2_STD:.4f}\n'
            + f'Population Standard Deviation (σ3): {POP_3_STD:.4f}\n'
            + f'Population Standard Deviation (σ4): {POP_4_STD:.4f}\n\n'

            + f'S I - squared (Residual): {S_I_squared:.4f}\n'
            + f'S II - squared (Sample): {S_II_squared:.4f}\n\n'
            + f'Ratio: {ratio:.6f}\n\n'
            + f'SciPy F-Statistic: {scipy_res.statistic:.6f}\n'
            + f'SciPy p-value: {scipy_res.pvalue:.6f}\n\n'
            + f'H0 (μ1=μ2=μ3=μ4) is TRUE: {h0_counter} (Correct)\n'
            + f'H1 (∃i,j,i≠j: μi≠μj) is TRUE: {h1_counter} (False positive)\n\n'
            + f'Actuall Type I Error Percent: {100 * h1_counter / (h0_counter + h1_counter):.2f} %'
            # + f'H0 (μ1=μ2=μ3=μ4) is TRUE: {h0_counter} (False negative)\n'
            # + f'H1 (∃i,j,i≠j: μi≠μj) is TRUE: {h1_counter} (Correct)\n\n'
            # + f'Actuall Type II Error Percent: {100 * h0_counter / (h0_counter + h1_counter):.2f} %'
        )

        text_x_dash_1.set_position((s1_mean, 1.01 * POP_1_PROB_MAX))
        text_x_dash_2.set_position((s2_mean, 1.01 * POP_2_PROB_MAX))
        text_x_dash_3.set_position((s3_mean, 1.01 * POP_3_PROB_MAX))
        text_x_dash_4.set_position((s4_mean, 1.01 * POP_4_PROB_MAX))

        text_s_1_left.set_position((s1_mean - s1_std, 1.01 * POP_1_PROB_MAX))
        text_s_1_rght.set_position((s1_mean + s1_std, 1.01 * POP_1_PROB_MAX))
        text_s_2_left.set_position((s2_mean - s2_std, 1.01 * POP_2_PROB_MAX))
        text_s_2_rght.set_position((s2_mean + s2_std, 1.01 * POP_2_PROB_MAX))
        text_s_3_left.set_position((s3_mean - s3_std, 1.01 * POP_3_PROB_MAX))
        text_s_3_rght.set_position((s3_mean + s3_std, 1.01 * POP_3_PROB_MAX))
        text_s_4_left.set_position((s4_mean - s4_std, 1.01 * POP_4_PROB_MAX))
        text_s_4_rght.set_position((s4_mean + s4_std, 1.01 * POP_4_PROB_MAX))

        # Sample Means
        vlines1.remove()
        vlines1 = ax[0].vlines([s1_mean], [0], [POP_1_PROB_MAX], color='b', linestyle='dashed', linewidth=2)
        vlines2.remove()
        vlines2 = ax[0].vlines([s2_mean], [0], [POP_2_PROB_MAX], color='g', linestyle='dashed', linewidth=2)
        vlines3.remove()
        vlines3 = ax[0].vlines([s3_mean], [0], [POP_3_PROB_MAX], color='m', linestyle='dashed', linewidth=2)
        vlines4.remove()
        vlines4 = ax[0].vlines([s4_mean], [0], [POP_4_PROB_MAX], color='y', linestyle='dashed', linewidth=2)
        vlines5.remove()
        vlines5 = ax[0].vlines([S_mean], [0], [POP_4_PROB_MAX], color='k', linestyle='dashed', linewidth=3)

        # Dots
        dots1.set_data(sample1, stats.norm.pdf(sample1, loc = POP_1_MEAN, scale = POP_1_STD))
        dots2.set_data(sample2, stats.norm.pdf(sample2, loc = POP_2_MEAN, scale = POP_2_STD))
        dots3.set_data(sample3, stats.norm.pdf(sample3, loc = POP_3_MEAN, scale = POP_3_STD))
        dots4.set_data(sample4, stats.norm.pdf(sample4, loc = POP_4_MEAN, scale = POP_4_STD))

        # Confidence Interval
        fill1[0].remove()
        fill1[1].remove()
        fill1[2].remove()
        fill1[3].remove()
        fill1 = ax[0].fill([
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
        fill2 = ax[0].fill([
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
        fill3 = ax[0].fill_between([
                S_mean - S_II, S_mean - S_II, S_mean + S_II, S_mean + S_II
            ], [
                0, POP_4_PROB_MAX, POP_4_PROB_MAX, 0
            ], facecolor="none", alpha=1.0, hatch="-", edgecolor='red')
        
        ax[1].cla()
        ax[1].grid(axis='both', linestyle='--', color='0.95')
        ax[1].xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax[1].set_xlim(POP_4_MEAN - 4.2*POP_4_STD, max(POP_1_MEAN + 4.2*POP_1_STD, POP_4_MEAN + 4.2*POP_4_STD))
        ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        boxplot = ax[1].boxplot([
            sample4, sample3, sample2, sample1
        ], labels = [
            'Sample 4', 'Sample 3', 'Sample 2', 'Sample 1'
        ], showmeans=True, meanline=True, patch_artist=True, vert=False) # showmeans=True, showmedians=False,

        boxplot['boxes'][3].set_facecolor('b')
        boxplot['boxes'][3].set_alpha(0.2)
        boxplot['boxes'][2].set_facecolor('g')
        boxplot['boxes'][2].set_alpha(0.2)
        boxplot['boxes'][1].set_facecolor('m')
        boxplot['boxes'][1].set_alpha(0.2)
        boxplot['boxes'][0].set_facecolor('y')
        boxplot['boxes'][0].set_alpha(0.2)

        plt.tight_layout()
        plt.pause(0.5)

plt.tight_layout()
plt.show()