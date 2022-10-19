import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mathtext as mathtext
import numpy as np

from typing import List
from utils import ResultsData, confidence_interval_signal

TITLE_SIZE = 20
LEGEND_SIZE = 20
TICK_SIZE = 16
AXIS_TITLE = TITLE_SIZE
AXIS_LABEL = TITLE_SIZE
FONT_SIZE = TITLE_SIZE

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
mpl.rcParams['font.family'] = "serif"

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=AXIS_TITLE)     # fontsize of the axes title
plt.rc('axes', labelsize=AXIS_LABEL)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=TITLE_SIZE)  # fontsize of the figure title

deltas = np.geomspace(1e-4, 1e-1, 50)
NUM_SIMS = 100
true_delta_ab_norm = np.zeros((NUM_SIMS))
true_residuals_variance = np.zeros((NUM_SIMS))

poisoned_delta_ab_norm_optimal_atk = np.zeros((NUM_SIMS,len(deltas)))
poisoned_delta_ab_norm_gaussian_atk = np.zeros((NUM_SIMS,len(deltas)))
poisoned_residuals_variance_optimal_atk = np.zeros((NUM_SIMS,len(deltas)))
poisoned_residuals_variance_gaussian_atk = np.zeros((NUM_SIMS,len(deltas)))


results: List[ResultsData] = np.load('data_residuals_variance.npy', allow_pickle=True)


for i in range(len(results)):
    true_delta_ab_norm[i] = results[i].true_delta_ab_norm
    true_residuals_variance[i] = results[i].true_residuals_variance

    poisoned_delta_ab_norm_optimal_atk[i,:] = results[i].poisoned_delta_ab_norm_optimal_atk
    poisoned_delta_ab_norm_gaussian_atk[i,:] = results[i].poisoned_delta_ab_norm_gaussian_atk
    poisoned_residuals_variance_optimal_atk[i,:] = results[i].poisoned_residuals_variance_optimal_atk
    poisoned_residuals_variance_gaussian_atk[i,:] = results[i].poisoned_residuals_variance_gaussian_atk




fig, ax = plt.subplots(1, 2, figsize=(12,5))

impact_optimal_mean, impact_optimal_lower, impact_optimal_upper = confidence_interval_signal(poisoned_delta_ab_norm_optimal_atk, NUM_SIMS)
impact_gaussian_mean, impact_gaussian_lower, impact_gaussian_upper = confidence_interval_signal(poisoned_delta_ab_norm_gaussian_atk, NUM_SIMS)
ax[0].plot(deltas, impact_optimal_mean, label='Optimized attack')
ax[0].fill_between(deltas, impact_optimal_lower, impact_optimal_upper, alpha=0.3)
ax[0].plot(deltas, impact_gaussian_mean, label='Gaussian attack')
ax[0].fill_between(deltas, impact_gaussian_lower, impact_gaussian_upper, alpha=0.3)
ax[0].grid()
ax[0].set_xscale('log')
ax[0].set_xlabel(r'$\delta$')
ax[0].set_ylabel(r"$\|\begin{bmatrix} \Delta A & \Delta B \end{bmatrix}\|_2$")


resvar_optimal_mean, resvar_optimal_lower, resvar_optimal_upper = confidence_interval_signal(poisoned_residuals_variance_optimal_atk, NUM_SIMS)
resvar_gaussian_mean, resvar_gaussian_lower, resvar_gaussian_upper = confidence_interval_signal(poisoned_residuals_variance_gaussian_atk, NUM_SIMS)
ax[1].plot(deltas, resvar_optimal_mean, label='Optimized attack')
ax[1].fill_between(deltas, resvar_optimal_lower, resvar_optimal_upper, alpha=0.3)
ax[1].plot(deltas, resvar_gaussian_mean, label='Gaussian attack')
ax[1].fill_between(deltas, resvar_gaussian_lower, resvar_gaussian_upper, alpha=0.3)
ax[1].grid()
ax[1].set_xlabel(r'$\delta$')
ax[1].set_ylabel(r'$P_0(X\leq  \|\tilde R\|_{\textnormal{F}}^2 )$')
ax[1].set_xscale('log')
plt.legend(bbox_to_anchor=(0.725, 0.875), loc="lower right",
                bbox_transform=fig.transFigure, ncol=2, frameon=False)

plt.savefig('example_residuals_variance.pdf',bbox_inches='tight')