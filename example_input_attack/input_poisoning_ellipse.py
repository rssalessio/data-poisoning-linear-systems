import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scipystats
from typing import Tuple
from matplotlib.patches import Patch
from plot_options import *
from scipy.stats import f
np.random.seed(200)


TITLE_SIZE = 28
LEGEND_SIZE = 26
TICK_SIZE = 20
AXIS_TITLE = TITLE_SIZE
AXIS_LABEL = TITLE_SIZE
FONT_SIZE = TITLE_SIZE
plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=AXIS_TITLE)     # fontsize of the axes title
plt.rc('axes', labelsize=AXIS_LABEL)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=TITLE_SIZE)  # fontsize of the figure title


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

A = np.array([[0.7]])
B = np.array([[0.5]])

dim_x = 1
dim_u = 1

def collect_data(steps: int, std_u: float, std_w: float) -> Tuple[np.ndarray, np.ndarray]:
    U = np.zeros((dim_u, steps))
    X = np.zeros((dim_x, steps + 1))

    X[:, 0] = np.random.normal(size=(dim_x))

    for i in range(steps):
        U[:, i] = std_u * np.random.normal(size=(dim_u))
        X[:, i+1] = A @ X[:, i] +  np.squeeze(B * U[:, i]) + std_w * np.random.normal(size=(dim_x))

    return X.T, U.T

std_w = 0.1#e-1
std_u = 1
samples = [30, 100, 1000]

fig, ax = plt.subplots(1, 3, figsize=(12,4))

for id, sample_size in enumerate(samples):
    X, U = collect_data(sample_size, std_u, std_w)

    AttackU = std_u * np.random.normal(size=U.shape)
    AB_noU = X[1:].T @ np.linalg.pinv(X[:-1].T)
    AB_poisoned = X[1:].T @ np.linalg.pinv(np.vstack([X[:-1].T, AttackU.T]))
    
    Y1 = AB_noU @ X[:-1].T
    Y2 = AB_poisoned @ np.vstack([X[:-1].T, AttackU.T])
    k = 1
    p = 2
    nu = sample_size - (p+1)
    R1 = np.power(X[1:] - Y1.T, 2).sum()
    R2 = np.power(X[1:] - Y2.T, 2).sum()

    residuals_pois = R2
    F = ((R1-R2)/k)/(  R2 / nu)
    print(f'T: {sample_size} - F: {F} - P: {1-scipystats.f.cdf(F, k, nu)}')

    AB_noU = X[1:].T @ np.linalg.pinv(X[:-1].T)
    AB = X[1:].T @ np.linalg.pinv(np.vstack([X[:-1].T, U.T]))
    
    Y1 = AB_noU @ X[:-1].T
    Y2 = AB @ np.vstack([X[:-1].T, U.T])
    k = 1
    p = 2
    nu = sample_size - (p+1)
    R1 = np.power(X[1:] - Y1.T, 2).sum()
    R2 = np.power(X[1:] - Y2.T, 2).sum()
    residuals_orig = R2
    F = ((R1-R2)/k)/(  R2 / nu)
    print(f'T: {sample_size} - F: {F} - P: {1-scipystats.f.cdf(F, k, nu)}')

    
    data_pois = np.vstack([X[:-1].T, AttackU.T])
    data_orig = np.vstack([X[:-1].T, U.T])

    data_orig = np.linalg.inv(data_orig @ data_orig.T)
    data_pois = np.linalg.inv(data_pois @ data_pois.T)

    std_sq_dev_orig = residuals_orig / (sample_size - 2)
    std_sq_dev_pois = residuals_pois / (sample_size - 2)

    cov_pois = f.ppf(0.975, 2, sample_size-2)*std_sq_dev_pois * data_pois
    cov_orig = f.ppf(0.975, 2, sample_size-2)*std_sq_dev_orig * data_orig

    eigenvalues_pois, eigenvectors_pois = np.linalg.eig(cov_pois)
    eigenvalues_orig, eigenvectors_orig = np.linalg.eig(cov_orig)

    theta = np.linspace(0, 2*np.pi, 1000)
    ellipsis = AB.T+ (np.sqrt(eigenvalues_orig[None,:]) * eigenvectors_orig) @ [np.sin(theta), np.cos(theta)]
    ax[id].plot(ellipsis[0,:], ellipsis[1,:], label='Original')

    ax[id].fill(ellipsis[0,:], ellipsis[1,:], alpha=0.2, facecolor=CB_color_cycle[0],
        edgecolor=CB_color_cycle[0], linewidth=1, zorder=1, hatch="X")
    ax[id].fill_between(ellipsis[0,:], ellipsis[1,:],   hatch='+', zorder=2, facecolor=CB_color_cycle[0], alpha=0.2)

    ellipsis = AB_poisoned.T +(np.sqrt(eigenvalues_pois[None,:]) * eigenvectors_pois) @ [np.sin(theta), np.cos(theta)]
    ax[id].plot( ellipsis[0,:], ellipsis[1,:], label='Poisoned ')
    ax[id].fill(ellipsis[0,:], ellipsis[1,:], alpha=0.2, facecolor=CB_color_cycle[1],
        edgecolor=CB_color_cycle[1], linewidth=1, zorder=1)
    ax[id].fill_between(ellipsis[0,:], ellipsis[1,:],   hatch='///', zorder=2, facecolor=CB_color_cycle[1], alpha=0.2)


    ax[id].set_xlim(0.4, 0.9)
    ax[id].set_ylim(-0.15, 0.7)
    ax[id].set_xlabel(r'$a$',weight='bold')
    if id == 0:
        ax[id].set_ylabel(r'$b$',weight='bold')
    ax[id].grid(alpha=0.3)
    ax[id].set_title(f'T={sample_size} samples')


plt.legend(bbox_to_anchor=(0.735, 0.95), loc="lower right",
                bbox_transform=fig.transFigure, ncol=2, frameon=False)
plt.savefig(f'figures/input_poisoning_{std_w}.pdf',bbox_inches='tight')