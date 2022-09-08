from abc import ABC
import numpy as np
import scipy.signal as scipysig
from typing import Tuple
from pyzonotope import MatrixZonotope, Zonotope, concatenate_zonotope
from pydatadrivenreachability import compute_LTI_matrix_zonotope
import scipy.stats as scipystats
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from constants import *
np.random.seed(200)


TITLE_SIZE = 30
LEGEND_SIZE = 20
TICK_SIZE = 17
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
        X[:, i+1] = A @ X[:, i] +  np.squeeze(B * U[:, i]) + std_w * np.random.normal(size=(dim_x)) #(low=-0.1, high=0.1)#(size=(dim_x))

    return X.T, U.T

std_w = 1e-1
std_u = 1
samples = [15, 100, 1000]

fig, ax = plt.subplots(1, 3, figsize=(12,6))
#fig.suptitle("Identification of $(a,b)$ - Input poisoning", fontsize=TITLE_SIZE)

for id, sample_size in enumerate(samples):
    W = Zonotope([0], [[std_w]])
    X, U = collect_data(sample_size, std_u, std_w)
    Mw = concatenate_zonotope(W, sample_size)

    Msigma: MatrixZonotope = compute_LTI_matrix_zonotope(X[:-1], X[1:], U, Mw)

    AttackU = std_u * np.random.normal(size=U.shape)
    Msigma_poisoned: MatrixZonotope = compute_LTI_matrix_zonotope(X[:-1], X[1:], AttackU, Mw)
    Mz = Msigma.zonotope.reduce(100)
    Mz_poisoned = Msigma_poisoned.zonotope.reduce(100)

    AB_noU = X[1:].T @ np.linalg.pinv(X[:-1].T)
    AB_poisoned = X[1:].T @ np.linalg.pinv(np.vstack([X[:-1].T, AttackU.T]))
    
    Y1 = AB_noU @ X[:-1].T
    Y2 = AB_poisoned @ np.vstack([X[:-1].T, AttackU.T])
    k = 1
    p = 2
    nu = sample_size - (p+1)
    R1 = np.power(X[1:] - Y1.T, 2).sum()
    R2 = np.power(X[1:] - Y2.T, 2).sum()
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
    F = ((R1-R2)/k)/(  R2 / nu)
    print(f'T: {sample_size} - F: {F} - P: {1-scipystats.f.cdf(F, k, nu)}')

    

    collection1 = PatchCollection([Mz.polygon],  facecolor=CB_color_cycle[0], edgecolor='black', lw=1.2, label='Original data')
    collection2 = PatchCollection([Mz_poisoned.polygon],  facecolor=CB_color_cycle[1], edgecolor='black', lw=1.2, label='Poisoned data')

    ax[id].add_collection(collection1)
    ax[id].add_collection(collection2)


    ax[id].set_xlim(0.4, 0.9)
    ax[id].set_ylim(-0.15, 0.7)
    ax[id].set_xlabel(r'$a$',weight='bold')
    if id == 0:
        ax[id].set_ylabel(r'$b$',weight='bold')
    ax[id].grid(alpha=0.3)
    ax[id].set_title(f'T={sample_size} samples')
    # if id > 0:
    #     ax[id].get_yaxis().set_ticklabels([])

    
    if id == 0:
        ax[id].legend(fancybox = True,facecolor="whitesmoke", loc='lower left', handles = [Patch(color=CB_color_cycle[0], label='Original data'), Patch(color=CB_color_cycle[1], label='Poisoned data')] )
#fig.tight_layout(rect=[0, 0, 1, 0.95])



plt.savefig('input_poisoning.pdf',bbox_inches='tight')