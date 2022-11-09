import numpy as np
from typing import Tuple
import scipy.stats as scipystats
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from utils import collect_data, CollectedData
import scipy.signal as scipysig
from scipy.special import gamma
np.random.seed(200)

A = np.array([[0.7]])
B = np.array([[0.5]])

dim_x = 1
dim_u = 1

sys = scipysig.StateSpace(A, B, np.eye(dim_x), 0*B, dt=1)

std_w = 1e-1
std_u = 1

T = 30
data = collect_data(T, std_u, std_w, sys)

tildeU = std_u * np.random.normal(size=data.U.shape)

Psi = np.vstack([data.X[:, :-1], data.U])
Psi_poisoned = np.vstack([data.X[:, :-1], tildeU])

AB = data.X[:, 1:] @ np.linalg.pinv(Psi)
AB_poisoned = data.X[:, 1:] @ np.linalg.pinv(Psi_poisoned)

def gen_gamma_fun(x: float, p: int):
    return  (np.pi ** (p*(p-1)/4)) * np.prod(gamma((2*x + 1 - np.arange(1, p+1))/2))

def compute_marginal(data: CollectedData, mu0: np.ndarray, k0: float, Lmbd0: np.ndarray, nu0: float):
    #n = T
    T = data.U.shape[1]
    kT = k0 + T
    nuT = nu0 + T
    d = data.X.shape[1]

    Psi = np.vstack([data.X[:, :-1], data.U])
    AB = data.X[:, 1:] @ np.linalg.pinv(Psi)
    
    residuals = data.X[:, 1:] - AB @ Psi

    res0 = AB @ Psi - mu0 @ Psi

    S = residuals @ residuals.T
    
    LmbdT = Lmbd0 + S + res0 @ res0.T * (k0 * T) /  kT

    t1 = 1#( 1/ np.pi )** (T*d/2)
    t2 = gen_gamma_fun(nuT/2, d) / gen_gamma_fun(nu0/2, d)
    t3 = (np.linalg.det(Lmbd0) ** (nu0 / 2)) / (np.linalg.det(LmbdT) ** (nuT / 2))
    t4 = (k0 / kT) ** (d/2)
    import pdb
    pdb.set_trace()
    return t1 * t2 * t3 * t4

res = compute_marginal(
    CollectedData(data.X, tildeU, data.W, data.std_u, data.std_w),
    mu0 = np.array([[0.65, 0.55]]),
    k0 = 1,
    Lmbd0 = np.eye(1),
    nu0= 1
)

print(res)



# fig, ax = plt.subplots(1, 3, figsize=(12,4))
# #fig.suptitle("Identification of $(a,b)$ - Input poisoning", fontsize=TITLE_SIZE)

# for id, sample_size in enumerate(samples):
#     X, U = collect_data(sample_size, std_u, std_w)

    # AttackU = std_u * np.random.normal(size=U.shape)
    # AB_noU = X[1:].T @ np.linalg.pinv(X[:-1].T)
    # AB_poisoned = X[1:].T @ np.linalg.pinv(np.vstack([X[:-1].T, AttackU.T]))
    
    # Y1 = AB_noU @ X[:-1].T
    # Y2 = AB_poisoned @ np.vstack([X[:-1].T, AttackU.T])
#     k = 1
#     p = 2
#     nu = sample_size - (p+1)
#     R1 = np.power(X[1:] - Y1.T, 2).sum()
#     R2 = np.power(X[1:] - Y2.T, 2).sum()

#     residuals_pois = R2
#     F = ((R1-R2)/k)/(  R2 / nu)
#     print(f'T: {sample_size} - F: {F} - P: {1-scipystats.f.cdf(F, k, nu)}')



#     AB_noU = X[1:].T @ np.linalg.pinv(X[:-1].T)
#     AB = X[1:].T @ np.linalg.pinv(np.vstack([X[:-1].T, U.T]))
    
#     Y1 = AB_noU @ X[:-1].T
#     Y2 = AB @ np.vstack([X[:-1].T, U.T])
#     k = 1
#     p = 2
#     nu = sample_size - (p+1)
#     R1 = np.power(X[1:] - Y1.T, 2).sum()
#     R2 = np.power(X[1:] - Y2.T, 2).sum()
#     residuals_orig = R2
#     F = ((R1-R2)/k)/(  R2 / nu)
#     print(f'T: {sample_size} - F: {F} - P: {1-scipystats.f.cdf(F, k, nu)}')

    
#     data_pois = np.vstack([X[:-1].T, AttackU.T])
#     data_orig = np.vstack([X[:-1].T, U.T])

#     data_orig = np.linalg.inv(data_orig @ data_orig.T)
#     data_pois = np.linalg.inv(data_pois @ data_pois.T)

#     std_sq_dev_orig = residuals_orig / (sample_size - 2)
#     std_sq_dev_pois = residuals_pois / (sample_size - 2)

#     cov_pois = 5.99*std_sq_dev_pois * data_pois
#     cov_orig = 5.99*std_sq_dev_orig * data_orig

#     eigenvalues_pois, eigenvectors_pois = np.linalg.eig(cov_pois)
#     eigenvalues_orig, eigenvectors_orig = np.linalg.eig(cov_orig)

#     theta = np.linspace(0, 2*np.pi, 1000)
#     ellipsis = AB.T+ (np.sqrt(eigenvalues_orig[None,:]) * eigenvectors_orig) @ [np.sin(theta), np.cos(theta)]
#     ax[id].plot(ellipsis[0,:], ellipsis[1,:], label='Original')

#     ax[id].fill(ellipsis[0,:], ellipsis[1,:], alpha=0.2, facecolor=CB_color_cycle[0],
#         edgecolor=CB_color_cycle[0], linewidth=1, zorder=1, hatch="X")
#     ax[id].fill_between(ellipsis[0,:], ellipsis[1,:],   hatch='+', zorder=2, facecolor=CB_color_cycle[0], alpha=0.2)

#     ellipsis = AB_poisoned.T +(np.sqrt(eigenvalues_pois[None,:]) * eigenvectors_pois) @ [np.sin(theta), np.cos(theta)]
#     ax[id].plot( ellipsis[0,:], ellipsis[1,:], label='Poisoned ')
#     ax[id].fill(ellipsis[0,:], ellipsis[1,:], alpha=0.2, facecolor=CB_color_cycle[1],
#         edgecolor=CB_color_cycle[1], linewidth=1, zorder=1)
#     ax[id].fill_between(ellipsis[0,:], ellipsis[1,:],   hatch='///', zorder=2, facecolor=CB_color_cycle[1], alpha=0.2)


#     ax[id].set_xlim(0.4, 0.9)
#     ax[id].set_ylim(-0.15, 0.7)
#     ax[id].set_xlabel(r'$a$',weight='bold')
#     if id == 0:
#         ax[id].set_ylabel(r'$b$',weight='bold')
#     ax[id].grid(alpha=0.3)
#     ax[id].set_title(f'T={sample_size} samples')
#     # if id > 0:
#     #     ax[id].get_yaxis().set_ticklabels([])

    
#     # if id == 1:
#     #     ax[id].legend(
#     #         fancybox = True,facecolor="whitesmoke", loc='lower left', handles = [Patch(color=CB_color_cycle[0], label='Original'), Patch(color=CB_color_cycle[1], label='Poisoned')] )
# #fig.tight_layout(rect=[0, 0, 1, 0.95])


# plt.legend(bbox_to_anchor=(0.735, 0.95), loc="lower right",
#                 bbox_transform=fig.transFigure, ncol=2, frameon=False)
# plt.savefig('input_poisoning.pdf',bbox_inches='tight')