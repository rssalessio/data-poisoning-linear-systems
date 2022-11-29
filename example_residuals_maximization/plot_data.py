import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import List, Tuple
from utils import ResultsData, residuals_variance_test, confidence_interval_signal
TITLE_SIZE = 20
LEGEND_SIZE = 20
TICK_SIZE = 14
AXIS_TITLE = TITLE_SIZE-4
AXIS_LABEL = TITLE_SIZE-4
FONT_SIZE = TITLE_SIZE-4

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
def project(w, d):
    return d*np.dot(w, d)/np.dot(d, d)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) #* 180/np.pi
    
def evaluate_attack(Xm: np.ndarray, Xp: np.ndarray, U: np.ndarray, DeltaX: np.ndarray, DeltaU: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tildeXm = Xm + DeltaX[:, :-1]
    tildeXp = Xp + DeltaX[:, 1:]
    tildeU = U + DeltaU
    Dtilde = np.vstack((tildeXm, tildeU))

    ABtilde = tildeXp @ np.linalg.pinv(Dtilde)

    residuals = tildeXp - ABtilde @ Dtilde        

    return ABtilde, residuals


results: List[ResultsData] = np.load('data/data.npy', allow_pickle=True)
N_SIMS = len(results)
deltas = results[0].deltas
dim_x, dim_u = results[0].X.shape[0], results[0].U.shape[0]
sigma_eigs = np.linalg.eigvals(results[0].Sigma).tolist()
N_DELTAS = len(deltas)

opt_poisoned_delta_norm = np.zeros((N_SIMS,  N_DELTAS))
gauss_poisoned_delta_norm = np.zeros((N_SIMS,  N_DELTAS))
cem_poisoned_delta_norm = np.zeros((N_SIMS,  N_DELTAS))

opt_poisoned_delta_test = np.zeros((N_SIMS,  N_DELTAS))
gauss_poisoned_delta_test = np.zeros((N_SIMS,  N_DELTAS))
cem_poisoned_delta_test = np.zeros((N_SIMS,  N_DELTAS))

opt_poisoned_delta_dot = np.zeros((N_SIMS,  N_DELTAS))
gauss_poisoned_delta_dot = np.zeros((N_SIMS,  N_DELTAS))
cem_poisoned_delta_dot = np.zeros((N_SIMS,  N_DELTAS))

for i in range(N_SIMS):
    res = results[i]
    Xm, Xp = res.X[:, :-1], res.X[:, 1:]
    AB = Xp @ np.linalg.pinv(np.vstack((Xm, res.U)))
    ABnorm = AB.flatten() / np.linalg.norm(AB.flatten())

    for j in range(len(res.deltas)):
        ABtilde, poisoned_residuals = evaluate_attack(Xm, Xp, res.U, res.opt_DeltaX[j], res.opt_DeltaU[j])
        opt_poisoned_delta_norm[i,j] = np.linalg.norm(ABtilde - res.TrueAB, 2)
        opt_poisoned_delta_test[i,j] = residuals_variance_test(poisoned_residuals, dim_u, sigma_eigs).p_right

        DeltaAB = ABtilde - res.TrueAB
        opt_poisoned_delta_dot[i,j] = angle_between(ABnorm, DeltaAB.flatten())# np.abs(np.dot(ABnorm, DeltaAB.flatten()))

        ABtilde, poisoned_residuals = evaluate_attack(Xm, Xp, res.U, res.gauss_DeltaX[j], res.gauss_DeltaU[j])
        gauss_poisoned_delta_norm[i,j] = np.linalg.norm(ABtilde - res.TrueAB, 2)
        gauss_poisoned_delta_test[i,j] = residuals_variance_test(poisoned_residuals, dim_u, sigma_eigs).p_right

        DeltaAB = ABtilde - res.TrueAB
        gauss_poisoned_delta_dot[i,j] = angle_between(ABnorm, DeltaAB.flatten())#np.abs(np.dot(ABnorm, DeltaAB.flatten()))


        ABtilde, poisoned_residuals = evaluate_attack(Xm, Xp, res.U, res.cem_DeltaX[j], res.cem_DeltaU[j])
        cem_poisoned_delta_norm[i,j] = np.linalg.norm(ABtilde - res.TrueAB, 2)
        cem_poisoned_delta_test[i,j] = residuals_variance_test(poisoned_residuals, dim_u, sigma_eigs).p_right

        DeltaAB = ABtilde - res.TrueAB
        cem_poisoned_delta_dot[i,j] = angle_between(ABnorm, DeltaAB.flatten())#np.abs(np.dot(ABnorm, DeltaAB.flatten()))

        

fig, ax = plt.subplots(1, 3, figsize=(17.5,4))

impact_optimal_mean, impact_optimal_lower, impact_optimal_upper = confidence_interval_signal(opt_poisoned_delta_norm, N_SIMS)
impact_gaussian_mean, impact_gaussian_lower, impact_gaussian_upper = confidence_interval_signal(gauss_poisoned_delta_norm, N_SIMS)
impact_cem_mean, impact_cem_lower, impact_cem_upper = confidence_interval_signal(cem_poisoned_delta_norm, N_SIMS)

ax[0].plot(deltas, impact_optimal_mean, label='Optimized attack')
ax[0].fill_between(deltas, impact_optimal_lower, impact_optimal_upper, alpha=0.3)
ax[0].plot(deltas, impact_cem_mean, ':', label='CEM attack')
ax[0].fill_between(deltas, impact_cem_lower, impact_cem_upper, alpha=0.3)
ax[0].plot(deltas, impact_gaussian_mean, '--', label='Gaussian attack')
ax[0].fill_between(deltas, impact_gaussian_lower, impact_gaussian_upper, alpha=0.3)

ax[0].grid()
ax[0].set_xscale('log')
ax[0].set_xlabel(r'$\delta$')
ax[0].set_ylabel(r"$\mathrm{E}\|\begin{bmatrix} \Delta A & \Delta B \end{bmatrix}\|_2$")

resvar_optimal_mean, resvar_optimal_lower, resvar_optimal_upper = confidence_interval_signal(opt_poisoned_delta_test, N_SIMS)
resvar_gaussian_mean, resvar_gaussian_lower, resvar_gaussian_upper = confidence_interval_signal(gauss_poisoned_delta_test, N_SIMS)
resvar_cem_mean, resvar_cem_lower, resvar_cem_upper = confidence_interval_signal(cem_poisoned_delta_test, N_SIMS)

ax[1].plot(deltas, resvar_optimal_mean, label='Optimized attack')
ax[1].fill_between(deltas, resvar_optimal_lower, resvar_optimal_upper, alpha=0.3)
ax[1].plot(deltas, resvar_cem_mean, ':', label='CEM attack')
ax[1].fill_between(deltas, resvar_cem_lower, resvar_cem_upper, alpha=0.3)
ax[1].plot(deltas, resvar_gaussian_mean, '--', label='Gaussian attack')
ax[1].fill_between(deltas, resvar_gaussian_lower, resvar_gaussian_upper, alpha=0.3)

ax[1].grid()
ax[1].set_xlabel(r'$\delta$')
ax[1].set_ylabel(r'$P_0(X\leq  \|\tilde R\|_{\textnormal{F}}^2 )$')
ax[1].set_xscale('log')


dot_optimal_mean, dot_optimal_lower, dot_optimal_upper = confidence_interval_signal(opt_poisoned_delta_dot, N_SIMS)
dot_gaussian_mean, dot_gaussian_lower, dot_gaussian_upper = confidence_interval_signal(gauss_poisoned_delta_dot, N_SIMS)
dot_cem_mean, dot_cem_lower, dot_cem_upper = confidence_interval_signal(cem_poisoned_delta_dot, N_SIMS)

ax[2].plot(deltas, dot_optimal_mean, label='Optimized attack')
ax[2].fill_between(deltas, dot_optimal_lower, dot_optimal_upper, alpha=0.3)
ax[2].plot(deltas, dot_cem_mean, ':', label='CEM attack')
ax[2].fill_between(deltas, dot_cem_lower, dot_cem_upper, alpha=0.3)
ax[2].plot(deltas, dot_gaussian_mean, '--', label='Gaussian attack')
ax[2].fill_between(deltas, dot_gaussian_lower, dot_gaussian_upper, alpha=0.3)

ax[2].grid()
ax[2].set_xlabel(r'$\delta$')
ax[2].set_ylabel(r'$\angle (\theta_{\textrm{LS}},  \Delta \tilde{\theta}_{\textrm{LS}})$ [rad]')
ax[2].set_xscale('log')
print([round(r/np.pi,2) for r in ax[2].get_yticks()])
print([r"$." + str(round(r/np.pi,2))[2:]+ r"\pi$" for r in ax[2].get_yticks()])
ax[2].set_yticklabels([r"$." + str(round(r/np.pi,2))[2:]+ r"\pi$" for r in ax[2].get_yticks()])


plt.legend(bbox_to_anchor=(0.750, 0.875), loc="lower right",
                bbox_transform=fig.transFigure, ncol=3, frameon=False)
plt.savefig('figures/example_residuals_variance.png',bbox_inches='tight',dpi=300)
