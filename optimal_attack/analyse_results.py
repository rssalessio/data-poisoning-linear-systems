import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult
from typing import List, Dict, NamedTuple
import scipy.signal as scipysig
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mathtext as mathtext
from utils import CollectedData, correlate, pvalue_residuals_variance_test, TestStatistics, pvalue_whiteness_test
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


algo_name = 'SLSQP'
T = 500
path = f'./results/{algo_name}/{T}'
dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()

deltas = [0.01, 0.025, 0.05, 0.075, 0.1]
NUM_SEEDS = [i for i in range(1, 10)]
sims = [i for i in range(7)]

class ResultsData(NamedTuple):
    sim_data: CollectedData
    value: float
    AB_unpoisoned: np.ndarray
    AB_poisoned: np.ndarray
    DeltaX: np.ndarray
    DeltaU: np.ndarray
    unpoisoned_residuals: np.ndarray
    poisoned_residuals: np.ndarray
    avg_leverage_unpoisoned: float
    avg_leverage_poisoned: float
    unpoisoned_correlations: np.ndarray
    poisoned_correlations: np.ndarray
    pvalue_residuals_unpoisoned: float
    pvalue_residuals_poisoned: float
    pvalue_correlation_unpoisoned: float
    pvalue_correlation_poisoned: float

trueAB = np.hstack((sys.A, sys.B))

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) #* 180/np.pi
    

def compute_data(data_sim: CollectedData, res_attack: OptimizeResult) -> ResultsData:
    X = data_sim.X
    U = data_sim.U
    dim_u, T = U.shape
    dim_x = X.shape[0]
    D_unpoisoned = np.vstack((X[:,:-1], U))
    AB_unpoisoned = X[:, 1:] @ np.linalg.pinv(D_unpoisoned)
    M = np.linalg.pinv(D_unpoisoned) @ D_unpoisoned
    R = X[:,1:] - AB_unpoisoned @ D_unpoisoned
    avg_leverage_unpoisoned = np.diag(M).mean()

    pvalue_unpoisoned = pvalue_residuals_variance_test(R, dim_u, [data_sim.std_w ** 2] * dim_x).pvalue

    num_lags = int(T * 0.025) * 2

    unpoisoned_correlations = correlate(R, num_lags)
    pvalue_c_unpoisoned =  pvalue_whiteness_test(unpoisoned_correlations, num_lags-1, T).pvalue
    C0inv = np.linalg.inv(unpoisoned_correlations[0])
    unpoisoned_correlations = [np.linalg.norm(unpoisoned_correlations[tau] @ C0inv, 'fro')**2 for tau in range(1, num_lags)]

    Delta = res_attack.x

    DeltaX = Delta[:dim_x * (T+1)].reshape(dim_x, T+1)
    DeltaU = Delta[dim_x * (T+1):].reshape(dim_u, T)
    Xtilde = X + DeltaX
    Utilde = U + DeltaU
    D_poisoned = np.vstack((Xtilde[:,:-1], Utilde))
    Mtilde = np.linalg.pinv(D_poisoned) @ D_poisoned
    AB_poisoned = Xtilde[:, 1:] @ np.linalg.pinv(D_poisoned)
    avg_leverage_poisoned = np.diag(Mtilde).mean()
    Rtilde = Xtilde[:,1:] - AB_poisoned @ D_poisoned

    poisoned_correlations = correlate(Rtilde, num_lags)
    pvalue_c_poisoned =  pvalue_whiteness_test(poisoned_correlations, num_lags-1, T).pvalue
    C0poisonedinv = np.linalg.inv(poisoned_correlations[0])
    poisoned_correlations = [np.linalg.norm(poisoned_correlations[tau] @ C0poisonedinv, 'fro')**2 for tau in range(1, num_lags)]

    statistics = np.array([chi2.cdf(T* x, dim_x**2) for x in unpoisoned_correlations])
    statistics2 = np.array([chi2.cdf(T* x, dim_x**2) for x in poisoned_correlations])
    #print(statistics2/statistics)


    pvalue_poisoned = pvalue_residuals_variance_test(Rtilde, dim_u, [data_sim.std_w ** 2] * dim_x).pvalue
    
    return ResultsData(data_sim, -res_attack.fun, AB_unpoisoned, AB_poisoned, DeltaX, DeltaU, R,
    Rtilde, avg_leverage_unpoisoned, avg_leverage_poisoned, np.array(unpoisoned_correlations), 
    np.array(poisoned_correlations), pvalue_unpoisoned, pvalue_poisoned, pvalue_c_unpoisoned, pvalue_c_poisoned)


summary_results = np.zeros((len(deltas), len(sims), len(NUM_SEEDS)))
summary_pvalues_residuals_poisoned = np.zeros((len(deltas), len(sims), len(NUM_SEEDS)))
summary_pvalues_residuals_unpoisoned = np.zeros((len(deltas), len(sims), len(NUM_SEEDS)))

summary_pvalues_c_poisoned = np.zeros((len(deltas), len(sims), len(NUM_SEEDS)))
summary_pvalues_c_unpoisoned = np.zeros((len(deltas), len(sims), len(NUM_SEEDS)))


summary_residuals_poisoned = np.zeros((len(deltas), len(sims), len(NUM_SEEDS), 500))
summary_residuals_unpoisoned = np.zeros((len(deltas), len(sims), len(NUM_SEEDS), 500))


angle = np.zeros((len(deltas), len(sims), len(NUM_SEEDS)))
sims_data: List[CollectedData] = []
results_data: Dict[float, List[ResultsData]] = {delta: [] for delta in deltas}

for id_sim in sims:
    with open(f'./data/data_{T}timesteps_{id_sim}.pkl', 'rb') as handle:
        sims_data.append(pickle.load(handle))

    for idx_seed, seed in enumerate(NUM_SEEDS):
        for idx_delta, delta in enumerate(deltas):
            suffix_file = f'{id_sim}_{seed}_{delta}'
            with open(f'{path}/results_{suffix_file}.pickle', 'rb') as handle:
                res: OptimizeResult = pickle.load(handle)
                attack_data = compute_data(sims_data[-1], res)

                summary_pvalues_residuals_poisoned[idx_delta, id_sim, idx_seed] = np.linalg.norm(attack_data.poisoned_residuals, 'fro')
                summary_pvalues_residuals_unpoisoned[idx_delta, id_sim, idx_seed] = np.linalg.norm(attack_data.unpoisoned_residuals, 'fro')

                summary_pvalues_c_poisoned[idx_delta, id_sim, idx_seed] = attack_data.pvalue_correlation_poisoned
                summary_pvalues_c_unpoisoned[idx_delta, id_sim, idx_seed] = attack_data.pvalue_correlation_unpoisoned
                
                summary_residuals_poisoned[idx_delta, id_sim, idx_seed] = np.linalg.norm(attack_data.poisoned_residuals, 2, axis=0)
                summary_residuals_unpoisoned[idx_delta, id_sim, idx_seed] = np.linalg.norm(attack_data.unpoisoned_residuals, 2, axis=0)
                
                DeltaAB = attack_data.AB_poisoned - trueAB
                ABnorm = attack_data.AB_unpoisoned.flatten() / np.linalg.norm(attack_data.AB_unpoisoned.flatten())
                angle[idx_delta, id_sim, idx_seed] = (180/np.pi)*angle_between(ABnorm, DeltaAB.flatten())#np.abs(np.dot(ABnorm, DeltaAB.flatten()))

                
                results_data[delta].append(attack_data)
                summary_results[idx_delta, id_sim, idx_seed] = attack_data.value




summary_residuals = np.vstack(([summary_residuals_unpoisoned[0]], summary_residuals_poisoned))
summary_residuals=summary_residuals.reshape(6,-1,500).mean(1)
plt.boxplot(summary_residuals.T)
plt.show()
import pdb
pdb.set_trace()


# results_norm_error = summary_results.max(-1)

# fig, ax = plt.subplots(1, 3, figsize=(16,4))

# ax[0].boxplot(results_norm_error.T, labels=deltas)



fig, ax = plt.subplots(1,2)
for delta in [0.1]:
    data = results_data[delta]
    unpoisoned_residuals = np.zeros((len(data),)+data[0].unpoisoned_residuals.shape)
    poisoned_residuals = np.zeros((len(data),)+ data[0].unpoisoned_residuals.shape)

    unpoisoned_corr = np.zeros((len(data),)+data[0].unpoisoned_correlations.shape)
    poisoned_corr = np.zeros((len(data),)+ data[0].unpoisoned_correlations.shape)
    for i in range(len(data)):
        unpoisoned_residuals[i] = data[i].unpoisoned_residuals
        poisoned_residuals[i] = data[i].poisoned_residuals

        unpoisoned_corr[i] = data[i].unpoisoned_correlations
        poisoned_corr[i] = data[i].poisoned_correlations

    X = np.linalg.norm(unpoisoned_residuals, 2, axis=1)
    Y = np.linalg.norm(poisoned_residuals, 2, axis=1)
    Z = np.vstack(([X],[Y])).reshape(2,-1)
    import pdb
    pdb.set_trace()
    ax[0].plot(np.linalg.norm(unpoisoned_residuals, 2, axis=1).mean(0), label='unpoisoned')
    ax[0].plot(np.linalg.norm(poisoned_residuals, 2, axis=1).mean(0), label='poisoned')

    ax[1].plot(unpoisoned_corr.mean(0), label='unpoisoned')
    ax[1].plot(poisoned_corr.mean(0), label='poisoned')

plt.legend()
plt.grid()
plt.show()

results_norm_error = summary_results.max(-1)

fig, ax = plt.subplots(1, 3, figsize=(16,4))

ax[0].boxplot(results_norm_error.T, labels=deltas)

for idx_delta, delta in enumerate(deltas):
    ax[0].scatter((1+idx_delta)* np.ones_like(results_norm_error[idx_delta]), results_norm_error[idx_delta], alpha=0.4)

ax[0].set_xlabel('$\delta$')
ax[0].set_ylabel(r"$\mathrm{E}\|\begin{bmatrix} \Delta A & \Delta B \end{bmatrix}\|_2$")
ax[0].grid()


dim_x = 4
dim_u = 1
U = results_data[0.1][-1].sim_data.U
X = results_data[0.1][-1].sim_data.X
DeltaU = results_data[0.1][-1].DeltaU
DeltaX = results_data[0.1][-1].DeltaX


ax[1].plot(np.linalg.norm( U, axis=0), label=r'$\|u_t\|_2$')
ax[1].plot(np.linalg.norm( DeltaU, axis=0), label=r'$\|\Delta u_t\|_2$')


ax[1].set_xlabel('$t$')
ax[1].legend(bbox_to_anchor=(1.1,0.95), loc="lower right", ncol=2, frameon=False)
ax[1].grid()


ax[2].plot(np.linalg.norm( X, axis=0), label=r'$\|x_t\|_2$')
ax[2].plot(np.linalg.norm( DeltaX, axis=0), label=r'$\|\Delta x_t\|_2$')

ax[2].set_xlabel('$t$')
ax[2].legend(bbox_to_anchor=(1.1, 0.95), loc="lower right", ncol=2, frameon=False)
ax[2].grid()

summary_pvalues_residuals = summary_pvalues_residuals_poisoned /summary_pvalues_residuals_unpoisoned

# summary_pvalues_residuals = np.hstack((summary_pvalues_residuals_unpoisoned, summary_pvalues_residuals_poisoned))
summary_pvalues_residuals = summary_pvalues_residuals.reshape(5, -1)

# ax[3].boxplot(summary_pvalues_residuals.T, labels= deltas)


# for idx in range(len(deltas)):
#     ax[3].scatter((1+idx)* np.ones_like(summary_pvalues_residuals[idx]), summary_pvalues_residuals[idx], alpha=0.4)

# ax[3].set_xlabel('$\delta$')
# ax[3].grid()



# summary_pvalues_c = np.hstack((summary_pvalues_c_unpoisoned, summary_pvalues_c_poisoned))
# summary_pvalues_c = summary_pvalues_c.reshape(6, -1)

# ax[4].boxplot(summary_pvalues_c.T, labels=[0]+ deltas)

# for idx in range(len(deltas) + 1):
#     ax[4].scatter((1+idx)* np.ones_like(summary_pvalues_c[idx]), summary_pvalues_c[idx], alpha=0.4)

# ax[4].set_xlabel('$\delta$')
# ax[4].grid()


# summary_angle = angle.reshape(len(deltas), -1)
# ax[4].boxplot(summary_angle.T, labels=deltas)

# for idx in range(len(deltas)):
#     ax[4].scatter((1+idx)* np.ones_like(summary_angle[idx]), summary_angle[idx], alpha=0.4)

# ax[4].set_xlabel('$\delta$')
# ax[4].grid()


plt.show()