import numpy as np
import scipy.signal as scipysig
from typing import Tuple
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import cvxpy as cp
import scipy.stats as stats
import dccp
from momentchi2 import hbe
np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
dim_x, dim_u = sys.B.shape
T = 200

def collect_data(steps: int, std_u: float, std_w: float, sys: scipysig.StateSpace) -> Tuple[np.ndarray, np.ndarray]:
    dim_x, dim_u = sys.B.shape
    U = np.zeros((dim_u, steps))
    X = np.zeros((dim_x, steps + 1))
    W = np.zeros((dim_x, steps))
    X[:, 0] = np.random.normal(size=(dim_x))

    for i in range(steps):
        U[:, i] = std_u * np.random.normal(size=(dim_u))
        W[:, i] = std_w * np.random.normal(size=(dim_x))
        X[:, i+1] = sys.A @ X[:, i] +  np.squeeze(sys.B * U[:, i]) + W[:, i]

    return X, U, W

def correlate(x: np.ndarray, num_lags: int):
    n, T = x.shape
    R = np.zeros((num_lags,n,n))

    for m in range(num_lags):
        for i in range(m, T):
            R[m] += x[:,i:i+1] @ x[:, i-m:i-m+1].T

    return R/ T

std_w = 1e-1
std_u = 1

NUM_SIMS = 10
deltas = np.geomspace(1e-5, 1e-1, 40)

results_optimal = np.zeros((NUM_SIMS,len(deltas)))
poisoned_residuals_optimal = np.zeros((NUM_SIMS,len(deltas)))
abdiff_optimal = np.zeros((NUM_SIMS,len(deltas)))
test_poisoned_optimal = np.zeros((NUM_SIMS,len(deltas)))
poisoned_residuals_gaussian = np.zeros((NUM_SIMS,len(deltas)))
abdiff_gaussian = np.zeros((NUM_SIMS,len(deltas)))
test_poisoned_gaussian = np.zeros((NUM_SIMS,len(deltas)))



for sim_idx in range(NUM_SIMS):
    X, U, W = collect_data(T, std_u, std_w, sys)
    Xp, Xm = X[:, 1:], X[:, :-1]
    D = np.vstack((Xm, U))
    AB = Xp @ np.linalg.pinv(D)

    true_residuals = Xp - AB @ D 
    test_variance = np.linalg.norm(true_residuals, 'fro')

    def evaluate_attack(DeltaX, DeltaU):
        tildeXm = Xm + DeltaX[:, :-1]
        tildeXp = Xp + DeltaX[:, 1:]
        tildeU = U + DeltaU
        Dtilde = np.vstack((tildeXm, tildeU))

        ABtilde = tildeXp @ np.linalg.pinv(Dtilde)


        _pos_res = tildeXp - ABtilde @ Dtilde        
        _pois_res= np.linalg.norm(_pos_res, 'fro')
        _abdiff = np.linalg.norm(np.hstack((sys.A, sys.B)) - ABtilde, 2)


        _test_pois=hbe(coeff=[std_w**2 ] * dim_x* (T-dim_x-dim_u), x=_pois_res** 2)

        return _pois_res, _abdiff, _test_pois

    for idx_delta, delta in enumerate(deltas):
        print(f'Iteration {sim_idx}-{idx_delta}')
        DeltaX = cp.Variable((dim_x, T+1))
        DeltaU = cp.Variable((dim_u, T))


        objective = cp.norm(DeltaX[:, 1:] - AB @ cp.vstack((DeltaX[:, :-1], DeltaU)), 'fro')
        objective += 2*cp.trace(true_residuals @ (DeltaX[:, 1:] - AB @ cp.vstack((DeltaX[:, :-1], DeltaU))).T)
        constraints = [
            cp.norm(DeltaX, 'fro') <= delta * np.linalg.norm(X, 'fro'),
            cp.norm(DeltaU, 'fro') <= delta * np.linalg.norm(U, 'fro') 
        ]
        problem = cp.Problem(cp.Maximize(objective), constraints)
        result = problem.solve(method='dccp', ccp_times=10, solver=cp.MOSEK)

        results_optimal[sim_idx,idx_delta] = result[0]

        _pois_res, _abdiff, _test_pois = evaluate_attack(DeltaX.value, DeltaU.value)
        poisoned_residuals_optimal[sim_idx,idx_delta] = _pois_res
        abdiff_optimal[sim_idx,idx_delta] = _abdiff
        test_poisoned_optimal[sim_idx,idx_delta]=_test_pois

        stdX = (delta * np.linalg.norm(X, 'fro')) / np.sqrt(T* dim_x)
        stdU = (delta * np.linalg.norm(U, 'fro')) / np.sqrt(T* dim_u)


        
        DeltaX = np.random.normal(size=X.shape) * stdX
        DeltaU = np.random.normal(size=U.shape) * stdU


        _pois_res, _abdiff, _test_pois = evaluate_attack(DeltaX, DeltaU)
        poisoned_residuals_gaussian[sim_idx,idx_delta] = _pois_res
        abdiff_gaussian[sim_idx,idx_delta] = _abdiff
        test_poisoned_gaussian[sim_idx,idx_delta] =_test_pois


fig, ax1 = plt.subplots()


mu = abdiff_optimal.mean(0)
std = abdiff_optimal.std(0)

ax1.plot(deltas, mu, label='optimal')
ax1.fill_between(deltas, mu - 1.96 * std/np.sqrt(NUM_SIMS), mu + 1.96 * std / np.sqrt(NUM_SIMS), alpha=0.3)
mu = abdiff_gaussian.mean(0)
std = abdiff_gaussian.std(0)

ax1.plot(deltas, mu, label='gaussian')
ax1.fill_between(deltas, mu - 1.96 * std/np.sqrt(NUM_SIMS), mu + 1.96 * std / np.sqrt(NUM_SIMS), alpha=0.3)
ax1.grid()
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend()


mu = test_poisoned_optimal.mean(0)
std = test_poisoned_optimal.std(0)

ax2 = ax1.twinx() 
ax2.plot(deltas,mu)
ax2.fill_between(deltas, mu - 1.96 * std/np.sqrt(NUM_SIMS), mu + 1.96 * std / np.sqrt(NUM_SIMS), alpha=0.3)
mu = test_poisoned_gaussian.mean(0)
std = test_poisoned_gaussian.std(0)

ax2.plot(deltas,mu)
ax2.fill_between(deltas, mu - 1.96 * std/np.sqrt(NUM_SIMS), mu + 1.96 * std / np.sqrt(NUM_SIMS), alpha=0.3)
ax2.grid()
ax2.set_xscale('log')
ax2.set_yscale('log')
plt.show()