import numpy as np
import scipy.signal as scipysig
from typing import Tuple, NamedTuple
import cvxpy as cp
import multiprocessing as mp
import dccp
from utils import collect_data, ResultsSimulation


np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
trueAB = np.hstack((sys.A, sys.B))
dim_x, dim_u = sys.B.shape
T = 100
NUM_SIMS = 100
NUM_RANDOM_POINTS =  100
NUM_CPUS = 12

std_w = 1e-1
std_u = 1

def evaluate_attack(true_residuals: np.ndarray, AB: np.ndarray, DeltaX: np.ndarray, DeltaU: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    objective = np.linalg.norm(DeltaX[:, 1:] - AB @ np.vstack((DeltaX[:, :-1], DeltaU)), 'fro')
    objective += 2*np.trace(true_residuals @ (DeltaX[:, 1:] - AB @ np.vstack((DeltaX[:, :-1], DeltaU))).T)
    return objective



deltas = np.geomspace(1e-4, 1e-1, 50)


def run_simulation(id_sim: int):
    np.random.seed(id_sim)
    X, U, W = collect_data(T, std_u, std_w, sys)
    Xp, Xm = X[:, 1:], X[:, :-1]
    D = np.vstack((Xm, U))
    AB = Xp @ np.linalg.pinv(D)

    true_residuals = Xp - AB @ D

    res = ResultsSimulation(X, U, W, np.diag([std_w **2] * dim_x), trueAB, deltas, len(deltas))

    for idx_delta, delta in enumerate(deltas):
        print(f'Running {id_sim}/{idx_delta}')
        # Optimal attack
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

        res.opt_DeltaX[idx_delta] = DeltaX.value
        res.opt_DeltaU[idx_delta] = DeltaU.value

        # Gaussian attack
        gauss_val = 0
        unif_val = 0
        for idx_sample in range(NUM_RANDOM_POINTS):
            DeltaX = np.random.normal(size=X.shape) * (delta * np.linalg.norm(X, 'fro')) / np.sqrt((T+1)* dim_x)
            DeltaU = np.random.normal(size=U.shape) * (delta * np.linalg.norm(U, 'fro')) / np.sqrt(T* dim_u)
            val = evaluate_attack(true_residuals, AB, DeltaX, DeltaU)

            if val > gauss_val:
                res.gauss_DeltaX[idx_delta] = DeltaX
                res.gauss_DeltaU[idx_delta] = DeltaU
                gauss_val = val

            # Uniform attack
            alpha = 0.5 * (12 ** 0.5)
            DeltaX, DeltaU = np.random.uniform(low=-alpha , high=alpha, size=X.shape), np.random.uniform(low=-alpha, high=alpha, size=U.shape)
            DeltaX = DeltaX * (delta * np.linalg.norm(X, 'fro')) / np.linalg.norm(DeltaX, 'fro')
            DeltaU = DeltaU * (delta * np.linalg.norm(U, 'fro')) / np.linalg.norm(DeltaU, 'fro')

            val = evaluate_attack(true_residuals, AB, DeltaX, DeltaU)

            if val > unif_val:
                res.unif_DeltaX[idx_delta] = DeltaX
                res.unif_DeltaU[idx_delta] = DeltaU
                unif_val = val

    return res

if __name__ == '__main__':
    with mp.Pool(NUM_CPUS) as p:
        results = p.map(run_simulation, [x for x in range(NUM_SIMS)])

    np.save('data.npy', results, allow_pickle=True)