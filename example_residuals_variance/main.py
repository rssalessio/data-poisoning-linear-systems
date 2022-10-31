import numpy as np
import scipy.signal as scipysig
from typing import Tuple, NamedTuple
import cvxpy as cp
import multiprocessing as mp
import dccp
from .utils import collect_data, ResultsSimulation


np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
trueAB = np.hstack((sys.A, sys.B))
dim_x, dim_u = sys.B.shape
T = 200
NUM_SIMS = 12
NUM_CPUS = 4

std_w = 1e-1
std_u = 1


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
        res.gauss_DeltaX[idx_delta] = np.random.normal(size=X.shape) * (delta * np.linalg.norm(X, 'fro')) / np.sqrt(T* dim_x)
        res.gauss_DeltaU[idx_delta] = np.random.normal(size=U.shape) * (delta * np.linalg.norm(U, 'fro')) / np.sqrt(T* dim_u)

        # Uniform attack
        DeltaX, DeltaU = np.random.uniform(low=-1, high=-1, size=X.shape), np.random.uniform(low=-1, high=-1, size=U.shape)
        res.unif_DeltaX[idx_delta] = DeltaX * (delta * np.linalg.norm(X, 'fro')) / np.linalg.norm(DeltaX, 'fro')
        res.unif_DeltaU[idx_delta] = DeltaU * (delta * np.linalg.norm(U, 'fro')) / np.linalg.norm(DeltaU, 'fro')

    return res

if __name__ == '__main__':
    with mp.Pool(NUM_CPUS) as p:
        results = p.map(run_simulation, [x for x in range(NUM_SIMS)])

    np.save('data.npy', results, allow_pickle=True)