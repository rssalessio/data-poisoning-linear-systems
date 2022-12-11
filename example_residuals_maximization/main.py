# Simulate residuals attack
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import multiprocessing as mp
import cem
from typing import Tuple
from utils import collect_data, ResultsData


np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

# Define pulley system
dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
trueAB = np.hstack((sys.A, sys.B))
dim_x, dim_u = sys.B.shape

# Simulation parameters
T = 200
NUM_SIMS = 100
NUM_RANDOM_POINTS = 50
NUM_GAUSSIAN_RANDOM_POINTS = 100
NUM_CPUS = 8
std_w = 1e-1
std_u = 1

def evaluate_attack(true_residuals: np.ndarray, AB: np.ndarray, DeltaX: np.ndarray, DeltaU: np.ndarray) -> float:
    """Evaluate an attack

    Args:
        true_residuals (np.ndarray): true unpoisoned residuals
        AB (np.ndarray): Matrices (A,B) horizontally stacked
        DeltaX (np.ndarray): Poisoning on X
        DeltaU (np.ndarray): Poisoning on U

    Returns:
        float: attack evaluation
    """
    objective = np.linalg.norm(DeltaX[:, 1:] - AB @ np.vstack((DeltaX[:, :-1], DeltaU)), 'fro')
    objective += 2*np.trace(true_residuals @ (DeltaX[:, 1:] - AB @ np.vstack((DeltaX[:, :-1], DeltaU))).T)
    return objective


# Delta used by the attack
deltas = np.geomspace(1e-4, 1e-1, 50)


def run_simulation(id_sim: int):
    # Set seed according to simulation number)
    np.random.seed(id_sim)
    
    # Collect a set of data
    X, U, W = collect_data(T, std_u, std_w, sys)
    Xp, Xm = X[:, 1:], X[:, :-1]
    D = np.vstack((Xm, U))
    
    # Unpoisoned (A,B) and residuals
    AB = Xp @ np.linalg.pinv(D)
    true_residuals = Xp - AB @ D

    res = ResultsData(X, U, W, np.diag([std_w **2] * dim_x), trueAB, deltas)

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

        Xtilde = X+DeltaX.value
        Utilde = U + DeltaU.value
        Dtilde = np.vstack((Xtilde[:, :-1], Utilde))

        # Gaussian attack
        gauss_val = 0
        for _ in range(NUM_GAUSSIAN_RANDOM_POINTS):
            DeltaX = np.random.normal(size=X.shape) * (delta * np.linalg.norm(X, 'fro')) / np.sqrt((T+1)* dim_x)
            DeltaU = np.random.normal(size=U.shape) * (delta * np.linalg.norm(U, 'fro')) / np.sqrt(T* dim_u)
            val = evaluate_attack(true_residuals, AB, DeltaX, DeltaU)

            if val > gauss_val:
                res.gauss_DeltaX[idx_delta] = DeltaX
                res.gauss_DeltaU[idx_delta] = DeltaU
                gauss_val = val


        # CEM Attack
        num_p = np.prod(X.shape) + np.prod(U.shape)
        guassian_population = cem.GaussianPopulation(np.zeros(num_p), np.eye(num_p))

        def _obtain_attack(p: np.ndarray) -> float:
            DeltaX = p[: np.prod(X.shape)].reshape(X.shape)
            DeltaU = p[np.prod(X.shape):].reshape(U.shape)

            DeltaX = delta * np.linalg.norm(X, 'fro') * DeltaX / np.linalg.norm(DeltaX, 'fro')
            DeltaU = delta * np.linalg.norm(U, 'fro') * DeltaU / np.linalg.norm(DeltaU, 'fro')
            return DeltaX, DeltaU

        def _func(p: np.ndarray) -> float:
            DeltaX, DeltaU = _obtain_attack(p)
            return evaluate_attack(true_residuals, AB, DeltaX, DeltaU)

        for i in range(10):
            try:
                best_val, best_p = cem.optimize(_func, guassian_population, num_points=NUM_RANDOM_POINTS, max_iterations=500)
                DeltaX, DeltaU = _obtain_attack(best_p)
                break
            except Exception as e:
                print(f'Exception for{id_sim}/{idx_delta} - tentative {i}')
                DeltaX = np.zeros(X.shape)
                DeltaU = np.zeros(U.shape)
        res.cem_DeltaX[idx_delta] = DeltaX
        res.cem_DeltaU[idx_delta] = DeltaU

    return res

if __name__ == '__main__':
    # Execute simulations according to the number of CPUs and save data   
    with mp.Pool(NUM_CPUS) as p:
       results = p.map(run_simulation, [x for x in range(NUM_SIMS)])
    np.save('data/data.npy', results, allow_pickle=True)