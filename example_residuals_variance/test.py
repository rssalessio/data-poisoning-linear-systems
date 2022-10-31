import numpy as np
import scipy.signal as scipysig
from typing import Tuple, NamedTuple
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import cvxpy as cp
import scipy.stats as stats
import multiprocessing as mp
import dccp
from scipy.stats import chi2, norm
from momentchi2 import hbe
from utils import collect_data, residuals_variance_test, TestStatistics, confidence_interval_signal, ResultsData


np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
trueAB = np.hstack((sys.A, sys.B))
dim_x, dim_u = sys.B.shape
T = 200
NUM_SIMS = 1
NUM_CPUS = 12

std_w = 1e-1
std_u = 1


deltas = [1e-1]#np.geomspace(1e-4, 1e-1, 50)

true_delta_ab_norm = np.zeros((NUM_SIMS))
true_residuals_variance = np.zeros((NUM_SIMS))

poisoned_delta_ab_norm_optimal_atk = np.zeros((NUM_SIMS,len(deltas)))
poisoned_delta_ab_norm_gaussian_atk = np.zeros((NUM_SIMS,len(deltas)))
poisoned_residuals_variance_optimal_atk = np.zeros((NUM_SIMS,len(deltas)))
poisoned_residuals_variance_gaussian_atk = np.zeros((NUM_SIMS,len(deltas)))

tests = []




def evaluate_attack(Xm: np.ndarray, Xp: np.ndarray, U: np.ndarray, DeltaX: np.ndarray, DeltaU: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tildeXm = Xm + DeltaX[:, :-1]
    tildeXp = Xp + DeltaX[:, 1:]
    tildeU = U + DeltaU
    Dtilde = np.vstack((tildeXm, tildeU))

    ABtilde = tildeXp @ np.linalg.pinv(Dtilde)

    residuals = tildeXp - ABtilde @ Dtilde        

    return ABtilde, residuals

def run_simulation(id_sim: int):
    np.random.seed(id_sim)
    res = ResultsData(len(deltas))
    X, U, W = collect_data(T, std_u, std_w, sys)
    Xp, Xm = X[:, 1:], X[:, :-1]
    D = np.vstack((Xm, U))
    AB = Xp @ np.linalg.pinv(D)

    true_residuals = Xp - AB @ D
    res.true_residuals_variance = residuals_variance_test(true_residuals, dim_u, [std_w ** 2] * dim_x).p_right
    res.true_delta_ab_norm = np.linalg.norm(AB - trueAB, 2)

    Hbar_unpoisoned = np.kron(np.hstack((Xm.T, U.T)), np.eye(dim_x))
    H_unpoisoned = np.linalg.inv(Hbar_unpoisoned.T @ Hbar_unpoisoned) @ Hbar_unpoisoned.T
    SigmaW = np.diag([std_w ** 2] * dim_x)

    SigmaAB = H_unpoisoned @ np.kron(SigmaW,np.eye(200)) @ H_unpoisoned.T
    print(np.linalg.norm(SigmaAB,2))


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

        ABtilde, poisoned_residuals = evaluate_attack(Xm, Xp, U, DeltaX.value, DeltaU.value)
        res.poisoned_delta_ab_norm_optimal_atk[idx_delta] = np.linalg.norm(ABtilde - trueAB, 2)
        res.poisoned_residuals_variance_optimal_atk[idx_delta] = residuals_variance_test(poisoned_residuals, dim_u, [std_w ** 2] * dim_x).p_right

        
        L1 = []
        L2 = []
        for j in range(20):
            L = T//5
            idx = np.random.choice(T-L-1)
            _AB, _ = evaluate_attack(
                Xm[:,idx:idx+L], Xp[:, idx:idx+L], U[:, idx:idx+L], 0*DeltaX.value[:, idx:idx+L+1], 0*DeltaU.value[:, idx:idx+L])
            _ABtilde, poisoned_residuals = evaluate_attack(
                Xm[:,idx:idx+L], Xp[:, idx:idx+L], U[:, idx:idx+L], DeltaX.value[:, idx:idx+L+1], DeltaU.value[:, idx:idx+L])
            
            print(f'{AB} - {ABtilde}')
            L1.append(_AB)
            L2.append(_ABtilde)
        # Hbar_poisoned = np.kron(np.hsthttps://jobs.ericsson.com/job/Stockholm-Security-Product-Management-Team-Stoc/756572502/ack((Xm.T + (DeltaX.value[:,:-1]).T, U.T + DeltaU.value.T)), np.eye(dim_x))
        # H_poisoned = np.linalg.inv(Hbar_poisoned.T @ Hbar_poisoned) @ Hbar_poisoned.T
        # SigmaAB_poisoned = H_poisoned @ np.kron(SigmaW,np.eye(200)) @ H_poisoned.T
        # print(np.linalg.norm(SigmaAB_poisoned,2))
            
    
    print(AB)
    print(trueAB)
    L1 = np.array(L1).mean(0)
    L2 = np.array(L2).mean(0)
    import pdb
    pdb.set_trace()
    return res

run_simulation(0)