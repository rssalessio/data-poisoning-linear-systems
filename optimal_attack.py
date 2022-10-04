import numpy as np
import scipy.signal as scipysig
from typing import Tuple
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import cvxpy as cp
import scipy.stats as stats
import dccp
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




NUM_SIMS = 2000

X, U, W = collect_data(T, std_u, std_w, sys)
Xp, Xm = X[:, 1:], X[:, :-1]
D = np.vstack((Xm, U))
AB = Xp @ np.linalg.pinv(D)

true_residuals = Xp - AB @ D 
test_variance = np.linalg.norm(true_residuals, 'fro')

DeltaX = cp.Variable((dim_x, T+1))
DeltaU = cp.Variable((dim_u, T))


objective = cp.norm(DeltaX[:, 1:] - AB @ cp.vstack((DeltaX[:, :-1], DeltaU)), 'fro')
objective += 2*cp.trace(true_residuals @ (DeltaX[:, 1:] - AB @ cp.vstack((DeltaX[:, :-1], DeltaU))).T)
constraints = [
    cp.norm(DeltaX, 'fro') <= 1e-3 * np.linalg.norm(X, 'fro'),
    cp.norm(DeltaU, 'fro') <= 1e-3 * np.linalg.norm(U, 'fro') 
]
problem = cp.Problem(cp.Maximize(objective), constraints)

print('Solving problem')
result = problem.solve(method='dccp', ccp_times=10, solver=cp.MOSEK)


print(f'Result: {result[0]}')


tildeXm = Xm + DeltaX.value[:, :-1]
tildeXp = Xp + DeltaX.value[:, 1:]
tildeU = U + DeltaU.value


Dtilde = np.vstack((tildeXm, tildeU))

ABtilde = tildeXp @ np.linalg.pinv(Dtilde)


poisoned_residuals = tildeXp - ABtilde @ Dtilde        
test_poisoned_variance = np.linalg.norm(poisoned_residuals, 'fro')
print(f'True residuals: {test_variance} - poisoned: {test_poisoned_variance}')


print(f'AB {AB}')
print(f'ABtilde {ABtilde}')
print(f'Difference {np.linalg.norm(AB-ABtilde)}')
import pdb
pdb.set_trace()