import numpy as np
import scipy.signal as scipysig
from typing import Tuple
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

std_w = 1e-1
std_u = 1

X, U, W = collect_data(T, std_u, std_w, sys)
Xp, Xm = X[:, 1:], X[:, :-1]

D = np.vstack((Xm, U))
AB = Xp @ np.linalg.pinv(D)

true_residuals = Xp - AB @ D + np.random.uniform(low=-1, high =1, size=(dim_x, T))

#assert np.isclose(0, np.linalg.norm(true_residuals @ D.T)), "Error with LS"


def correlate(x: np.ndarray, num_lags: int):
    n, T = x.shape
    R = np.zeros((num_lags,n,n))

    for m in range(num_lags):
        for i in range(m, T):
            R[m] += x[:,i:i+1] @ x[:, i-m:i-m+1].T
        #R[m] /= (T-m)

    return R/T
num_lags = T//2
R = correlate(true_residuals,num_lags)
Z = 0
for i in range(num_lags):
    Z += R[i]@R[i].T

Z1 = Z  * T/(std_w**4)

print(num_lags)
print(Z1)
import pdb
pdb.set_trace()