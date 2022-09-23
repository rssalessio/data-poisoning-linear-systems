from random import sample
import numpy as np
import scipy.signal as scipysig
from typing import Tuple
import scipy.stats as scipystats
import matplotlib.pyplot as plt

from scipy.linalg import null_space
np.set_printoptions(formatter={'float_kind':"{:.2f}".format})
A = np.array([[0.7, 0], [0.1, 0.3]])
B = np.array([[0.5], [0.1]])

dim_x = 2
dim_u = 1


def collect_data(steps: int, std_u: float, std_w: float) -> Tuple[np.ndarray, np.ndarray]:
    U = np.zeros((dim_u, steps))
    X = np.zeros((dim_x, steps + 1))
    W = np.zeros((dim_x, steps))
    X[:, 0] = np.random.normal(size=(dim_x))

    for i in range(steps):
        U[:, i] = std_u * np.random.normal(size=(dim_u))
        W[:, i] = std_w * np.random.normal(size=(dim_x))
        X[:, i+1] = A @ X[:, i] +  np.squeeze(B * U[:, i]) + W[:, i]#(low=-0.1, high=0.1)#(size=(dim_x))

    return X, U, W

std_w = 1e-1
std_u = 1
sample_size = 1000


X, U, W = collect_data(sample_size, std_u, std_w)

attack_amplitude = 1
attack_X = 0*np.random.uniform(low=-attack_amplitude, high=attack_amplitude, size=(dim_x, sample_size+1))
attack_U = np.random.uniform(low=0*-attack_amplitude, high=attack_amplitude, size=(dim_u, sample_size))



Xtilde = X + attack_X
Utilde = U + attack_U

D = np.vstack([X[:,:-1], U])
Dtilde = np.vstack([Xtilde[:,:-1], Utilde])

AB = X[:,1:] @ np.linalg.pinv(D)
ABtilde = Xtilde[:,1:] @ np.linalg.pinv(Dtilde)
Delta = -B @ attack_U @ np.linalg.pinv(Dtilde)


resid =   W @ (np.eye(sample_size) - np.linalg.pinv(D) @ D)
eigs = np.abs(np.linalg.eigvals(np.linalg.pinv(D) @ D))
print(np.sort(eigs)[::-1][:20])
plt.plot(eigs)
plt.show()
plt.plot(resid[0,:])
plt.plot(W[0,:])
plt.show()

Wtilde = W + np.hstack((np.eye(dim_x), -A, -B)) @ np.vstack([attack_X[:, 1:], attack_X[:, :-1], attack_U])

print(ABtilde - np.hstack((A,B)))
print(Wtilde @ np.linalg.pinv(Dtilde))
print(Delta)
exit(-1)

null_basis = null_space(Dtilde)


Wtilde = np.hstack((np.eye(dim_x), -A, -B)) @ np.vstack((Xtilde[:, 1:], Dtilde))
Wtilde2 = W + np.hstack((np.eye(dim_x), -A, -B)) @ np.vstack((attack_X[:, 1:], attack_X[:,:-1], attack_U))
resid = np.linalg.norm((Xtilde[:, 1:] - Wtilde) @ null_basis)
print(np.linalg.norm(Wtilde -Wtilde2))
import pdb
pdb.set_trace()