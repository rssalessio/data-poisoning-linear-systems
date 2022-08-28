import numpy as np
import scipy.signal as scipysig
from typing import Tuple
from pyzonotope import MatrixZonotope, Zonotope, concatenate_zonotope
from pydatadrivenreachability import compute_LTI_matrix_zonotope
import scipy.stats as scipystats
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

A = np.array([[0.7]])
B = np.array([[0.5]])

dim_x = 1
dim_u = 1


def collect_data(steps: int, std_u: float, std_w: float) -> Tuple[np.ndarray, np.ndarray]:
    U = np.zeros((dim_u, steps))
    X = np.zeros((dim_x, steps + 1))

    X[:, 0] = np.random.normal(size=(dim_x))

    for i in range(steps):
        U[:, i] = std_u * np.random.normal(size=(dim_u))
        X[:, i+1] = A @ X[:, i] +  np.squeeze(B * U[:, i]) + std_w * np.random.uniform(low=-0.1, high=0.1)#(size=(dim_x))

    return X.T, U.T

std_w = 1e-1
std_u = 1
samples = [15,100,1000]
k = 1
p = 2

F_vals = {}
p_vals = {}
N = 1000 
for id, sample_size in enumerate(samples):
    F_vals[sample_size] = []
    p_vals[sample_size] = []
    for n in range(N):
        X, U = collect_data(sample_size, std_u, std_w)
        AttackU = std_u * np.random.normal(size=U.shape)

        AB_noU = X[1:].T @ np.linalg.pinv(X[:-1].T)
        AB_poisoned = X[1:].T @ np.linalg.pinv(np.vstack([X[:-1].T, AttackU.T]))
        
        Y1 = AB_noU @ X[:-1].T
        Y2 = AB_poisoned @ np.vstack([X[:-1].T, AttackU.T])
        
        nu = sample_size - (p+1)
        R1 = np.power(X[1:] - Y1.T, 2).sum()
        R2 = np.power(X[1:] - Y2.T, 2).sum()
        F = ((R1-R2)/k)/(R2 / nu)
        F_vals[sample_size] .append(F)
        p_vals[sample_size] .append(1-scipystats.f.cdf(F, k, nu))


    
    print(f'T: {sample_size} - F: {np.mean(F_vals[sample_size] )} - P: {np.mean(p_vals[sample_size] )}')


sns.distplot(p_vals[15])
plt.show()

sns.distplot(p_vals[100])
plt.show()

sns.distplot(p_vals[1000])
plt.show()