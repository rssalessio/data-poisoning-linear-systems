import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax

import numpy as np
import scipy.signal as scipysig
from utils import collect_data, torch_correlate, pvalue_whiteness_test, correlate, pvalue_residuals_variance_test, TestStatistics, jax_correlate
from typing import List, Optional, NamedTuple, Tuple
import torch
import time

def objective_function(x: jnp.ndarray, X: jnp.ndarray, U: jnp.ndarray, AB_unpoisoned: jnp.ndarray):
    dim_x, dim_u, T = X.shape[0], U.shape[0], U.shape[1]
    DeltaX = x[:dim_x * (T+1)].reshape(dim_x, T+1)
    DeltaU = x[dim_x * (T+1):].reshape(dim_u, T)

    tildeX = X + DeltaX
    tildeU = U + DeltaU
    AB_poisoned = tildeX[:, 1:] @ jnp.linalg.pinv(jnp.vstack((tildeX[:, :-1], tildeU)))
    DeltaAB = AB_poisoned - AB_unpoisoned
    loss = jnp.linalg.norm(DeltaAB, 2)
    return -loss
def grad_objective_function(x: np.ndarray, X, U, AB_unpoisoned):
    dim_x, dim_u, T = X.shape[0], U.shape[0], U.shape[1]
    x = torch.tensor(x, requires_grad=True)
    DeltaX = x[:dim_x * (T+1)].reshape(dim_x, T+1)
    DeltaU = x[dim_x * (T+1):].reshape(dim_u, T)

    tildeX = torch.tensor(X) + DeltaX
    tildeU = torch.tensor(U) + DeltaU
    AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(torch.vstack((tildeX[:, :-1], tildeU)))
    DeltaAB = AB_poisoned - torch.tensor(AB_unpoisoned)
    loss= -torch.linalg.norm(DeltaAB, 2)
    loss.backward()

    return x.grad.detach().numpy().flatten()


def compute_constraints(x: jnp.ndarray, X: jnp.ndarray, U: jnp.ndarray, 
        AB_unpoisoned: jnp.ndarray, norm_unpoisoned_residuals, R, R0inv_unpoisoned, norm_unpoisoned_correlation_terms):
    dim_x, dim_u, T = X.shape[0], U.shape[0], U.shape[1]
    DeltaX = x[:dim_x * (T+1)].reshape(dim_x, T+1)
    DeltaU = x[dim_x * (T+1):].reshape(dim_u, T)
    tildeX = X + DeltaX
    tildeU = U + DeltaU
    D_poisoned = jnp.vstack((tildeX[:, :-1], tildeU))
    AB_poisoned = tildeX[:, 1:] @ jnp.linalg.pinv(D_poisoned)

    residuals_poisoned = tildeX[:, 1:] - AB_poisoned @ D_poisoned

    s=24


    R = jax_correlate(residuals_poisoned, s + 1)
    R0Inv = jnp.linalg.inv(R[0])
    c_r = jnp.linalg.norm(residuals_poisoned, 'fro') ** 2 / norm_unpoisoned_residuals

    c_c = []
        # Correlation constraints
    for tau in range(s):
        c_c.append(jnp.linalg.norm(R[tau+1] @ R0Inv, 'fro') **2 / norm_unpoisoned_correlation_terms[tau])

    c_c = jnp.vstack(c_c)
    # Build augmented lagrangian loss
    stacked_constraints = jnp.vstack((
        jnp.abs(1-c_r) - 0.1, 
        jnp.abs(1-c_c) - 0.1)
        ).flatten()
    return -stacked_constraints
# key = random.PRNGKey(0)
# def selu(x, alpha=1.67, lmbda=1.05):
#   return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

# x = random.normal(key, (1000000,))
# selu(x).block_until_ready()
# selu_jit = jit(selu)
# selu_jit(x).block_until_ready()


dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
dim_x, dim_u = sys.B.shape


NUM_SIMS = 10
T = 500
STD_U = 1
STD_W = 0.1
X, U, W = collect_data(T, STD_U, STD_W, sys)

dim_x = X.shape[0]
dim_u, T=  U.shape
num_lags = int(T * 0.025)

s = 2 * num_lags

D = np.vstack((X[:, :-1], U))
AB_unpoisoned = X[:, 1:] @ np.linalg.pinv(D)
residuals_unpoisoned = X[:, 1:] - AB_unpoisoned @ D
R_unpoisoned = correlate(residuals_unpoisoned, T-dim_x-dim_u)
SigmaW = np.diag([STD_W ** 2] * X.shape[0])
norm_unpoisoned_residuals = np.linalg.norm(residuals_unpoisoned, 'fro') ** 2

R = correlate(residuals_unpoisoned, s+1)
R0inv_unpoisoned = np.linalg.inv(R[0])
norm_unpoisoned_correlation_terms = [np.linalg.norm(R[idx+1] @ R0inv_unpoisoned, 'fro') ** 2 for idx in range(s)]
    
jit_obj_fun = jax.jit(objective_function)
grads_obj_fun = jax.jit(jax.grad(jit_obj_fun))

print('Starting')
start=time.time()

grad_constraints = jax.jit(jax.jacobian(compute_constraints))
print(f'odne {time.time()-start}')

input = np.zeros(dim_x * (T+1) + dim_u * T)
print('Starting')
start=time.time()

res  = grad_constraints(input, X, U, AB_unpoisoned, norm_unpoisoned_residuals, R, R0inv_unpoisoned, norm_unpoisoned_correlation_terms)
print(f'odne {time.time()-start}')

print(res.shape)
print('Starting')
start=time.time()
res  = grad_constraints(input, X, U, AB_unpoisoned, norm_unpoisoned_residuals, R, R0inv_unpoisoned, norm_unpoisoned_correlation_terms)
print(f'odne {time.time()-start}')

print(res.shape)
times = []
for _ in range(1000):
    start_time = time.time()
    grads_obj_fun(input,
    X, U, AB_unpoisoned)
    end_time = time.time() - start_time
    times.append(end_time)

print(times[0])
print(times[1])
print(f'Jax - Mean: {np.mean(times)} - std: {np.std(times)}')


times = []
for _ in range(1000):
    start_time = time.time()
    grad_objective_function(input,
        X, U, AB_unpoisoned)
    end_time = time.time() - start_time
    times.append(end_time)
print(times[0])
print(times[1])
print(f'Torch - Mean: {np.mean(times)} - std: {np.std(times)}')
