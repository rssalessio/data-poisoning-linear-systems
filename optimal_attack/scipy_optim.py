from cmath import isnan
from functools import reduce
from importlib.metadata import requires
import torch
import numpy as np
import optuna
import scipy.signal as scipysig
import matplotlib.pyplot as plt
from optuna.trial import TrialState
from utils import collect_data, torch_correlate, pvalue_whiteness_test, correlate, pvalue_residuals_variance_test, TestStatistics
from typing import List, Optional, NamedTuple, Tuple
from data_structures import *
from modules import * 
from scipy.optimize import minimize,NonlinearConstraint
np.set_printoptions(formatter={'float_kind':"{:.4f}".format})
import time
from functorch import jacrev, jacfwd




def objective_function(x: np.ndarray, X, U, AB_unpoisoned, gradients=False):
    print(f'obj - {gradients}')
    torch.set_grad_enabled(gradients)
    dim_x, dim_u, T = X.shape[0], U.shape[0], U.shape[1]
    x = torch.tensor(x, requires_grad=gradients)
    DeltaX = x[:dim_x * (T+1)].reshape(dim_x, T+1)
    DeltaU = x[dim_x * (T+1):].reshape(dim_u, T)

    tildeX = torch.tensor(X) + DeltaX
    tildeU = torch.tensor(U) + DeltaU
    AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(torch.vstack((tildeX[:, :-1], tildeU)))
    DeltaAB = AB_poisoned - torch.tensor(AB_unpoisoned)
    loss= -torch.linalg.norm(DeltaAB, 2)

    if gradients is False:
        return loss.item()
    loss.backward()

    return x.grad.detach().numpy().flatten()



def _compute_constraints(x: torch.Tensor, X: np.ndarray, U: np.ndarray, AB_unpoisoned: np.ndarray, norm_unpoisoned_residuals, R, R0inv_unpoisoned, norm_unpoisoned_correlation_terms):
    dim_x, dim_u, T = X.shape[0], U.shape[0], U.shape[1]
    #x = torch.tensor(x, requires_grad=gradients)
    DeltaX = x[:dim_x * (T+1)].reshape(dim_x, T+1)
    DeltaU = x[dim_x * (T+1):].reshape(dim_u, T)

    tildeX = torch.tensor(X) + DeltaX
    tildeU = torch.tensor(U) + DeltaU
    D_poisoned = torch.vstack((tildeX[:, :-1], tildeU))
    AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(D_poisoned)

    residuals_poisoned = tildeX[:, 1:] - AB_poisoned @ D_poisoned

    R = torch_correlate(residuals_poisoned, s + 1)
    R0Inv = torch.linalg.inv(R[0])
    c_r = torch.linalg.norm(residuals_poisoned, 'fro') ** 2 / norm_unpoisoned_residuals

    c_c = []
    # Correlation constraints
    for tau in range(s):
        c_c.append(torch.linalg.norm(R[tau+1] @ R0Inv, 'fro') **2 / norm_unpoisoned_correlation_terms[tau])

    c_c = torch.vstack(c_c)
    # Build augmented lagrangian loss
    stacked_constraints = torch.vstack((
        torch.abs(1-c_r) - 0.1, 
        torch.abs(1-c_c) - 0.1)
        ).flatten()

    return -stacked_constraints

    # if gradients is False:
    #     return _compute(_x)
    # start = time.time()
    # grads = jacfwd(_compute)(_x).detach().numpy()
    # #grads = torch.autograd.functional.jacobian(_compute, _x, vectorize=True).detach().numpy()
    # print(f'Gradient computation {time.time()-start}')
    # import pdb
    # pdb.set_trace()
    # return grads

# from functorch.compile import aot_function
# grads_constraints = jacrev(_compute_constraints)



def compute_gradients(x: np.ndarray, X: np.ndarray, U: np.ndarray, AB_unpoisoned: np.ndarray, norm_unpoisoned_residuals, R, R0inv_unpoisoned, norm_unpoisoned_correlation_terms, gradients=True):
    print(gradients)
    torch.set_grad_enabled(gradients)
    x = torch.tensor(x, requires_grad=gradients)
    if gradients is True:
        start = time.time()
        grads = torch.autograd.functional.jacobian(lambda y: _compute_constraints(y, X, U, AB_unpoisoned, norm_unpoisoned_residuals, R, R0inv_unpoisoned, norm_unpoisoned_correlation_terms), x, vectorize=True).detach().numpy()
        #grads= grads_constraints(x, X, U, AB_unpoisoned, norm_unpoisoned_residuals, R, R0inv_unpoisoned, norm_unpoisoned_correlation_terms).detach().numpy()
        print(f'Gradient computation {time.time()-start}')
        return grads
    return _compute_constraints(x, X, U, AB_unpoisoned, norm_unpoisoned_residuals, R, R0inv_unpoisoned, norm_unpoisoned_correlation_terms)

if __name__ == '__main__':
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
    
    

    #constraints = NonlinearConstraint(compute_constraints, -np.infty * np.ones(s+1), np.zeros(s+1), jac=grad_compute_constraints, keep_feasible=True)
    constraints = [{
            'type': 'ineq',
            'fun': lambda *args: compute_gradients(*args, gradients=False),
            'jac': lambda *args: compute_gradients(*args, gradients=True),
            'args': (X, U, AB_unpoisoned, norm_unpoisoned_residuals, R, R0inv_unpoisoned, norm_unpoisoned_correlation_terms)
    }]


    

    res = minimize(
        fun= objective_function,
        x0=np.zeros(dim_x * (T+1) + dim_u * T),
        args=(X, U, AB_unpoisoned),
        bounds=[(-0.3,0.3) for i in range(dim_x * (T+1) + dim_u * T)],
        jac= lambda *args: objective_function(*args, gradients=True),
        method = 'SLSQP',
        constraints=constraints,
        options={'disp': True, 'maxiter': 100})
    print(res)
    import pdb
    pdb.set_trace()

    