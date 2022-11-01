from cmath import isnan
from functools import reduce
import torch
import numpy as np
import optuna
import scipy.signal as scipysig
import matplotlib.pyplot as plt
from optuna.trial import TrialState
from utils import collect_data, torch_correlate, pvalue_whiteness_test, correlate, pvalue_residuals_variance_test, TestStatistics
from typing import List, Optional, NamedTuple
from data_structures import *
from modules import * 
np.set_printoptions(formatter={'float_kind':"{:.4f}".format})


def log(epoch: int, info: AttackOptimizationInfo) -> None:
    SPACING = 30
    print(f"---------- Epoch {epoch} ----------")
    print(f"{'Eval':<40} {'Statistics'}")
    print(f"    {'loss:' + f'{info.loss[-1]:.4f}':<40} whiteness_unp: {info.unpoisoned_data.whiteness_statistics_test_unpoisoned.statistics:.4f}")
    print(f"    {'delta_norm:' +f'{info.delta_norm[-1]:.4f}':<40} whiteness_pos: {info.whiteness_statistics_test_poisoned[-1].statistics:.4f}")
    print(f"    {'residuals_norm:' f'{info.residuals_norm[-1]:.4f}':<40} res_var_unpos: {info.unpoisoned_data.residuals_variance_test_unpoisoned.statistics:.4f}")
    print(f"    {'regularizer:' f'{info.regularizer[-1]:.4f}':<40} res_var_pos: {info.residuals_variance_test_poisoned[-1].statistics:.4f}")



def compute_attack(
        unpoisoned_data: UnpoisonedDataInfo,
        delta0: float = 0.1,
        delta1: float = 0.1,
        max_iterations: int = 1000,
        learning_rate: float = 1e-3,
        learning_rate_regularizer: float = 1e-1,
        max_grad_norm: float = 1e-1,
        lagrange_regularizer: float = 1e-2,
        penalty_regularizer: float = 1e-1,
        rel_tol: float = 1e-4,
        beta: float = 1.01,
        regularizers: List[ConstraintModule] = [],
        verbose: bool = True,
        trial: optuna.trial.Trial = None):
    X = torch.tensor(unpoisoned_data.X)
    U = torch.tensor(unpoisoned_data.U)
    AB_unpoisoned = torch.tensor(unpoisoned_data.AB_unpoisoned)
    s = unpoisoned_data.num_lags * 2

    # Poisoning signals
    DeltaX = torch.tensor(np.zeros((unpoisoned_data.dim_x, unpoisoned_data.T+1)), requires_grad=True)
    DeltaU = torch.tensor(np.zeros((unpoisoned_data.dim_u, unpoisoned_data.T)), requires_grad=True)

    # Optimizer
    optimizer = torch.optim.Adam([DeltaX, DeltaU], lr=learning_rate)

    # Lagrangian
    lmbd = torch.tensor(lagrange_regularizer * np.ones(s+1), requires_grad=False)
    rho = penalty_regularizer

    # Attack information
    info = AttackOptimizationInfo(unpoisoned_data)

    # Other info
    R0inv_unpoisoned = np.linalg.inv(unpoisoned_data.R[0])
    norm_unpoisoned_residuals = np.linalg.norm(unpoisoned_data.residuals_unpoisoned, 'fro') ** 2
    norm_unpoisoned_correlation_terms = [np.linalg.norm(unpoisoned_data.R[idx+1] @ R0inv_unpoisoned, 'fro') ** 2 for idx in range(s)]

    # Attack training loop
    for iteration in range(max_iterations):
        # Poisoned data
        tildeX = X + DeltaX
        tildeU = U + DeltaU
        D_poisoned = torch.vstack((tildeX[:, :-1], tildeU))
        AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(D_poisoned)

        residuals_poisoned = tildeX[:, 1:] - AB_poisoned @ D_poisoned
        DeltaAB = AB_poisoned - AB_unpoisoned

        R = torch_correlate(residuals_poisoned, s + 1)
        R0Inv = torch.linalg.inv(R[0])
        attack_data = AttackData(unpoisoned_data, DeltaU, DeltaX, D_poisoned, AB_poisoned, residuals_poisoned, DeltaAB, R)
        main_objective = torch.norm(DeltaAB, p=2)

        # Residuals norm constraint
        c_r = torch.linalg.norm(residuals_poisoned, 'fro') ** 2 / norm_unpoisoned_residuals

        c_c = []

        # Correlation constraints
        for tau in range(s):
            c_c.append(torch.linalg.norm(R[tau+1] @ R0Inv, 'fro') **2 / norm_unpoisoned_correlation_terms[tau])

        c_c = torch.vstack(c_c)
        # Build augmented lagrangian loss
        stacked_constraints = torch.vstack((
            torch.abs(1-c_r) - delta0, 
            torch.abs(1-c_c) - delta1)
            ).flatten()
        clamped_constraints = torch.clamp(stacked_constraints, min=0)
        loss = -main_objective + torch.dot(lmbd, clamped_constraints) + (rho/2) * torch.square(torch.linalg.norm(clamped_constraints, 2))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_([DeltaX, DeltaU], max_norm=1)
        optimizer.step()



        with torch.no_grad():
            if clamped_constraints.max().item() > max(delta0,delta1)//2:
                rho = beta * rho
            lmbd = lmbd + rho * clamped_constraints
            print(f'Loss: {loss.item()}  - norm ABDelta : {main_objective.item()} - constraints: {c_r.item()} - {c_c.detach().numpy().flatten()} - lmbd {lmbd.detach().numpy()} - rho { rho}')


        # info.loss.append(loss.item())
        # info.regularizer.append(regularizer.item())
        # info.residuals_norm.append(np.linalg.norm(residuals_poisoned.detach().numpy(), 'fro'))
        # info.delta_norm.append(np.linalg.norm(DeltaAB.detach().numpy(), 2))
        # info.AB_poisoned.append(AB_poisoned.detach().numpy())
        # info.DeltaX.append(DeltaX.detach().numpy())
        # info.DeltaU.append(DeltaU.detach().numpy())
        # info.residuals_poisoned.append(residuals_poisoned.detach().numpy())
        
        # with torch.no_grad():
        #     _R = torch.stack(R).detach().numpy()
        #     info.whiteness_statistics_test_poisoned.append(pvalue_whiteness_test(_R, self.num_lags, self.T))
        #     info.residuals_variance_test_poisoned.append(
        #         pvalue_residuals_variance_test(
        #             residuals_poisoned.detach().numpy(), self.dim_u, self.unpoisoned_data.eigs_SigmaW))

        # if verbose:
        #     log(iteration, info)

        # if len(info.loss) > 2:
        #     if np.abs(info.loss[-1] - info.loss[-2]) / np.abs(info.loss[-2]) < rel_tol:
        #         print('Optimization complete')
        #         break
    return DeltaX.detach().numpy(), DeltaU.detach().numpy()

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

    D = np.vstack((X[:, :-1], U))
    AB_unpoisoned = X[:, 1:] @ np.linalg.pinv(D)
    residuals_unpoisoned = X[:, 1:] - AB_unpoisoned @ D
    R_unpoisoned = correlate(residuals_unpoisoned, T-dim_x-dim_u)
    SigmaW = np.diag([STD_W ** 2] * X.shape[0])

    unpoisoned_data = UnpoisonedDataInfo(
        X, U, R_unpoisoned, num_lags, SigmaW,
        np.linalg.eigvalsh(SigmaW).tolist(),
        AB_unpoisoned, residuals_unpoisoned, pvalue_whiteness_test(R_unpoisoned, num_lags, T),
        pvalue_residuals_variance_test(residuals_unpoisoned, dim_u, np.linalg.eigvalsh(SigmaW).tolist())
    )

    DeltaX, DeltaU = compute_attack(unpoisoned_data)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, dim_x +dim_u)
    for i in range(dim_x):
        ax[i].plot(X[i], label='Original')
        ax[i].plot(X[i] + DeltaX[0], label='poisoned')
        ax[0].grid()
    ax[-1].plot(U[0], label='original')
    ax[-1].plot(U[0] + DeltaU[0], label='poisoned')
    plt.legend()
    ax[-1].grid()
    plt.show()