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
        delta0: float = 1e-1,
        delta1: float = 1e-1,
        max_iterations: int = 10000,
        learning_rate: float = 1e-2,
        learning_rate_regularizer: float = 1e-1,
        max_grad_norm: float = 1e-1,
        lagrange_regularizer: float = 1e-2,
        penalty_regularizer: float = 1e-1,
        rel_tol: float = 1e-4,
        beta: float = 1.005,
        regularizers: List[ConstraintModule] = [],
        verbose: bool = True,
        trial: optuna.trial.Trial = None):
    X = torch.tensor(unpoisoned_data.X)
    U = torch.tensor(unpoisoned_data.U)
    AB_unpoisoned = torch.tensor(unpoisoned_data.AB_unpoisoned)
    A_unpoisoned2 = torch.tensor(X[:, 1:] @ np.linalg.pinv(X[:, :-1]))
    s = unpoisoned_data.num_lags * 2

    U_norm = np.linalg.norm(unpoisoned_data.U, 'fro')
    X_norm = np.linalg.norm(unpoisoned_data.X, 'fro')

    residuals_unpoisoned_2 = X[:, 1:] - A_unpoisoned2 @ X[:, :-1]
    Zd = -1 + (np.linalg.norm(residuals_unpoisoned_2,'fro') / (np.linalg.norm(unpoisoned_data.residuals_unpoisoned, 'fro')))**2

    # Poisoning signals
    DeltaX = torch.tensor(np.zeros((unpoisoned_data.dim_x, unpoisoned_data.T+1)), requires_grad=True)
    DeltaU = torch.tensor(np.zeros((unpoisoned_data.dim_u, unpoisoned_data.T)), requires_grad=True)

    # Optimizer
    optimizer = torch.optim.Adam([DeltaX, DeltaU], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95, verbose=True)

    # Lagrangian
    lmbd = torch.tensor(lagrange_regularizer * np.ones(s+4), requires_grad=False)
    rho = penalty_regularizer

    # Attack information
    info = AttackOptimizationInfo(unpoisoned_data)
    losses = []
    # Other info
    R0inv_unpoisoned = np.linalg.inv(unpoisoned_data.R[0])
    norm_unpoisoned_residuals = np.linalg.norm(unpoisoned_data.residuals_unpoisoned, 'fro') ** 2
    norm_unpoisoned_correlation_terms = [np.linalg.norm(unpoisoned_data.R[idx+1] @ R0inv_unpoisoned, 'fro') ** 2 for idx in range(s)]


    clamp_scheduler_value = 0.25

    # Attack training loop
    for iteration in range(max_iterations):
        # Poisoned data
        tildeX = X + DeltaX
        tildeU = U + DeltaU
        D_poisoned = torch.vstack((tildeX[:, :-1], tildeU))
        AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(D_poisoned)
        A_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(tildeX[:, :-1])

        residuals_poisoned = tildeX[:, 1:] - AB_poisoned @ D_poisoned
        residuals_poisoned_2 = tildeX[:, 1:] - A_poisoned @ tildeX[:, :-1]
        DeltaAB = AB_poisoned - AB_unpoisoned


        Zdtilde = -1 + (torch.linalg.norm(residuals_poisoned_2,'fro') / (torch.linalg.norm(residuals_poisoned, 'fro')))**2
        R = torch_correlate(residuals_poisoned, s + 1)
        R0Inv = torch.linalg.inv(R[0])
        attack_data = AttackData(unpoisoned_data, DeltaU, DeltaX, D_poisoned, AB_poisoned, residuals_poisoned, DeltaAB, R)
        main_objective = torch.norm(DeltaAB, p=2)


        # Residuals norm constraint
        c_r = torch.linalg.norm(residuals_poisoned, 'fro') ** 2 / norm_unpoisoned_residuals
        c_z = Zdtilde/Zd

        c_u = torch.linalg.norm(DeltaU,'fro')/ U_norm
        c_x = torch.linalg.norm(DeltaX, 'fro')/ X_norm
        c_c = []

        # Correlation constraints
        for tau in range(s):
            c_c.append(torch.linalg.norm(R[tau+1] @ R0Inv, 'fro') **2 / norm_unpoisoned_correlation_terms[tau])

        c_c = torch.vstack(c_c)
        # Build augmented lagrangian loss
        stacked_constraints = torch.vstack((
            torch.abs(1-c_r) - delta0,
            torch.abs(1-c_z) - delta0**2,
            c_x - delta0,
            c_u - delta0,
            torch.abs(1-c_c) - delta1)
            ).flatten()
        clamped_constraints = torch.clamp(stacked_constraints, min=0)
        loss = -main_objective + torch.dot(lmbd, clamped_constraints) + (rho/2) * torch.square(torch.linalg.norm(clamped_constraints, 2))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_([DeltaX, DeltaU], max_norm=1)
        optimizer.step()
        losses.append(main_objective.item())


        with torch.no_grad():
            if clamped_constraints.max().item() > max(delta0,delta1)//2:
                rho = min(1e7, beta * rho)
            lmbd = lmbd + rho * clamped_constraints
            
            # if clamped_constraints.mean() > 0:
            #     clamp_scheduler_value = max(0, clamp_scheduler_value - 1e-2 * clamped_constraints.mean().item())


            if clamped_constraints.mean() > 0.05:
                scheduler.step()
                
            if iteration % 300 == 0:
                print(f'[{iteration}] Loss: {loss.item()}  - norm ABDelta : {main_objective.item()} - {clamped_constraints.detach().numpy().flatten()} - lmbd {lmbd.detach().numpy()} - rho { rho}')


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
    return DeltaX.detach().numpy(), DeltaU.detach().numpy(), losses

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

    DeltaX, DeltaU, losses = compute_attack(unpoisoned_data, max_iterations=10000)
    tildeX = X + DeltaX
    tildeU = U + DeltaU
    D_poisoned = np.vstack((tildeX[:, :-1], tildeU))
    AB_poisoned = tildeX[:, 1:] @ np.linalg.pinv(D_poisoned)
    import pdb
    pdb.set_trace()

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