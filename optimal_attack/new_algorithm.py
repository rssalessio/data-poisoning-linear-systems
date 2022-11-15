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

class Detector(torch.nn.Module):
    def __init__(self, batch_size: int, dim: int, hidden: int = 256, lr: float = 5e-4):
        super().__init__()
        self.network = torch.nn.Sequential(*[
            torch.nn.Linear(batch_size * dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
            torch.nn.ReLU()
        ])

        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probability = torch.clip((-self.network(x)), max=1e9).exp()
        return torch.clip(probability, 1e-12, 1-1e-12)

    def train(self, epochs: int, unpoisoned_residuals: torch.Tensor, poisoned_residuals: torch.Tensor):

        dim_x, T = unpoisoned_residuals.shape
        for epoch in range(epochs):
            idxs = np.arange(T-self.batch_size+1)#np.random.choice(T - self.batch_size, size=T//10, replace=False)
            np.random.shuffle(idxs)
            loss_d_pois = []
            loss_d_unpois = []
            for idx in idxs:
                _pos = poisoned_residuals[:,idx:idx+self.batch_size].T.flatten().unsqueeze(0)
                _unpos = poisoned_residuals[:, idx: idx+self.batch_size].T.flatten().unsqueeze(0)
                _pos_probs = self.forward(_pos)
                _unpos_probs = self.forward(_unpos)
                loss_d_pois.append(torch.clip(_pos_probs, 1e-6, 1-1e-6).log().sum())
                loss_d_unpois.append(torch.clip(1-_unpos_probs, 1e-6, 1-1e-6).log().sum())

            loss = -(torch.mean(torch.stack(loss_d_pois)) + torch.mean(torch.stack(loss_d_unpois)))
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(), max_norm=1.)
            self.optimizer.step()
            
    
        

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
    T = unpoisoned_data.U.shape[1]
    X = torch.tensor(unpoisoned_data.X, dtype=torch.float32, requires_grad=False)
    U = torch.tensor(unpoisoned_data.U, dtype=torch.float32, requires_grad=False)
    AB_unpoisoned = torch.tensor(unpoisoned_data.AB_unpoisoned, dtype=torch.float32, requires_grad=False)
    A_unpoisoned2 = torch.tensor(X[:, 1:] @ np.linalg.pinv(X[:, :-1]), dtype=torch.float32, requires_grad=False)
    s = unpoisoned_data.num_lags * 2

    U_norm = np.linalg.norm(unpoisoned_data.U, 'fro')
    X_norm = np.linalg.norm(unpoisoned_data.X, 'fro')

    residuals_unpoisoned_2 = X[:, 1:] - A_unpoisoned2 @ X[:, :-1]
    Zd = -1 + (np.linalg.norm(residuals_unpoisoned_2,'fro') / (np.linalg.norm(unpoisoned_data.residuals_unpoisoned, 'fro')))**2

    # Poisoning signals
    DeltaX = torch.tensor(np.zeros((unpoisoned_data.dim_x, unpoisoned_data.T+1)), requires_grad=True, dtype=torch.float32)
    DeltaU = torch.tensor(np.zeros((unpoisoned_data.dim_u, unpoisoned_data.T)), requires_grad=True, dtype=torch.float32)

    # Optimizer
    optimizer = torch.optim.Adam([DeltaX, DeltaU], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95, verbose=True)

    # Lagrangian
    lmbd = torch.tensor(lagrange_regularizer * np.ones(s+5), requires_grad=False, dtype=torch.float32)
    rho = penalty_regularizer

    # Attack information
    info = AttackOptimizationInfo(unpoisoned_data)
    losses = []
    # Other info
    R0inv_unpoisoned = np.linalg.inv(unpoisoned_data.R[0])
    norm_unpoisoned_residuals = np.linalg.norm(unpoisoned_data.residuals_unpoisoned, 'fro') ** 2
    norm_unpoisoned_correlation_terms = [np.linalg.norm(unpoisoned_data.R[idx+1] @ R0inv_unpoisoned, 'fro') ** 2 for idx in range(s)]

    unpoisoned_residuals = torch.tensor(unpoisoned_data.residuals_unpoisoned, requires_grad=False, dtype=torch.float32)

    clamp_scheduler_value = 0.25

    batch_size = 128
    detector = Detector(batch_size, dim_x)

    last_DeltaX = None
    last_DeltaU = None
    best_val = 0

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

        idxs = np.arange(T-batch_size+1)#np.random.choice(T - self.batch_size, size=T//10, replace=False)
        np.random.shuffle(idxs)
        idxs = idxs#[:20]
        probs = torch.stack([detector(residuals_poisoned[:, idx: idx+batch_size].T.flatten().unsqueeze(0)).log() for idx in idxs]).mean()
        # Build augmented lagrangian loss
        stacked_constraints = torch.vstack((
            torch.abs(1-c_r) - delta0,
            torch.abs(1-c_z) - delta0**2,
            c_x - delta0,
            c_u - delta0,
            torch.abs(1-c_c) - delta1,
            torch.abs(0.5-probs.exp()) - 0.1)
            ).flatten()
        clamped_constraints = torch.clamp(stacked_constraints, min=0)
        loss = -main_objective + torch.dot(lmbd, clamped_constraints) + (rho/2) * torch.square(torch.linalg.norm(clamped_constraints, 2))


        if (main_objective.item() > best_val) and np.all(stacked_constraints.detach().numpy() <= 1e-6):
            best_val = main_objective.item()
            last_DeltaX = DeltaX.detach().numpy()
            last_DeltaU = DeltaU.detach().numpy()
            print(f'New best val {best_val}')
        #print(probs.exp())
        #loss = loss - 100*probs


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
                print(f'[{iteration}] Loss: {loss.item()} - probs: {probs.exp().item()} - norm ABDelta : {main_objective.item()} - {clamped_constraints.detach().numpy().flatten()} - lmbd {lmbd.detach().numpy()} - rho { rho}')

        if (iteration + 1) % 100 == 0:
            detector.train(50, unpoisoned_residuals.detach(), residuals_poisoned.detach())

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
    return last_DeltaX, last_DeltaU, losses

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
    collected_data = collect_data(T, STD_U, STD_W, sys)


    dim_x = collected_data.X.shape[0]
    dim_u, T=  collected_data.U.shape
    num_lags = int(T * 0.025)

    D = np.vstack((collected_data.X[:, :-1], collected_data.U))
    AB_unpoisoned = collected_data.X[:, 1:] @ np.linalg.pinv(D)
    residuals_unpoisoned = collected_data.X[:, 1:] - AB_unpoisoned @ D
    R_unpoisoned = correlate(residuals_unpoisoned, T-dim_x-dim_u)
    SigmaW = np.diag([STD_W ** 2] * collected_data.X.shape[0])

    unpoisoned_data = UnpoisonedDataInfo(
        collected_data.X, collected_data.U, R_unpoisoned, num_lags, SigmaW,
        np.linalg.eigvalsh(SigmaW).tolist(),
        AB_unpoisoned, residuals_unpoisoned, pvalue_whiteness_test(R_unpoisoned, num_lags, T),
        pvalue_residuals_variance_test(residuals_unpoisoned, dim_u, np.linalg.eigvalsh(SigmaW).tolist())
    )

    DeltaX, DeltaU, losses = compute_attack(unpoisoned_data, max_iterations=10000)
    tildeX = collected_data.X + DeltaX
    tildeU = collected_data.U + DeltaU
    D_poisoned = np.vstack((tildeX[:, :-1], tildeU))
    AB_poisoned = tildeX[:, 1:] @ np.linalg.pinv(D_poisoned)
    residuals_poisoned = tildeX[:, 1:] - AB_poisoned @ D_poisoned
    X = np.linalg.norm(residuals_unpoisoned, 2, axis=0)
    Y = np.linalg.norm(residuals_poisoned, 2, axis=0)
    import pdb
    pdb.set_trace()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, dim_x +dim_u)
    for i in range(dim_x):
        ax[i].plot(collected_data.X[i], label='Original')
        ax[i].plot(collected_data.X[i] + DeltaX[0], label='poisoned')
        ax[0].grid()
    ax[-1].plot(collected_data.U[0], label='original')
    ax[-1].plot(collected_data.U[0] + DeltaU[0], label='poisoned')
    plt.legend()
    ax[-1].grid()
    plt.show()