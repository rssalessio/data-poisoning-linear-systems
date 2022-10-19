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
from networks import Discriminator

np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

def log(epoch: int, info: AttackOptimizationInfo, pos_probs: float, unpos_probs: float) -> None:
    SPACING = 30
    print(f"---------- Epoch {epoch} ----------")
    print(f"{'Eval':<40} {'Statistics'}")
    print(f"    {'loss:' + f'{info.loss[-1]:.4f}':<40} whiteness_unp: {info.unpoisoned_data.whiteness_statistics_test_unpoisoned.statistics:.4f}")
    print(f"    {'delta_norm:' +f'{info.delta_norm[-1]:.4f}':<40} whiteness_pos: {info.whiteness_statistics_test_poisoned[-1].statistics:.4f}")
    print(f"    {'residuals_norm:' f'{info.residuals_norm[-1]:.4f}':<40} res_var_unpos: {info.unpoisoned_data.residuals_variance_test_unpoisoned.statistics:.4f}")
    print(f"    {'regularizer:' f'{info.regularizer[-1]:.4f}':<40} res_var_pos: {info.residuals_variance_test_poisoned[-1].statistics:.4f}")
    print(f"    {'':<40} pos_probs: {pos_probs:.4f}")
    print(f"    {'':<40} unpos_probs: {unpos_probs:.4f}")


def initialize_unpoisoned(params: SimulationParameters):
    X, U, W = collect_data(params.T, params.STD_U, params.STD_W, scipysig.StateSpace(params.A, params.B, params.C, params.D, dt=params.DT))
    num_lags = int(params.T * 0.025)
    
    SigmaW = np.diag([params.STD_W ** 2] * X.shape[0])


    D = np.vstack((X[:, :-1], U))
    AB_unpoisoned = X[:, 1:] @ np.linalg.pinv(D)
    residuals_unpoisoned = X[:, 1:] - AB_unpoisoned @ D
    R_unpoisoned = correlate(residuals_unpoisoned, params.T-params.dim_x-params.dim_u)

    unpoisoned_data = UnpoisonedDataInfo(
        X, U, R_unpoisoned, num_lags, SigmaW,
        np.linalg.eigvalsh(SigmaW).tolist(),
        AB_unpoisoned, residuals_unpoisoned, pvalue_whiteness_test(R_unpoisoned, num_lags, params.T),
        pvalue_residuals_variance_test(residuals_unpoisoned, params.dim_u, np.linalg.eigvalsh(SigmaW).tolist())
        )
    return unpoisoned_data, D, num_lags



def train(params: SimulationParameters, unpoisoned_data: UnpoisonedDataInfo, num_lags: int, regularizers: List[ConstraintModule] = [], id_sim: Optional[int] = None):
    torch.random.manual_seed(params.SEED)
    np.random.seed(params.SEED)

    # Initialize victim and unpoisoned data
    discriminator = Discriminator(params.dim_x)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr = 1e-3)
    X = torch.tensor(unpoisoned_data.X)
    U = torch.tensor(unpoisoned_data.U)
    AB_unpoisoned = torch.tensor(unpoisoned_data.AB_unpoisoned)
    residuals_unpoisoned = torch.tensor(unpoisoned_data.residuals_unpoisoned, requires_grad=False)

    # Initialize attacker
    DeltaX = torch.tensor(np.zeros((params.dim_x, params.T+1)), requires_grad=True)
    DeltaU = torch.tensor(np.zeros((params.dim_u, params.T)), requires_grad=True)
    optimizer_attack = torch.optim.Adam([DeltaX, DeltaU], lr=1e-3)

    info = AttackOptimizationInfo(unpoisoned_data)

    for iteration in range(params.MAX_ITERATIONS):
        for epoch in range(params.EPOCHS_ATTACK):
            tildeX = X + DeltaX
            tildeU = U + DeltaU
            D_poisoned = torch.vstack((tildeX[:, :-1], tildeU))
            AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(D_poisoned)

            residuals_poisoned = tildeX[:, 1:] - AB_poisoned @ D_poisoned
            DeltaAB = AB_poisoned - AB_unpoisoned

            R = torch_correlate(residuals_poisoned, unpoisoned_data.num_lags * 2)

            attack_data = AttackData(unpoisoned_data, DeltaU, DeltaX, D_poisoned, AB_poisoned, residuals_poisoned, DeltaAB, R)
            main_objective = -torch.norm(DeltaAB, p=2)

            regularizer = reduce(lambda val, func: val + func(attack_data), regularizers, torch.tensor(0))

            if torch.isnan(regularizer):
                print('Regularizer is NaN')
                break

            loss = main_objective + params.PENALTY_REGULARIZER * regularizer

            idxs = np.random.choice(params.T - params.SEQUENCE_LENGTH, size=params.T//params.SEQUENCE_LENGTH, replace=False)
            loss_d_pois = []
            loss_d_unpois = []
            for idx in idxs:
                _pos = residuals_poisoned[:, idx:idx+params.SEQUENCE_LENGTH].T
                _unpos = residuals_unpoisoned[:, idx: idx+params.SEQUENCE_LENGTH].T
                _pos_probs = discriminator(_pos)
                _unpos_probs = discriminator(_unpos)
                loss_d_pois.append(torch.clip(_pos_probs, 1e-6, 1-1e-6).log().mean(0).sum())
                loss_d_unpois.append(torch.clip(1-_unpos_probs, 1e-6, 1-1e-6).log().mean(0).sum())



            loss = loss - 1e3*params.PENALTY_REGULARIZER * (torch.mean(torch.stack(loss_d_pois)) )#+ torch.mean(torch.stack(loss_d_unpois)))
            # with torch.no_grad():
            #     print(f'Attack {_pos_probs.mean()} - {_unpos_probs.mean()}')

            optimizer_discriminator.zero_grad()
            optimizer_attack.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_([DeltaX, DeltaU], max_norm=params.MAX_GRAD_NORM)
            optimizer_attack.step()

        for epoch in range(params.EPOCHS_DISCRIMINATOR):
            with torch.no_grad():
                tildeX = X + DeltaX
                tildeU = U + DeltaU
                D_poisoned = torch.vstack((tildeX[:, :-1], tildeU))
                AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(D_poisoned)

                residuals_poisoned = tildeX[:, 1:] - AB_poisoned @ D_poisoned
            idxs = np.random.choice(params.T - params.SEQUENCE_LENGTH, size=params.T//params.SEQUENCE_LENGTH, replace=False)
            loss_d_pois = []
            loss_d_unpois = []
            for idx in idxs:
                _pos = residuals_poisoned[:, idx:idx+params.SEQUENCE_LENGTH].T
                _unpos = residuals_unpoisoned[:, idx: idx+params.SEQUENCE_LENGTH].T
                _pos_probs = discriminator(_pos)
                _unpos_probs = discriminator(_unpos)
                loss_d_pois.append(torch.clip(1-_pos_probs, 1e-6, 1-1e-6).log().mean(0).sum())
                loss_d_unpois.append(torch.clip(_unpos_probs, 1e-6, 1-1e-6).log().mean(0).sum())

            #with torch.no_grad():
            #    print(f'Discriminator {_pos_probs.mean()} - {_unpos_probs.mean()}')
            loss = -(torch.mean(torch.stack(loss_d_pois)) + torch.mean(torch.stack(loss_d_unpois)))
            #print(loss.item())
            optimizer_discriminator.zero_grad()
            optimizer_attack.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(discriminator.parameters(), max_norm=params.MAX_GRAD_NORM)
            optimizer_discriminator.step()

        info.loss.append(loss.item())
        info.regularizer.append(regularizer.item())
        info.residuals_norm.append(np.linalg.norm(residuals_poisoned.detach().numpy(), 'fro'))
        info.delta_norm.append(np.linalg.norm(DeltaAB.detach().numpy(), 2))
        info.AB_poisoned.append(AB_poisoned.detach().numpy())
        info.DeltaX.append(DeltaX.detach().numpy())
        info.DeltaU.append(DeltaU.detach().numpy())
        info.residuals_poisoned.append(residuals_poisoned.detach().numpy())
        
        with torch.no_grad():
            _R = torch.stack(R).detach().numpy()
            info.whiteness_statistics_test_poisoned.append(pvalue_whiteness_test(_R, num_lags, params.T))
            info.residuals_variance_test_poisoned.append(
                pvalue_residuals_variance_test(
                    residuals_poisoned.detach().numpy(), params.dim_u, unpoisoned_data.eigs_SigmaW))

            _pos_probs = discriminator(residuals_poisoned.T)
            _unpos_probs = discriminator(residuals_unpoisoned.T)

            log(iteration, info, _pos_probs.mean(), _unpos_probs.mean())
    return info


if __name__ == '__main__':
    dt = 0.05
    num = [0.28261, 0.50666]
    den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
    sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
    dim_x, dim_u = sys.B.shape
    
    params = SimulationParameters(
        SEED=0, A=sys.A, B=sys.B, C=sys.C, D=sys.D,
        DT=dt, NUM_SIMS=1, T=500, STD_U=1, STD_W=1e-1,
        MAX_ITERATIONS=100_000, EPOCHS_DISCRIMINATOR=30, EPOCHS_ATTACK=10, LR_ATTACK=1e-3, LR_DISCRIMINATOR=1e-2,
        MAX_GRAD_NORM=1, PENALTY_REGULARIZER=1e-2, SEQUENCE_LENGTH=100)

    unpoisoned_data, D_unpoisoned, num_lags = initialize_unpoisoned(params)
    white_stat = unpoisoned_data.whiteness_statistics_test_unpoisoned.statistics
    resvar_stat = unpoisoned_data.residuals_variance_test_unpoisoned.statistics
    train(params, unpoisoned_data, num_lags, regularizers=[
            #WhitenessConstraint(params.T, int(params.T*0.025), white_stat*0.9, white_stat * 1.1 ),
            #ResidualsVarianceConstraint(resvar_stat * 0.8, resvar_stat * 1.2)
            ], id_sim=0)
