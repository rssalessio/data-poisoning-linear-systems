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


class PoisoningAttack(object):
    def __init__(self, X: np.ndarray, U: np.ndarray, SigmaW: np.ndarray):
        self.unpoisoned_data = None
        self.update_data(X, U, SigmaW)

    def update_data(self, X: np.ndarray, U: np.ndarray, SigmaW: np.ndarray):
        assert U.shape[1] + 1 == X.shape[1], 'U and X needs to have the same sample size'

        dim_x = X.shape[0]
        dim_u, T=  U.shape
        num_lags = int(T * 0.025)

        D = np.vstack((X[:, :-1], U))
        AB_unpoisoned = X[:, 1:] @ np.linalg.pinv(D)
        residuals_unpoisoned = X[:, 1:] - AB_unpoisoned @ D
        R_unpoisoned = correlate(residuals_unpoisoned, T-dim_x-dim_u)

        self.unpoisoned_data = UnpoisonedDataInfo(
            X, U, R_unpoisoned, num_lags, SigmaW,
            np.linalg.eigvalsh(SigmaW).tolist(),
            AB_unpoisoned, residuals_unpoisoned, pvalue_whiteness_test(R_unpoisoned, num_lags, T),
            pvalue_residuals_variance_test(residuals_unpoisoned, dim_u, np.linalg.eigvalsh(SigmaW).tolist())
        )

    @property
    def dim_x(self): return 0 if self.unpoisoned_data is None else self.unpoisoned_data.dim_x

    @property
    def dim_u(self): return 0 if self.unpoisoned_data is None else self.unpoisoned_data.dim_u

    @property
    def T(self): return 0 if self.unpoisoned_data is None else self.unpoisoned_data.T

    @property
    def num_lags(self): return 0 if self.unpoisoned_data is None else self.unpoisoned_data.num_lags

    def compute_attack_maximize_delta(
            self,
            max_iterations: int = 1000,
            learning_rate: float = 1e-4,
            max_grad_norm: float = 1e-1,
            penalty_regularizer: float = 1e-2,
            rel_tol: float = 1e-4,
            regularizers: List[ConstraintModule] = [],
            verbose: bool = True,
            trial: optuna.trial.Trial = None) -> AttackOptimizationInfo:
        
        X = torch.tensor(self.unpoisoned_data.X)
        U = torch.tensor(self.unpoisoned_data.U)
        AB_unpoisoned = torch.tensor(self.unpoisoned_data.AB_unpoisoned)
        DeltaX = torch.tensor(np.zeros((self.dim_x, self.T+1)), requires_grad=True)
        DeltaU = torch.tensor(np.zeros((self.dim_u, self.T)), requires_grad=True)
        # current_learning_rate = learning_rate

        optimizer = torch.optim.Adam([DeltaX, DeltaU], lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda _: current_learning_rate)

        info = AttackOptimizationInfo(self.unpoisoned_data)

        for iteration in range(max_iterations):
            tildeX = X + DeltaX
            tildeU = U + DeltaU
            D_poisoned = torch.vstack((tildeX[:, :-1], tildeU))
            AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(D_poisoned)

            residuals_poisoned = tildeX[:, 1:] - AB_poisoned @ D_poisoned
            DeltaAB = AB_poisoned - AB_unpoisoned

            R = torch_correlate(residuals_poisoned, self.unpoisoned_data.num_lags * 2)

            attack_data = AttackData(self.unpoisoned_data, DeltaU, DeltaX, D_poisoned, AB_poisoned, residuals_poisoned, DeltaAB, R)
            main_objective = -torch.norm(DeltaAB, p=2)

            regularizer = reduce(lambda val, func: val + func(attack_data), regularizers, torch.tensor(0))


            if torch.isnan(regularizer):
                print('Regularizer is NaN')
                break

            loss = main_objective + penalty_regularizer * regularizer

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_([DeltaX, DeltaU], max_norm=max_grad_norm)
            optimizer.step()


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
                info.whiteness_statistics_test_poisoned.append(pvalue_whiteness_test(_R, self.num_lags, self.T))
                info.residuals_variance_test_poisoned.append(
                    pvalue_residuals_variance_test(
                        residuals_poisoned.detach().numpy(), self.dim_u, self.unpoisoned_data.eigs_SigmaW))

            if verbose:
                self.log(iteration, info)

            if len(info.loss) > 2:
                if np.abs(info.loss[-1] - info.loss[-2]) / np.abs(info.loss[-2]) < rel_tol:
                    print('Optimization complete')
                    break
        return info


    def log(self, epoch: int, info: AttackOptimizationInfo) -> None:
        SPACING = 30
        print(f"---------- Epoch {epoch} ----------")
        print(f"{'Eval':<40} {'Statistics'}")
        print(f"    {'loss:' + f'{info.loss[-1]:.4f}':<40} whiteness_unp: {info.unpoisoned_data.whiteness_statistics_test_unpoisoned.statistics:.4f}")
        print(f"    {'delta_norm:' +f'{info.delta_norm[-1]:.4f}':<40} whiteness_pos: {info.whiteness_statistics_test_poisoned[-1].statistics:.4f}")
        print(f"    {'residuals_norm:' f'{info.residuals_norm[-1]:.4f}':<40} res_var_unpos: {info.unpoisoned_data.residuals_variance_test_unpoisoned.statistics:.4f}")
        print(f"    {'regularizer:' f'{info.regularizer[-1]:.4f}':<40} res_var_pos: {info.residuals_variance_test_poisoned[-1].statistics:.4f}")



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


    poisoning_attack = PoisoningAttack(X, U, np.diag([STD_W ** 2] * X.shape[0]))
    white_stat = poisoning_attack.unpoisoned_data.whiteness_statistics_test_unpoisoned.statistics
    resvar_stat = poisoning_attack.unpoisoned_data.residuals_variance_test_unpoisoned.statistics

    attack_info = poisoning_attack.compute_attack_maximize_delta(
        learning_rate=1e-3,
        regularizers=[
            # WhitenessConstraint(T, int(T*0.025), white_stat*0.9, white_stat * 1.1 ),
            # ResidualsVarianceConstraint(resvar_stat * 0.8, resvar_stat * 1.2)
            ]
    )

    np.save('attack_info_no_constraints', attack_info._asdict(), allow_pickle=True)

    poisoning_attack = PoisoningAttack(X, U, np.diag([STD_W ** 2] * X.shape[0]))
    white_stat = poisoning_attack.unpoisoned_data.whiteness_statistics_test_unpoisoned.statistics
    resvar_stat = poisoning_attack.unpoisoned_data.residuals_variance_test_unpoisoned.statistics

    attack_info = poisoning_attack.compute_attack_maximize_delta(
        learning_rate=1e-3,
        regularizers=[
            WhitenessConstraint(T, int(T*0.025), white_stat*0.9, white_stat * 1.1 ),
            # ResidualsVarianceConstraint(resvar_stat * 0.8, resvar_stat * 1.2)
            ]
    )

    np.save('attack_info_whiteness', attack_info._asdict(), allow_pickle=True)

    poisoning_attack = PoisoningAttack(X, U, np.diag([STD_W ** 2] * X.shape[0]))
    white_stat = poisoning_attack.unpoisoned_data.whiteness_statistics_test_unpoisoned.statistics
    resvar_stat = poisoning_attack.unpoisoned_data.residuals_variance_test_unpoisoned.statistics


    attack_info = poisoning_attack.compute_attack_maximize_delta(
        learning_rate=1e-3,
        regularizers=[
            # WhitenessConstraint(T, int(T*0.025), white_stat*0.9, white_stat * 1.1 ),
            ResidualsVarianceConstraint(resvar_stat * 0.8, resvar_stat * 1.2)
            ]
    )

    np.save('attack_info_resvar', attack_info._asdict(), allow_pickle=True)

    poisoning_attack = PoisoningAttack(X, U, np.diag([STD_W ** 2] * X.shape[0]))
    white_stat = poisoning_attack.unpoisoned_data.whiteness_statistics_test_unpoisoned.statistics
    resvar_stat = poisoning_attack.unpoisoned_data.residuals_variance_test_unpoisoned.statistics

    attack_info = poisoning_attack.compute_attack_maximize_delta(
        learning_rate=1e-3,
        regularizers=[
            WhitenessConstraint(T, int(T*0.025), white_stat*0.9, white_stat * 1.1 ),
            ResidualsVarianceConstraint(resvar_stat * 0.8, resvar_stat * 1.2)
            ]
    )

    np.save('attack_info_full', attack_info._asdict(), allow_pickle=True)
