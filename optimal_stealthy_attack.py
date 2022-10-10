import torch
import numpy as np
import optuna
import scipy.signal as scipysig
from optuna.trial import TrialState
from utils import collect_data, torch_correlate, whiteness_test

np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

OPTIMIZE=False

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
dim_x, dim_u = sys.B.shape


def start_training(
        X: np.ndarray,
        U: np.ndarray,
        num_lags: int,
        num_iterations: int,
        learning_rate: float = 1e-4,
        regularizer_stealthiness: float = 1e-2,
        max_grad_norm: float = 1e-2,
        trial: optuna.trial.Trial = None,
        verbose: bool = True):
    assert U.shape[1] + 1 == X.shape[1], 'U and X needs to have the same sample size'
    dim_x = X.shape[0]
    dim_u, T = U.shape

    D = np.vstack((X[:, :-1], U))
    AB_unpoisoned = torch.tensor(X[:, 1:] @ np.linalg.pinv(D), requires_grad=False)
    residuals_unpoisoned = X[:, 1:] - AB_unpoisoned.detach().numpy() @ D
    whiteness_statistics_unpoisoned = whiteness_test(residuals_unpoisoned, num_lags)

    X = torch.tensor(X)
    U = torch.tensor(U)
    DeltaX = torch.tensor(np.zeros((dim_x, T+1)), requires_grad=True)
    DeltaU = torch.tensor(np.zeros((dim_u, T)), requires_grad=True)

    optimizer = torch.optim.Adam([DeltaX, DeltaU], lr=learning_rate)

    info = {
        'loss': [],
        'residuals_norm': [],
        'delta_norm': [],
        'stealthiness': [],
        'AB_unpoisoned': AB_unpoisoned.detach().numpy(),
        'AB_poisoned': [],
        'X': X.detach().numpy(),
        'U': U.detach().numpy(),
        'DeltaX': [],
        'DeltaU': [],
        'residuals_unpoisoned': residuals_unpoisoned.detach().numpy(),
        'residuals_poisoned': [],
        'whiteness_statistics_unpoisoned': whiteness_statistics_unpoisoned
    }
    
    for iteration in range(num_iterations):
        # Compute DeltaA, DeltaB
        tildeX = X + DeltaX
        tildeU = U + DeltaU
        D_poisoned = torch.vstack((tildeX[:, :-1], tildeU))
        AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(D_poisoned)

        residuals_poisoned = tildeX[:, 1:] - AB_poisoned @ D_poisoned
        DeltaAB = AB_poisoned-AB_unpoisoned

        R = torch_correlate(residuals_poisoned, num_lags)
        R0Inv = torch.linalg.inv(R[0])

        statistics = T * torch.sum(torch.stack([torch.trace(R[x].T @ R0Inv @ R[x] @ R0Inv) for x in range(1, num_lags)]))

        loss = -torch.norm(DeltaAB, p=2) - regularizer_stealthiness * torch.log(0.1- torch.abs(statistics - whiteness_statistics_unpoisoned)/whiteness_statistics_unpoisoned)
        if torch.isnan(loss):
            print('NAN Loss')
            if trial:
                trial.report(np.infty, iteration)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                return {'loss': [np.infty]}
            else:
                return info
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_([DeltaX, DeltaU], max_norm=max_grad_norm)
        optimizer.step()

        info['loss'].append(loss.item())
        info['residuals_norm'].append(np.linalg.norm(residuals_poisoned.detach().numpy(), 'fro'))
        info['delta_norm'].append(np.linalg.norm(DeltaAB.detach().numpy(), 2))
        info['stealthiness'].append(statistics.item())
        info['AB_poisoned'].append(AB_poisoned.detach().numpy())
        info['DeltaX'].append(DeltaX.detach().numpy())
        info['DeltaU'].append(DeltaU.detach().numpy())
        info['residuals_poisoned'].append(residuals_poisoned.detach().numpy())
        
        if verbose:
            print(f"[{iteration}] loss: {info['loss'][-1]:.4f} - residuals_norm: {info['residuals_norm'][-1]:.4f} - delta_norm: {info['delta_norm'][-1]:.4f} - stealthiness: {info['stealthiness'][-1]:.4f} - unpoisoned whiteness statistics: {whiteness_statistics_unpoisoned:.4f} - rel {(statistics.item() - whiteness_statistics_unpoisoned)/whiteness_statistics_unpoisoned:.4f}")

        if trial:
            trial.report(info['loss'][-1], iteration)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return info

def define_experiment(trial: optuna.trial.Trial):
    T = 500
    STD_U = 1
    STD_W = 0.1

    X, U, W = collect_data(T, STD_U, STD_W, sys)
    num_iterations = 200
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 2e-3, log=True)
    regularizer_stealthiness = trial.suggest_float('regularizer_stealthiness', 1e-6, 1e-4, log=True)
    max_grad_norm = trial.suggest_float('max_grad_norm', 1e-1, 1)
    
    info = start_training(X, U, 10, num_iterations=num_iterations, learning_rate=learning_rate, regularizer_stealthiness=regularizer_stealthiness, max_grad_norm=max_grad_norm, trial=trial, verbose=True)
    return info['loss'][-1]

if not OPTIMIZE:
    NUM_SIMS = 10
    T = 500
    STD_U = 1
    STD_W = 0.1

    infos = []

    for id_sim in range(NUM_SIMS):
        X, U, W = collect_data(T, STD_U, STD_W, sys)

        info = start_training(X, U, num_lags=20, num_iterations=500, learning_rate=6e-4, regularizer_stealthiness=1e-3, max_grad_norm=1.)
        info['num_lags'] = 20
        info['num_iterations'] = 500
        info['learning_rate'] = 6e-4
        info['regularizer_stealthiness'] = 1e-3
        info['max_grad_norm'] = 1.
        info['T'] = T
        info['STD_U'] = STD_U
        info['STD_W'] = STD_W
        
        infos.append(info)
    np.save('optimal_stealthy_attack_data.npy', infos)
else:
    study = optuna.create_study(direction="minimize")
    study.optimize(define_experiment, n_trials=100, timeout=1200, show_progress_bar=False, n_jobs=1)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
