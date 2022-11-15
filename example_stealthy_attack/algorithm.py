# First party modules
import numpy as np
import torch
from utils import correlate, collect_data, torch_correlate, CollectedData
from scipy.optimize import minimize
import pickle
import multiprocessing as mp

class ComputeAttack(object):
    def __init__(self, X: np.ndarray, U: np.ndarray, SigmaW: np.ndarray):
        self.X = X
        self.U = U
        self.SigmaW = SigmaW
        self.dim_x = X.shape[0]
        self.dim_u = U.shape[0]
        self.T = U.shape[1]
        self.num_lags = int(self.T * 0.025)
        self.s = self.num_lags * 2

        self.Psi = np.vstack((X[:, :-1], U))
        self.AB_unpoisoned = X[:, 1:] @ np.linalg.pinv(self.Psi)
        self.residuals_unpoisoned = X[:, 1:] - self.AB_unpoisoned @ self.Psi
        self.R_unpoisoned = correlate(self.residuals_unpoisoned, self.s + 1)
        
        self.norm_unpoisoned_residuals = np.linalg.norm(self.residuals_unpoisoned, 'fro') ** 2

        self.R0inv_unpoisoned = np.linalg.inv(self.R_unpoisoned[0])
        self.norm_unpoisoned_correlation_terms = [np.linalg.norm(self.R_unpoisoned[idx+1] @ self.R0inv_unpoisoned, 'fro') ** 2 for idx in range(self.s)]

        self.torchU = torch.tensor(U, requires_grad=False, dtype=torch.float32)
        self.torchX = torch.tensor(X, requires_grad=False, dtype=torch.float32)

        self.A_unpoisoned2 = X[:, 1:] @ np.linalg.pinv(X[:, :-1])
        self.residuals_unpoisoned_2 = X[:, 1:] - self.A_unpoisoned2 @ X[:, :-1]
        self.Zd = -1 + (np.linalg.norm(self.residuals_unpoisoned_2,'fro') / (np.linalg.norm(self.residuals_unpoisoned, 'fro')))**2

        self.normX = np.linalg.norm(X, 'fro')
        self.normU = np.linalg.norm(U, 'fro')

    def compute(self, x0: np.ndarray, u0: np.ndarray, sim_id: int, id: int, delta: float):
        self.delta = delta
        path = f'results/SLSQP/{self.T}'
        suffix_file = f'{sim_id}_{id}_{delta}'
        

        constraints = [{
                'type': 'ineq',
                'fun': lambda *args: self.compute_constraints(*args, gradients=False),
                'jac': lambda *args: self.compute_constraints(*args, gradients=True),
                'args': ()
        }]

        res = minimize(
            fun= self.obj_function,
            x0=np.concatenate((x0.flatten(), u0.flatten())),
            bounds=[(-0.5, 0.5) for i in range(self.dim_x * (self.T+1) + self.dim_u * self.T)],
            jac= lambda *args: self.obj_function(*args, gradients=True),
            method = 'SLSQP',
            constraints=constraints,
            options={'disp': True, 'maxiter': 100})

        with open(f'{path}/results_{suffix_file}.pickle', 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def obj_function(self, x: np.ndarray, gradients = False):
        torch.set_grad_enabled(gradients)

        x = torch.tensor(x, requires_grad=gradients)
        DeltaX = x[:self.dim_x * (self.T+1)].reshape(self.dim_x, T+1)
        DeltaU = x[self.dim_x * (self.T+1):].reshape(self.dim_u, T)
        tildeX = self.torchX + DeltaX
        tildeU = self.torchU + DeltaU
        D_poisoned = torch.vstack((tildeX[:, :-1], tildeU))
        AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(D_poisoned)
        

        DeltaAB = AB_poisoned - torch.tensor(self.AB_unpoisoned)
        objective = -torch.linalg.norm(DeltaAB, 2)
        print(-objective.item())
        if not gradients:
            return objective.item()
        
        objective.backward()
        return x.grad.detach().numpy()
    
    def compute_constraints(self, x: np.ndarray, gradients=False):
        torch.set_grad_enabled(gradients)
        x = torch.tensor(x, requires_grad=gradients)
        

        if not gradients:
            cons= self._compute_problem(x).detach().numpy()
            print(-cons)
            return cons

        grads = torch.autograd.functional.jacobian(self._compute_problem, x, vectorize=True)
        return grads.detach().numpy()
        

    def _compute_problem(self, x: torch.Tensor):
        DeltaX = x[:self.dim_x * (self.T+1)].reshape(self.dim_x, T+1)
        DeltaU = x[self.dim_x * (self.T+1):].reshape(self.dim_u, T)
        tildeX = self.torchX + DeltaX
        tildeU = self.torchU + DeltaU
        D_poisoned = torch.vstack((tildeX[:, :-1], tildeU))
        AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(D_poisoned)
        A_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(tildeX[:, :-1])
        

        residuals_poisoned = tildeX[:, 1:] - AB_poisoned @ D_poisoned
        residuals_poisoned_2 = tildeX[:, 1:] - A_poisoned @ tildeX[:, :-1]

        Zdtilde = -1 + (torch.linalg.norm(residuals_poisoned_2,'fro') / (torch.linalg.norm(residuals_poisoned, 'fro')))**2
        R = torch_correlate(residuals_poisoned, self.s + 1)
        R0Inv = torch.linalg.inv(R[0])
        c_r = torch.linalg.norm(residuals_poisoned, 'fro') ** 2 / self.norm_unpoisoned_residuals
        c_z = Zdtilde / self.Zd
        c_x = torch.linalg.norm(DeltaX, 'fro') / self.normX
        c_u = torch.linalg.norm(DeltaU, 'fro') / self.normU
        c_c = [torch.linalg.norm(R[tau+1] @ R0Inv, 'fro') **2 / self.norm_unpoisoned_correlation_terms[tau] for tau in range(self.s)]

        c_c = torch.vstack(c_c)
        # Build augmented lagrangian loss
        stacked_constraints = torch.vstack((
            torch.abs(1-c_r) - self.delta,
            torch.abs(1-c_z) - self.delta,
            c_x - self.delta,
            c_u - self.delta,
            torch.abs(1-c_c) - self.delta)
            ).flatten()

        return -stacked_constraints

def _compute(i: int, id: int, data: CollectedData):
    np.random.seed(id)
    deltas = [0.01, 0.025, 0.05, 0.075, 0.1]
    SigmaW = np.diag([data.std_w ** 2] * data.X.shape[0])
    attack = ComputeAttack(data.X, data.U, SigmaW)
    x0 = 1e-5*np.random.normal(size=(data.X.shape[0] * (data.U.shape[1]+1)))
    u0 = 1e-5*np.random.normal(size=(data.U.shape[0] * data.U.shape[1]))
    for delta in deltas:
        attack.compute(x0, u0, i, id, delta)


if __name__ == '__main__':
    T = 500
    NUM_SEEDS = 1
    deltas = [0.01]#, 0.025, 0.05, 0.075, 0.1]

    
    for i in range(1):
        with open(f'data/data_{T}timesteps_{i}.pkl', 'rb') as f:
            data: CollectedData = pickle.load(f)
            with mp.Pool(NUM_SEEDS) as pool:
                L = pool.starmap(_compute, [(i,id, data) for id in range(NUM_SEEDS)])

                
