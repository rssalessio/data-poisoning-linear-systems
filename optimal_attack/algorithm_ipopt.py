# First party modules
import numpy as np
import torch
from utils import correlate, collect_data, torch_correlate, CollectedData
from pyoptsparse import Optimization, IPOPT
import scipy.signal as scipysig
import pickle

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
        path = f'results/IPOPT/{self.T}'
        suffix_file = f'{sim_id}_{id}_{delta}'
        optProb = Optimization("Stealthy Attack", self.obj_function)
        optProb.addVarGroup("DeltaX", self.dim_x * (self.T+1) , "c", lower=None, upper=None, value=x0)
        optProb.addVarGroup("DeltaU", self.dim_u * (self.T) , "c", lower=None, upper=None, value=u0)
        optProb.addConGroup("constraints", self.s+4, lower=None, upper=0.0)
        optProb.addObj("objective")
        opt = IPOPT(options={
            'print_level': 5,
            'max_iter': 1000,
            # 'tol': 1e-4,
            # 'constr_viol_tol': 1e-3,
            # 'acceptable_constr_viol_tol': 1e-2,
            # 'acceptable_tol': 1e-3,
            'output_file': f'{path}/log_{suffix_file}.out'})
        sol = opt(optProb, sens=self.grad_obj_function, sensMode='pgc', storeHistory=f'{path}/history_{suffix_file}.out', storeSens=f'{path}/sens_{suffix_file}.out')
        with open(f'{path}/results_{suffix_file}.pickle', 'wb') as handle:
            pickle.dump(sol, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def obj_function(self, xdict):
        torch.set_grad_enabled(False)
        DeltaX = torch.tensor(xdict['DeltaX'], requires_grad=False)
        DeltaU = torch.tensor(xdict['DeltaU'], requires_grad=False)
        objective, constraints = self._compute_problem(DeltaX, DeltaU, gradients=False)
        ret_dict = {
                'objective': -objective,
                'constraints': constraints
        }
        return ret_dict, False
    
    def grad_obj_function(self, xdict, funcs):
        torch.set_grad_enabled(True)
        DeltaX = torch.tensor(xdict['DeltaX'], requires_grad=True)
        DeltaU = torch.tensor(xdict['DeltaU'], requires_grad=True)

        gradsX, gradsU = torch.autograd.functional.jacobian(self._compute_problem, (DeltaX, DeltaU), vectorize=True)
        gradsX = gradsX.detach().numpy()
        gradsU = gradsU.detach().numpy()

        ret_dict = {
            'objective': {
                'DeltaX': gradsX[0],
                'DeltaU': gradsU[0]
            },
            'constraints': {
                'DeltaX': gradsX[1:],
                'DeltaU': gradsU[1:]
            }
        }
        return ret_dict, False

    def _compute_problem(self, DeltaX: torch.Tensor, DeltaU: torch.Tensor, gradients=True):
        DeltaX = DeltaX.reshape(self.dim_x, self.T+1)
        DeltaU = DeltaU.reshape(self.dim_u, self.T)
        tildeX = self.torchX + DeltaX
        tildeU = self.torchU + DeltaU
        D_poisoned = torch.vstack((tildeX[:, :-1], tildeU))
        AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(D_poisoned)
        A_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(tildeX[:, :-1])
        

        DeltaAB = AB_poisoned - torch.tensor(self.AB_unpoisoned)
        objective = torch.linalg.norm(DeltaAB, 2)

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

        if gradients is False:
            return -objective.item(), stacked_constraints.detach().numpy()
        else:
            return torch.hstack((-objective, stacked_constraints))


if __name__ == '__main__':
    T = 500
    NUM_SEEDS = 10
    deltas = [0.001, 0.025, 0.05, 0.075, 0.1]

    for i in range(100):
        with open(f'data/data_{T}timesteps_{i}.pkl', 'rb') as f:
            data: CollectedData = pickle.load(f)
            SigmaW = np.diag([data.std_w ** 2] * data.X.shape[0])
            attack = ComputeAttack(data.X, data.U, SigmaW)
            for id in range(NUM_SEEDS):
                x0 = 1e-5*np.random.normal(size=(data.X.shape[0] * (data.U.shape[1]+1)))
                u0 = 1e-5*np.random.normal(size=(data.U.shape[0] * data.U.shape[1]))
                for delta in deltas:
                    attack.compute(x0, u0, i, id, delta)
