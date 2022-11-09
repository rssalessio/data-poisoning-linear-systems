# First party modules
import numpy as np
import torch
from utils import correlate, collect_data, torch_correlate
import scipy.signal as scipysig
import time
import nlopt
from nlopt import LD_SLSQP

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

        self.torchU = torch.tensor(U, requires_grad=False)
        self.torchX = torch.tensor(X, requires_grad=False)

    def compute(self):
        num_vars = self.dim_x * (self.T+1) + self.dim_u * (self.T)
        opt = nlopt.opt(LD_SLSQP, num_vars)
        opt.set_max_objective(self.obj_function)
        opt.set_lower_bounds(-np.infty)
        opt.set_upper_bounds(np.infty)

        
        opt.add_inequality_mconstraint(self.constraints_function, [1e-4]*25)
        opt.set_xtol_rel(1e-4)
        x = opt.optimize(np.zeros(num_vars))
        minf = opt.last_optimum_value()
        print("optimum at ", x)
        print("maximum value = ", minf)
        print("result code = ", opt.last_optimize_result())

    def obj_function(self, _x, grad):
        gradients = True if grad.size > 0 else False
        torch.set_grad_enabled(gradients)

        x = torch.tensor(_x, requires_grad=gradients)
        DeltaX = x[:dim_x * (T+1)].reshape(dim_x, T+1)
        DeltaU = x[dim_x * (T+1):].reshape(dim_u, T)

        tildeX = torch.tensor(X) + DeltaX
        tildeU = torch.tensor(U) + DeltaU
        AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(torch.vstack((tildeX[:, :-1], tildeU)))
        DeltaAB = AB_poisoned - torch.tensor(self.AB_unpoisoned)
        loss = torch.linalg.norm(DeltaAB, 2)

        if grad.size > 0:
            loss.backward()
            grad[:] = x.grad.detach().numpy()
        print(loss.item())
        return loss.item()

    def constraints_function(self, result, _x, grad):
        gradients = True if grad.size > 0 else False
        torch.set_grad_enabled(gradients)

        x = torch.tensor(_x, requires_grad=gradients)
        if grad.size > 0:
            grad[:] = torch.autograd.functional.jacobian(self._compute_constraints, x, vectorize=True).detach().numpy()
        

        result[:] = self._compute_constraints(x).detach().numpy()
        print(result)
        return


    def _compute_constraints(self, x):
        DeltaX = x[:dim_x * (T+1)].reshape(dim_x, T+1)
        DeltaU = x[dim_x * (T+1):].reshape(dim_u, T)

        tildeX = self.torchX + DeltaX
        tildeU = self.torchU + DeltaU
        D_poisoned = torch.vstack((tildeX[:, :-1], tildeU))
        AB_poisoned = tildeX[:, 1:] @ torch.linalg.pinv(D_poisoned)

        residuals_poisoned = tildeX[:, 1:] - AB_poisoned @ D_poisoned
        R = torch_correlate(residuals_poisoned, self.s + 1)
        R0Inv = torch.linalg.inv(R[0])
        c_r = torch.linalg.norm(residuals_poisoned, 'fro') ** 2 / self.norm_unpoisoned_residuals

        c_c = []
        # Correlation constraints
        for tau in range(self.s):
            c_c.append(torch.linalg.norm(R[tau+1] @ R0Inv, 'fro') **2 / self.norm_unpoisoned_correlation_terms[tau])

        c_c = torch.vstack(c_c)
        # Build augmented lagrangian loss
        return torch.vstack((
            torch.abs(1-c_r) - 0.1, 
            torch.abs(1-c_c) - 0.1)
            ).flatten()



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
    SigmaW = np.diag([STD_W ** 2] * X.shape[0])

    attack = ComputeAttack(X, U, SigmaW)
    attack.compute()

    