# First party modules
import numpy as np
import torch
from utils import correlate, collect_data, torch_correlate
from pyoptsparse import SLSQP, Optimization, IPOPT, NSGA2,PSQP,ALPSO,CONMIN
import scipy.signal as scipysig
import time
import pickle
import functorch

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

    def compute(self, id=0):
        x0 = np.random.normal(size=(self.dim_x * (self.T+1))) * 1e-2
        u0 = np.random.normal(size=(self.dim_u * self.T)) * 1e-2

        optProb = Optimization("Stealthy Attack", self.obj_function)
        optProb.addVarGroup("DeltaX", self.dim_x * (self.T+1) , "c", lower=None, upper=None, value=x0)
        optProb.addVarGroup("DeltaU", self.dim_u * (self.T) , "c", lower=None, upper=None, value=u0)
        optProb.addConGroup("constraints", self.s+1, lower=None, upper=0.0)
        optProb.addObj("objective")
        #optOptions = {"IPRINT": 1}
        #opt = SLSQP(options=optOptions)
        opt = IPOPT(options={'print_level': 5, 'tol': 1e-5, 'acceptable_tol': 1e-4, 'output_file': f'IPOPT_{id}.out'})
        #opt = NSGA2(options={'PopSize': 2000, 'xinit': 1})
        #opt = PSQP()
        #opt = ALPSO()#options={'parallelType': 'EXT'})
        #opt = CONMIN()
        sol = opt(optProb, sens=self.grad_obj_function, sensMode='pgc', storeHistory=f'history_{id}.txt', storeSens=f'sens_{id}.txt')
        with open(f'solution_ipopt_{id}.pickle', 'wb') as handle:
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
        #print(ret_dict)
        return ret_dict, False
    
    def grad_obj_function(self, xdict, funcs):
        #basis = torch.eye(26)
        # gradients_x = np.zeros((26, self.dim_x * (self.T+1)))
        # gradients_u = np.zeros((26, self.dim_u * (self.T)))
        torch.set_grad_enabled(True)
        DeltaX = torch.tensor(xdict['DeltaX'], requires_grad=True)
        DeltaU = torch.tensor(xdict['DeltaU'], requires_grad=True)

        

        # import pdb
        # pdb.set_trace()
        # res = self._compute_problem(DeltaX, DeltaU)
        # def get_vjp(v):
        #     return torch.autograd.grad(res, (DeltaX, DeltaU), v)
        # start = time.time()

        # jacobian_vmap = functorch.vmap(get_vjp)(basis)


        # for i in range(26):
        #     res.backward(basis[i], retain_graph=True)
        #     gradients_x[i] = DeltaX.grad.detach().numpy()
        #     gradients_u[i] = DeltaU.grad.detach().numpy()
        #     DeltaX.grad.zero_()
        #    DeltaU.grad.zero_()
            
        # print(f'Done in {time.time()-start}')
        # pdb.set_trace()
        start = time.time()
        gradsX, gradsU = torch.autograd.functional.jacobian(self._compute_problem, (DeltaX, DeltaU), vectorize=True)
        print(f'Done in {time.time()-start}')
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
        DeltaAB = AB_poisoned - torch.tensor(self.AB_unpoisoned)
        objective = torch.linalg.norm(DeltaAB, 2)

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
        stacked_constraints = torch.vstack((
            torch.abs(1-c_r) - 0.1, 
            torch.abs(1-c_c) - 0.1)
            ).flatten()

        if gradients is False:
            return -objective.item(), stacked_constraints.detach().numpy()
        else:
            return torch.hstack((-objective, stacked_constraints))


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
    attack.compute(0)
    attack.compute(1)
    attack.compute(2)

    