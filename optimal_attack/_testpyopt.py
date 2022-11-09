# First party modules
from pyoptsparse import SLSQP, Optimization, IPOPT


class ComputeAttack(object):
    def __init__(self):


# rst begin objfunc
def objfunc(xdict):
    x = xdict["xvars"]
    funcs = {}
    funcs["obj"] = -x[0] * x[1] * x[2]
    conval = [0] * 2
    conval[0] = x[0] + 2.0 * x[1] + 2.0 * x[2] - 72.0
    conval[1] = -x[0] - 2.0 * x[1] - 2.0 * x[2]
    funcs["con"] = conval
    fail = False

    return funcs, fail


# rst begin optProb
# Optimization Object
optProb = Optimization("TP037 Constraint Problem", objfunc)

# rst begin addVar
# Design Variables
optProb.addVarGroup("xvars", 3, "c", lower=[0, 0, 0], upper=[42, 42, 42], value=10)

# rst begin addCon
# Constraints
optProb.addConGroup("con", 2, lower=None, upper=0.0)

# rst begin addObj
# Objective
optProb.addObj("obj")

# rst begin print
# Check optimization problem
print(optProb)

# rst begin OPT
# Optimizer
optOptions = {"IPRINT": -1}
opt = SLSQP(options=optOptions)

#opt = IPOPT(options={'print_level': 1})
# rst begin solve
# Solve
sol = opt(optProb, sens="FD")

# rst begin check
# Check Solution
print(sol)


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

    s = 2 * num_lags

    D = np.vstack((X[:, :-1], U))
    AB_unpoisoned = X[:, 1:] @ np.linalg.pinv(D)
    residuals_unpoisoned = X[:, 1:] - AB_unpoisoned @ D
    R_unpoisoned = correlate(residuals_unpoisoned, T-dim_x-dim_u)
    SigmaW = np.diag([STD_W ** 2] * X.shape[0])
    norm_unpoisoned_residuals = np.linalg.norm(residuals_unpoisoned, 'fro') ** 2

    R = correlate(residuals_unpoisoned, s+1)
    R0inv_unpoisoned = np.linalg.inv(R[0])
    norm_unpoisoned_correlation_terms = [np.linalg.norm(R[idx+1] @ R0inv_unpoisoned, 'fro') ** 2 for idx in range(s)]
    
    

    #constraints = NonlinearConstraint(compute_constraints, -np.infty * np.ones(s+1), np.zeros(s+1), jac=grad_compute_constraints, keep_feasible=True)
    constraints = [{
            'type': 'ineq',
            'fun': lambda *args: compute_gradients(*args, gradients=False),
            'jac': lambda *args: compute_gradients(*args, gradients=True),
            'args': (X, U, AB_unpoisoned, norm_unpoisoned_residuals, R, R0inv_unpoisoned, norm_unpoisoned_correlation_terms)
    }]


    

    res = minimize(
        fun= objective_function,
        x0=np.zeros(dim_x * (T+1) + dim_u * T),
        args=(X, U, AB_unpoisoned),
        bounds=[(-0.3,0.3) for i in range(dim_x * (T+1) + dim_u * T)],
        jac= lambda *args: objective_function(*args, gradients=True),
        method = 'SLSQP',
        constraints=constraints,
        options={'disp': True, 'maxiter': 100})
    print(res)
    import pdb
    pdb.set_trace()

    