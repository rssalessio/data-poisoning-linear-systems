from random import random
import numpy as np
import scipy.signal as scipysig
from typing import Tuple, NamedTuple
import scipy.integrate as integrate
import cvxpy as cp
import scipy.stats as stats
import multiprocessing as mp
import dccp
from scipy.stats import chi2, norm
from momentchi2 import hbe
from utils import collect_data, residuals_variance_test, TestStatistics, confidence_interval_signal, ResultsData
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mayavi import mlab

np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
trueAB = np.hstack((sys.A, sys.B))
dim_x, dim_u = sys.B.shape
T = 200
std_w = 1e-1
std_u = 1


delta = 1e-1

def project(w, d):
    return d*np.dot(w, d)/np.dot(d, d)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)


def evaluate_attack(Xm: np.ndarray, Xp: np.ndarray, U: np.ndarray, DeltaX: np.ndarray, DeltaU: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tildeXm = Xm + DeltaX[:, :-1]
    tildeXp = Xp + DeltaX[:, 1:]
    tildeU = U + DeltaU
    Dtilde = np.vstack((tildeXm, tildeU))

    ABtilde = tildeXp @ np.linalg.pinv(Dtilde)

    residuals = tildeXp - ABtilde @ Dtilde        

    return ABtilde-trueAB, residuals

X, U, W = collect_data(T, std_u, std_w, sys)
Xp, Xm = X[:, 1:], X[:, :-1]
D = np.vstack((Xm, U))
AB = Xp @ np.linalg.pinv(D)

true_residuals = Xp - AB @ D

# Optimal attack
DeltaX = cp.Variable((dim_x, T+1))
DeltaU = cp.Variable((dim_u, T))

objective = cp.norm(DeltaX[:, 1:] - AB @ cp.vstack((DeltaX[:, :-1], DeltaU)), 'fro')
objective += 2*cp.trace(true_residuals @ (DeltaX[:, 1:] - AB @ cp.vstack((DeltaX[:, :-1], DeltaU))).T)
constraints = [
    cp.norm(DeltaX, 'fro') <= delta * np.linalg.norm(X, 'fro'),
    cp.norm(DeltaU, 'fro') <= delta * np.linalg.norm(U, 'fro') 
]
problem = cp.Problem(cp.Maximize(objective), constraints)
result = problem.solve(method='dccp', ccp_times=10, solver=cp.MOSEK)

stdX = (delta * np.linalg.norm(X, 'fro')) / np.sqrt(T* dim_x)
stdU = (delta * np.linalg.norm(U, 'fro')) / np.sqrt(T* dim_u)


DeltaX.value = np.random.uniform(size=X.shape, low=-1, high = 1)
DeltaU.value = np.random.uniform(size=U.shape, low=-1, high=1)

DeltaX.value *= delta * np.linalg.norm(X, 'fro') / np.linalg.norm(DeltaX.value, 'fro')
DeltaU.value *= delta * np.linalg.norm(U, 'fro') / np.linalg.norm(DeltaU.value, 'fro')

TrueDeltaAB,_=evaluate_attack(Xm, Xp, U, 0*DeltaX.value, 0*DeltaU.value)
DeltaAB,_=evaluate_attack(Xm, Xp, U, DeltaX.value, DeltaU.value)
Psi = np.vstack((Xm , U))
PsiTilde = np.vstack((Xm + DeltaX.value[:, :-1], U + DeltaU.value))
DeltaW = DeltaX.value[:, 1:] - sys.A @ DeltaX.value[:, :-1] - sys.B @ DeltaU.value

def compute_lower_bound(W, DeltaW, V, Psi):
    d1 = np.linalg.svd(W)[1].min()
    d2 = np.linalg.svd(DeltaW)[1].min()
    Vpsi = [V[i].reshape(dim_x, dim_x+dim_u) @ Psi for i in range(dim_x*(dim_x+dim_u))]
    angles = np.abs(np.cos(np.array([angle_between( (W+DeltaW).flatten(), Vpsi[i].flatten()) for i in range(dim_x*(dim_x+dim_u))])))
    lb = angles * (d1 + d2) / np.array([np.linalg.norm(Vpsi[i], 'fro') for i in range(dim_x*(dim_x+dim_u))])
    return lb

random_directions = [AB.flatten()]
random_directions.extend([np.random.normal(size=(dim_x*(dim_x+dim_u))).flatten() for _ in range(dim_x*(dim_x+dim_u)-1)])
V = gram_schmidt(random_directions)

errors = np.sqrt(np.array([np.dot(V[i], DeltaAB.flatten()) for i in range(dim_x*(dim_x+dim_u))]) ** 2)
errors2 = np.sqrt(np.array([np.dot(V[i], TrueDeltaAB.flatten()) for i in range(dim_x*(dim_x+dim_u))]) ** 2)

lb = compute_lower_bound(W, DeltaW, V, PsiTilde)
plt.plot(errors, label='poisoned')
plt.plot(errors2, label='original')
plt.plot(lb, label='lb')
plt.legend()
plt.show()
import pdb
pdb.set_trace()
# Gaussian attack

stdX = (delta * np.linalg.norm(X, 'fro')) / np.sqrt(T* dim_x)
stdU = (delta * np.linalg.norm(U, 'fro')) / np.sqrt(T* dim_u)




opt_attack_DeltaX = DeltaX.value
opt_attack_DeltaU = DeltaU.value
gauss_attack_DeltaX = np.random.normal(size=X.shape) * stdX
gauss_attack_DeltaU = np.random.normal(size=U.shape) * stdU

print(np.linalg.norm(evaluate_attack(Xm, Xp, U, opt_attack_DeltaX, opt_attack_DeltaU)[0], 2))
print(np.linalg.norm(evaluate_attack(Xm, Xp, U, gauss_attack_DeltaX, gauss_attack_DeltaU)[0], 2))

dim_params_x = np.prod(opt_attack_DeltaX.shape)
dim_params_u = np.prod(opt_attack_DeltaU.shape)
dim_params = dim_params_x + dim_params_u

dir = np.random.uniform(size=(2,dim_params))

# dir[1, :dim_params_x] = dir[1, :dim_params_x] / np.dot(dir[1, :dim_params_x], dir[1, :dim_params_x])
# dir[1, dim_params_x:] = dir[1, dim_params_x:] / np.dot(dir[1, dim_params_x:], dir[1, dim_params_x:])

dir[1, :dim_params_x] = dir[1, :dim_params_x] - project(dir[1, :dim_params_x], dir[0, :dim_params_x])
dir[1, dim_params_x:] = dir[1, dim_params_x:] - project(dir[1, dim_params_x:], dir[0, dim_params_x:])

dir[0, :dim_params_x] = dir[0, :dim_params_x] / np.dot(dir[0, :dim_params_x], dir[0, :dim_params_x])
dir[0, dim_params_x:] = dir[0, dim_params_x:] / np.dot(dir[0, dim_params_x:], dir[0, dim_params_x:])
dir[1, :dim_params_x] = dir[1, :dim_params_x] / np.dot(dir[1, :dim_params_x], dir[1, :dim_params_x])
dir[1, dim_params_x:] = dir[1, dim_params_x:] / np.dot(dir[1, dim_params_x:], dir[1, dim_params_x:])
# dir[1] = dir[1] / np.dot(dir[1], dir[1])
print(angle_between(dir[0, :dim_params_x], dir[1, :dim_params_x]))
print(angle_between(dir[0, dim_params_x:], dir[1, dim_params_x:]))


atkX_dir0 = dir[0, :dim_params_x].reshape(DeltaX.shape)
atkX_dir1 = dir[1, :dim_params_x].reshape(DeltaX.shape)
atkU_dir0 = dir[0, dim_params_x:].reshape(DeltaU.shape)
atkU_dir1 = dir[1, dim_params_x:].reshape(DeltaU.shape)

atkX_dir1 = atkX_dir1


#d.mul_(w.norm()/(d.norm() + 1e-10))

opt_atkX_dir0 = atkX_dir0 #np.linalg.norm(opt_attack_DeltaX,'fro') * atkX_dir0/np.linalg.norm(atkX_dir0,'fro')
opt_atkX_dir1 = atkX_dir1 #np.linalg.norm(opt_attack_DeltaX,'fro') * atkX_dir1/np.linalg.norm(atkX_dir1,'fro')
opt_atkU_dir0 = atkU_dir0 #np.linalg.norm(opt_attack_DeltaU,'fro') * atkU_dir0/np.linalg.norm(atkU_dir0,'fro')
opt_atkU_dir1 = atkU_dir1 #np.linalg.norm(opt_attack_DeltaU,'fro') * atkU_dir1/np.linalg.norm(atkU_dir1,'fro')

gauss_atkX_dir0 = atkX_dir0 #np.linalg.norm(gauss_attack_DeltaX) * atkX_dir0/np.linalg.norm(atkX_dir0,'fro')
gauss_atkX_dir1 = atkX_dir1 #np.linalg.norm(gauss_attack_DeltaX) * atkX_dir1/np.linalg.norm(atkX_dir1,'fro')
gauss_atkU_dir0 = atkU_dir0 #np.linalg.norm(gauss_attack_DeltaU) * atkU_dir0/np.linalg.norm(atkU_dir0,'fro')
gauss_atkU_dir1 = atkU_dir1 #np.linalg.norm(gauss_attack_DeltaU) * atkU_dir1/np.linalg.norm(atkU_dir1,'fro')

@np.vectorize
def eval_optimal(x, y):
    newX = opt_attack_DeltaX + x * opt_atkX_dir0 + y * opt_atkX_dir1
    newU = opt_attack_DeltaU + x * opt_atkU_dir0 + y * opt_atkU_dir1
    newX = delta * np.linalg.norm(X, 'fro') * newX / np.linalg.norm(newX, 'fro')
    newU = delta * np.linalg.norm(U, 'fro') * newU / np.linalg.norm(newU, 'fro')
    newAB, _ = evaluate_attack(Xm, Xp, U, newX, newU)
    return np.linalg.norm(newAB, 2)

@np.vectorize
def eval_gauss(x, y):
    newX = gauss_attack_DeltaX + x * gauss_atkX_dir0 + y * gauss_atkX_dir1
    newU = gauss_attack_DeltaU + x * gauss_atkU_dir0 + y * gauss_atkU_dir1
    newX = delta * np.linalg.norm(X, 'fro') * newX / np.linalg.norm(newX, 'fro')
    newU = delta * np.linalg.norm(U, 'fro') * newU / np.linalg.norm(newU, 'fro')
    newAB, _ = evaluate_attack(Xm, Xp, U, newX, newU)
    return np.linalg.norm(newAB, 2)


SCALE = 1000
DELTA = 1e-2 * SCALE
MIN_X = -1 * SCALE
MIN_Y = -1* SCALE
MAX_X = 1* SCALE
MAX_Y = 1* SCALE
x = np.arange(MIN_X, MAX_X, DELTA)
y = np.arange(MIN_Y, MAX_Y, DELTA)
X, Y = np.meshgrid(x, y)
Zopt = eval_optimal(X,Y)
Zgauss = eval_gauss(X,Y)

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Zopt)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Simplest default with labels')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Zgauss)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Simplest default with labels')
plt.show()

# Make the plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Zopt, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# fig = mlab.figure(bgcolor=(1,1,1))
# # note the transpose in surf due to different conventions compared to meshgrid
# su = mlab.surf(X.T, Y.T, Zgauss.T, warp_scale='auto')

# # manually set viridis for the surface
# cmap_name = 'viridis'
# cdat = np.array(cm.get_cmap(cmap_name,256).colors)
# cdat = (cdat*255).astype(int)
# su.module_manager.scalar_lut_manager.lut.table = cdat

# mlab.show()
 
# Make the plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Zopt, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()