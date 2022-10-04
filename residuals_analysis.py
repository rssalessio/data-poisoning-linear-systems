import numpy as np
import scipy.signal as scipysig
from typing import Tuple
import matplotlib.pyplot as plt
from momentchi2 import hbe
import scipy.integrate as integrate
import scipy.stats as stats
np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
dim_x, dim_u = sys.B.shape
T = 200

def collect_data(steps: int, std_u: float, std_w: float, sys: scipysig.StateSpace) -> Tuple[np.ndarray, np.ndarray]:
    dim_x, dim_u = sys.B.shape
    U = np.zeros((dim_u, steps))
    X = np.zeros((dim_x, steps + 1))
    W = np.zeros((dim_x, steps))
    X[:, 0] = np.random.normal(size=(dim_x))

    for i in range(steps):
        U[:, i] = std_u * np.random.normal(size=(dim_u))
        W[:, i] = std_w * np.random.normal(size=(dim_x))
        X[:, i+1] = sys.A @ X[:, i] +  np.squeeze(sys.B * U[:, i]) + W[:, i]

    return X, U, W

def correlate(x: np.ndarray, num_lags: int):
    n, T = x.shape
    R = np.zeros((num_lags,n,n))

    for m in range(num_lags):
        for i in range(m, T):
            R[m] += x[:,i:i+1] @ x[:, i-m:i-m+1].T

    return R/ T

std_w = 1e-1
std_u = 1




NUM_SIMS = 2000

# def compute_Z(alpha, Z):
#     hbe(coeff=[std_w**2 ] * dim_x* (T-dim_x-dim_u), x=Z)
#     stats.norm.pdf(alpha)
val_true = []
val_poisoned = []

amplitudes = np.geomspace(0.05,.5, 20)
val_true = np.zeros((NUM_SIMS, len(amplitudes)))
val_poisoned = np.zeros((NUM_SIMS, len(amplitudes)))

for i in range(NUM_SIMS):
    X, U, W = collect_data(T, std_u, std_w, sys)
    Xp, Xm = X[:, 1:], X[:, :-1]
    D = np.vstack((Xm, U))
    AB = Xp @ np.linalg.pinv(D)
    true_residuals = Xp - AB @ D 
    test_variance = np.linalg.norm(true_residuals, 'fro') ** 2
    for ampl_idx, amplitude in enumerate(amplitudes):        
        attack_amplitude = std_w * amplitude
        DeltaX = np.random.uniform(low=-attack_amplitude, high=attack_amplitude, size=(dim_x, T+1))
        DeltaU = np.random.uniform(low=-attack_amplitude, high=attack_amplitude, size=(dim_u, T))


        tildeXm = Xm + DeltaX[:, :-1]
        tildeXp = Xp + DeltaX[:, 1:]
        tildeU = U + DeltaU

        
        Dtilde = np.vstack((tildeXm, tildeU))
        
        ABtilde = tildeXp @ np.linalg.pinv(Dtilde)

        
        poisoned_residuals = tildeXp - ABtilde @ Dtilde        
        test_poisoned_variance = np.linalg.norm(poisoned_residuals, 'fro') ** 2

        val_true[i, ampl_idx] =hbe(coeff=[std_w**2 ] * dim_x* (T-dim_x-dim_u), x=test_variance)
        val_poisoned[i, ampl_idx]=hbe(coeff=[std_w**2 ] * dim_x* (T-dim_x-dim_u), x=test_poisoned_variance)

values = val_poisoned > 0.95
mu = values.mean(0)
std = values.std(0)
plt.plot(amplitudes, mu)
plt.fill_between(amplitudes, mu - 1.96 * std / np.sqrt(NUM_SIMS), mu + 1.96 * std / np.sqrt(NUM_SIMS),alpha=0.3)
#plt.xscale('log')
plt.grid()
plt.show()
