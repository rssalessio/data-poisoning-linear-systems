import numpy as np
import scipy.signal as scipysig
from typing import Tuple
import scipy.stats as scipystats
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
np.set_printoptions(formatter={'float_kind':"{:.2f}".format})
A = np.array([[0.7, 0], [0.1, 0.3]])
B = np.array([[0.5], [0.1]])

dim_x = 2
dim_u = 1


def collect_data(steps: int, std_u: float, std_w: float) -> Tuple[np.ndarray, np.ndarray]:
    U = np.zeros((dim_u, steps))
    X = np.zeros((dim_x, steps + 1))
    W = np.zeros((dim_x, steps))
    X[:, 0] = np.random.normal(size=(dim_x))

    for i in range(steps):
        U[:, i] = std_u * np.random.normal(size=(dim_u))
        W[:, i] = std_w * np.random.normal(size=(dim_x))
        X[:, i+1] = A @ X[:, i] +  np.squeeze(B * U[:, i]) + W[:, i]#(low=-0.1, high=0.1)#(size=(dim_x))

    return X, U, W

std_w = 1e-1
std_u = 1
sample_size = 1000


X, U, W = collect_data(sample_size, std_u, std_w)

attack_amplitude = 0.1
attack_X = 0* np.random.uniform(low=0*-attack_amplitude, high=attack_amplitude, size=(dim_x, sample_size+1))
attack_U = np.random.uniform(low=-attack_amplitude+10, high=attack_amplitude, size=(dim_u, sample_size))
attack_idxs = np.random.choice(sample_size, size=900, replace=False)
attack_U[:, attack_idxs] = 0


Xtilde = X + attack_X
Utilde = U + attack_U

D = np.vstack([X[:,:-1], U])
Dtilde = np.vstack([Xtilde[:,:-1], Utilde])

AB = X[:,1:] @ np.linalg.pinv(D)
ABtilde = Xtilde[:,1:] @ np.linalg.pinv(Dtilde)
Delta = -B @ attack_U @ np.linalg.pinv(Dtilde)



Dtilde = torch.tensor(Dtilde, requires_grad=False)
Xtilde = torch.tensor(Xtilde, requires_grad=False)

iterations = [100, 500, 1000, 2000]


for num_iterations in iterations:
    eta = 5e-2
    w = torch.tensor(np.ones(sample_size)/sample_size, requires_grad=True)
    optim = torch.optim.Adam([w], lr=1e-2)
    for iteration in tqdm(range(num_iterations)):
        W = torch.diag(w)
        theta_w = torch.linalg.inv(Dtilde @ W @ Dtilde.T) @ Dtilde @ W @ Xtilde[:, 1:].T
        loss = torch.dot(w, torch.linalg.norm(Xtilde[:, 1:] - theta_w.T @ Dtilde, dim=0, ord=2))
        optim.zero_grad()
        loss.backward()
        with torch.no_grad():
            w *= torch.exp(-(eta / (iteration + 1)**0.5) * w.grad)
            w /= w.sum()
            #eta *= 0.99
    indicators = np.zeros(sample_size)
    indicators[attack_idxs] = 1

    w_indi = w.detach().numpy() > 1e-3
    print(f"{num_iterations} - {np.abs((indicators-w_indi)).sum() / indicators.sum()}")

plt.plot(indicators)
plt.plot(w.detach().numpy() > 1e-3)
plt.show()