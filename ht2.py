import numpy as np
import scipy.signal as scipysig
from typing import Tuple
import scipy.stats as scipystats
import matplotlib.pyplot as plt

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

res_orig = X[:, 1:] - AB @ D
res_partial =  Xtilde[:, 1:] - AB @ Dtilde
res_partial2 =  Xtilde[:, 1:] - ABtilde @ Dtilde
res_partial_norm = np.linalg.norm(res_partial, axis=0, ord=2)
res_orig_norm = np.linalg.norm(res_orig, axis=0, ord=2)
res_partial_norm2 = np.linalg.norm(res_partial2, axis=0, ord=2)


L = Dtilde.T @ np.linalg.inv(Dtilde @ Dtilde.T) @ Dtilde
plt.plot(np.diag(L))
indicators = np.ones(len(np.diag(L))) *np.max(np.diag(L)) * 0.5
indicators[attack_idxs] = 0
plt.plot(indicators)
plt.show()
import pdb
pdb.set_trace()
print(f'{res_partial_norm.sum()} vs {res_partial_norm2.sum()} - {res_orig_norm.sum()}')

plt.plot(res_orig_norm, label='orig')
plt.plot(res_partial_norm, label='Tilde with true AB')
plt.plot(res_partial_norm2, label='Poisoned')
plt.legend()
plt.show()

exit(-1)
S = np.random.choice(sample_size, size= sample_size//3)

data_full = np.vstack([Xtilde[:,:-1], Utilde])
for _ in range(10):
    
    
    data = np.vstack([Xtilde[:,S], Utilde[:, S]])
    param = Xtilde[:, S+1] @ np.linalg.pinv(data)
    res_orig = X[:, 1:] - AB @ D
    res_partial =  Xtilde[:, 1:] - param @ data_full
    res_full = Xtilde[:, 1:] - ABtilde @ data_full
    res_partial_norm = np.linalg.norm(res_partial, axis=0, ord=2)
    res_full_norm = np.linalg.norm(res_full, axis=0, ord=2)
    res_orig_norm = np.linalg.norm(res_orig, axis=0, ord=2)

    print(f"{res_partial_norm.mean()} - {res_full_norm.mean()} - {res_orig_norm.mean()} - {np.sum(S >= 400)} - {np.linalg.norm(param - ABtilde)}  - {np.linalg.norm(param - AB)} - {np.linalg.norm(ABtilde - AB)}")
    if np.linalg.norm(param - AB) < np.linalg.norm(ABtilde - AB):
        S = np.argsort(res_partial_norm)[:sample_size//3]
        print('Choosing!')
    else:
        S = np.random.choice(sample_size, size= sample_size//3)
        print('Sampling!')
    #break


print(param)
print(AB)
print(ABtilde)
# cond_orig = np.hstack([-np.eye(dim_x), A,B]) @ np.vstack([X[:, 1:], X[:, :-1], U])
# cond_pois = np.hstack([-np.eye(dim_x), A+Delta[:,:dim_x], B +  Delta[:, dim_x:]]) @ np.vstack([Xtilde[:, 1:], Xtilde[:, :-1], Utilde])
# cond_pois2 = np.hstack([-np.eye(dim_x), AB+Delta]) @ np.vstack([Xtilde[:, 1:], Xtilde[:, :-1], Utilde])

# dx=np.hstack([np.eye(dim_x), -(AB+Delta)]) @ np.vstack([Xtilde[:, 10][:,None], Xtilde[:, 9][:,None], Utilde[:,9][:,None]])

# V = np.hstack([np.eye(dim_x), -AB-Delta]) @ np.vstack([attack_X[:, 1:], attack_X[:, :-1], attack_U])
# V -= Delta @ D
# print(cond_orig)
# print(cond_pois)
# print(cond_pois2)
# print(V)
# import pdb
# pdb.set_trace()
# # print(AB-ABtilde)
# # print(f"{cond_orig.max()} - {cond_pois.max()}")
# # print(f"{(cond_orig + W).max()} - {(cond_pois + W).max()}")
# # print((cond_orig+W).mean(axis=1))
# # print((cond_pois+W).mean(axis=1))
# # Z = np.hstack([-np.eye(dim_x), A,B]) @ np.vstack([attack_X[:, 1:], attack_X[:, :-1], attack_U])+np.hstack([-0*np.eye(dim_x), ABtilde-AB]) @ np.vstack([Xtilde[:, 1:], Xtilde[:, :-1], Utilde])
# # print(np.hstack([-np.eye(dim_x), A,B]) @ np.vstack([attack_X[:, 1:], attack_X[:, :-1], attack_U])+np.hstack([-0*np.eye(dim_x), ABtilde-AB]) @ np.vstack([Xtilde[:, 1:], Xtilde[:, :-1], Utilde]))
# # print(np.hstack([-np.eye(dim_x), ABtilde-AB]) @ np.vstack([attack_X[:, 1:], attack_X[:, :-1], attack_U]))
# ResOrig = np.hstack([np.eye(dim_x), -A,-B]) @ np.vstack([X[:, 1:], X[:, :-1], U])
# ResPois = np.hstack([np.eye(dim_x), -ABtilde]) @ np.vstack([Xtilde[:, 1:], Xtilde[:, :-1], Utilde])
# plt.plot(ResPois[0,:],label='poisd')
# plt.plot(ResOrig[0,:],label='orig')

# plt.legend()
# plt.show()
# exit(-1)
# # RESIDUALS ANALYSIS
# R0 = np.linalg.norm(X[:,1:] - AB @ D, ord=2, axis=0)
# R1 = np.linalg.norm(Xtilde[:,1:] -ABtilde @ Dtilde, ord=2, axis=0)

# print(f"{R0.sum()} - {R1.sum()}")
# plt.plot((X[0,1:]- (AB @ D)[0]).T )
# plt.plot((Xtilde[0,1:] - (ABtilde @ Dtilde)[0]).T)
# plt.grid()
# plt.show()