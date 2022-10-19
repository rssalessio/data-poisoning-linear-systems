import numpy as np
import scipy.signal as scipysig
from typing import Tuple
import matplotlib.pyplot as plt
from momentchi2 import hbe
import scipy.integrate as integrate
import scipy.stats as stats
from scipy.stats import chi2, norm
from utils import collect_data, correlate
np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
dim_x, dim_u = sys.B.shape
T = 500


std_w = 1e-1
std_u = 1

NUM_SIMS = 200

values = []
for i in range(NUM_SIMS):

    X, U, W = collect_data(T, std_u, std_w, sys)
    Xp, Xm = X[:, 1:], X[:, :-1]
    D = np.vstack((Xm, U))
    AB = Xp @ np.linalg.pinv(D)
    true_residuals = Xp - AB @ D 
        
        
    num_lags = 10#-dim_x-dim_u
    R = correlate(true_residuals, T-dim_x-dim_u)
    Z1 = 0
    Z2 = np.zeros(num_lags)
    Z3 = np.zeros(num_lags)

    InvCovEst = np.linalg.inv(R[0])
    for i in range(1, num_lags+1):
        Z1 += T*np.trace(R[i].T @ InvCovEst @ R[i] @ InvCovEst)
        Z2[i-1] = T*np.trace(R[i].T @ InvCovEst @ R[i] @ InvCovEst)
        Z3[i-1] = np.sqrt(T)*np.trace(R[i] @ InvCovEst)


    alpha = 0.05
    df =  dim_x**2

    cr_ub_1=chi2.ppf(q=1-alpha/2, df=num_lags * (dim_x**2))
    cr_lb_1=chi2.ppf(q=alpha/2, df=num_lags * (dim_x**2))


    cr_ub_2=chi2.ppf(q=1-alpha/2, df=(dim_x**2))
    cr_lb_2=chi2.ppf(q=alpha/2, df=(dim_x**2))

    cr_ub_3=norm.ppf(1-alpha/2)
    cr_lb_3=norm.ppf(alpha/2)
    print(f'{Z1} - {cr_lb_1}-{cr_ub_1} - {chi2.cdf(Z1, df=num_lags * (dim_x**2))}')
    # print(f'{Z2} - {cr_lb_2}-{cr_ub_2}')
    # print(f'{Z3} - {cr_lb_3}-{cr_ub_3}')
    values.append(chi2.cdf(Z1, df=num_lags * (dim_x**2)))
plt.hist(values)
plt.show()