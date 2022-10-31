import numpy as np
import scipy.signal as scipysig
from typing import Tuple
import matplotlib.pyplot as plt
from momentchi2 import hbe
import scipy.integrate as integrate
import scipy.stats as stats
from scipy.stats import chi2, norm
from tqdm import tqdm
from utils import collect_data, correlate, pvalue_whiteness_test
np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
dim_x, dim_u = sys.B.shape
T = 200


std_w = 1e-1
std_u = 1

NUM_SIMS = 200
num_lags = 50#-dim_x-dim_u
values = []
values2 = []
q = np.zeros((NUM_SIMS, num_lags, 10))
for i in tqdm(range(NUM_SIMS)):

    X, U, W = collect_data(T, std_u, std_w, sys)
    Xp, Xm = X[:, 1:], X[:, :-1]
    D = np.vstack((Xm, U))
    AB = Xp @ np.linalg.pinv(D)
    true_residuals = Xp - AB @ D 
        
    # T=1500 , numlags=10 -> plot 1
    
    R = correlate(true_residuals, T)
    Z1 = 0
    Z2 = 0
    #Z2 = np.zeros(num_lags)
    #Z3 = np.zeros(num_lags)
    

    InvCovEst = np.linalg.inv(R[0])
    for j in range(1, num_lags+1):
        Z1 = T*np.trace(R[j].T @ InvCovEst @ R[j] @ InvCovEst)
        Z2 = T*np.trace((R[j] @ InvCovEst).T @ R[j] @ InvCovEst)
        #Z2[i-1] = T*np.trace(R[i].T @ InvCovEst @ R[i] @ InvCovEst)
        #Z3[i-1] = np.sqrt(T)*np.trace(R[i] @ InvCovEst)
        # LOL = T * (np.linalg.norm(R[j] @ InvCovEst, 'fro')**2)
        # import pdb
        # pdb.set_trace()

    # alpha = 0.05
    # df =  dim_x**2

    # cr_ub_1=chi2.ppf(q=1-alpha/2, df=num_lags * (dim_x**2))
    # cr_lb_1=chi2.ppf(q=alpha/2, df=num_lags * (dim_x**2))


    # cr_ub_2=chi2.ppf(q=1-alpha/2, df=(dim_x**2))
    # cr_lb_2=chi2.ppf(q=alpha/2, df=(dim_x**2))

    # cr_ub_3=norm.ppf(1-alpha/2)
    # cr_lb_3=norm.ppf(alpha/2)
    # print(f'{Z1} - {cr_lb_1}-{cr_ub_1} - {chi2.cdf(Z1, df=num_lags * (dim_x**2))}')
    # print(f'{Z2} - {cr_lb_2}-{cr_ub_2}')
    # print(f'{Z3} - {cr_lb_3}-{cr_ub_3}')
    #values.append(pvalue_whiteness_test(R, 10, T).pvalue)
        q[i, j-1, 0]=chi2.cdf(Z1, df= (dim_x**2))
        q[i, j-1, 1]=chi2.cdf(Z1, df=max(1,(j-dim_x)) * (dim_x**2))
        q[i, j-1, 2]=chi2.cdf(Z1, df=max(1,(j-dim_u)) * (dim_x**2))
        q[i, j-1, 3]=chi2.cdf(Z1, df=max(1,(j-dim_x-dim_u)) * (dim_x**2))
        q[i, j-1, 4]=chi2.cdf(Z1, df=max(1,(j-dim_x*(dim_x+dim_u))) * (dim_x**2))
        q[i, j-1, 5]=chi2.cdf(Z2, df=(dim_x**2))
        q[i, j-1, 6]=chi2.cdf(Z2, df=max(1,(j-dim_x)) * (dim_x**2))
        q[i, j-1, 7]=chi2.cdf(Z2, df=max(1,(j-dim_u)) * (dim_x**2))
        q[i, j-1, 8]=chi2.cdf(Z2, df=max(1,(j-dim_x-dim_u)) * (dim_x**2))
        q[i, j-1, 9]=chi2.cdf(Z2, df=max(1,(j-dim_x*(dim_x+dim_u))) * (dim_x**2))
    #q2=chi2.cdf(Z1, df=(num_lags- dim_x* (dim_x+dim_u)) * (dim_x**2))
    #q2=chi2.cdf(Z2, df=(num_lags-dim_x-dim_u )  * (dim_x**2))
    #print(q)
    #values.append(q)
    #values2.append(q2)
import pdb
pdb.set_trace()
fig, ax = plt.subplots(1,q.shape[-1])
for i in range(q.shape[-1]):
    ax[i].hist(q[:,i], density=True)
plt.show()