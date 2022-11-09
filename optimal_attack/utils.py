import numpy as np
import scipy.signal as scipysig
import torch
from scipy.stats import chi2
from typing import Tuple, List, NamedTuple
from momentchi2 import hbe
import jax.numpy as jnp
import time
import torch.nn.functional as F

class CollectedData(NamedTuple):
    X: np.ndarray
    U: np.ndarray
    W: np.ndarray
    std_u: float
    std_w: float

class TestStatistics(NamedTuple):
    statistics: float
    pvalue: float

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

    return CollectedData(X, U, W, std_u, std_w)

def correlate(x: np.ndarray, num_lags: int) -> np.ndarray:
    n, T = x.shape
    R = np.zeros((num_lags,n,n))

    for m in range(num_lags):
        for i in range(T-m):
            R[m] += x[:,i:i+1] @ x[:, i+m:i+m+1].T

    
    return R/T

def jax_correlate(x: jnp.ndarray, num_lags: int) -> jnp.ndarray:
    n, T = x.shape
    R = jnp.zeros((num_lags,n,n))

    for m in range(num_lags):
        for i in range(T-m):
            R = R.at[m].set(R[m] + x[:,i:i+1] @ x[:, i+m:i+m+1].T)
            #R[m] += x[:,i:i+1] @ x[:, i+m:i+m+1].T

    return R/T



def torch_correlate(x: torch.Tensor, num_lags: int) -> List[torch.Tensor]:
    n, T = x.shape
    R = []

    # start = time.time()
    # for m in range(num_lags):
    #     y = 0
    #     for i in range(T-m):
    #         y = y + x[:,i:i+1] @ x[:, i+m:i+m+1].T
    #     R.append(y/T)
    # end= time.time()-start

    R = []
    for m in range(num_lags):
        R.append(F.conv1d(x.unsqueeze(1), x[:, m:].unsqueeze(1))[:,:,0]/T)

    return R

def whiteness_test(residuals: np.ndarray, num_lags: int) -> float:
    dim_x, T = residuals.shape
    R = correlate(residuals, num_lags)
    df = (T-dim_x-1) * (dim_x ** 2)
    R0Inv = np.linalg.inv(R[0])
    statistics = T * np.sum([np.trace(R[x].T @ R0Inv @ R[x] @ R0Inv) for x in range(1, num_lags)])
    q = chi2.cdf(statistics, df)
    return statistics


def pvalue_whiteness_test(R: np.ndarray, num_lags: int, T: int) -> Tuple[float,float]:
    R_lags, dim_x, _ = R.shape
    assert R_lags > num_lags, f"R has been computed only for {R_lags} lags, not for {num_lags} lags."
    df = (T-dim_x-1) * (dim_x ** 2)
    R0Inv = np.linalg.inv(R[0])
    statistics = T * np.sum([np.trace(R[x].T @ R0Inv @ R[x] @ R0Inv) for x in range(1, num_lags + 1)])
    p = chi2.cdf(statistics, df)
    return TestStatistics(statistics, p)


def pvalue_residuals_variance_test(residuals: np.ndarray, dim_u: int, sigma_eigs: List[float]) -> TestStatistics:
    """
    Compute the pvalue under H0 that the residuals follow a linear combination of chi_squared distribution

    :param residuals: a (dim_x, T) matrix with the residuals
    :param dim_u: dimension of the input
    :param sigma_eigs: list of eigenvalues of the noise covariance matrix
    :return pval: pvalue
    """
    assert isinstance(sigma_eigs, list), "Sigma_eigs needs to be a list"
    dim_x, T = residuals.shape
    assert len(sigma_eigs) == dim_x, "Sigma_eigs needs to have dim_x values"
    eigs = sigma_eigs * (T-dim_x-dim_u)
    fro_norm_squared  = np.linalg.norm(residuals, 'fro' ) ** 2
    p = hbe(coeff=eigs, x=fro_norm_squared)
    return TestStatistics(fro_norm_squared, 2*min(p, 1-p))
