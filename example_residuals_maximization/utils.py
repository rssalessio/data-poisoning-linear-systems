import numpy as np
import scipy.signal as scipysig
from typing import Tuple, List, NamedTuple
from momentchi2 import hbe
from scipy.stats import t
from scipy.linalg._fblas import dger, dgemm


def is_positive_definite(x: np.ndarray, atol: float = 1e-9) -> bool:
    """Check if a matrix is positive definite

    Args:
        x (np.ndarray): matrix
        atol (float, optional): absolute tolerance. Defaults to 1e-9.

    Returns:
        bool: Returns True if the matrix is positive definite
    """    
    return np.all(np.linalg.eigvals(x) > atol)

def is_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 0) -> bool:
    """Check if a matrix is symmetric

    Args:
        a (np.ndarray): matrix to check
        rtol (float, optional): relative tolerance. Defaults to 1e-05.
        atol (float, optional): absolute tolerance. Defaults to 1e-08.

    Returns:
        bool: returns True if the matrix is symmetric
    """    
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def mean_cov(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns mean and covariance of a matrix
    See https://groups.google.com/g/scipy-user/c/FpOU4pY8W2Y

    Args:
        X (np.ndarray): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Mean,Covariance) tuple
    """   
    n, p = X.shape
    m = X.mean(axis=0)
    # covariance matrix with correction for rounding error
    # S = (cx'*cx - (scx'*scx/n))/(n-1)
    # Am Stat 1983, vol 37: 242-247.
    cx = X - m
    scx = cx.sum(axis=0)
    scx_op = dger(-1.0/n,scx,scx)
    S = dgemm(1.0, cx.T, cx.T, beta=1.0,
    c = scx_op, trans_a=0, trans_b=1, overwrite_c=1)
    S[:] *= 1.0/(n-1)
    return m, S.T


class TestStatistics(NamedTuple):
    statistics: float
    p_right: float
    p_left: float

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


def residuals_variance_test(residuals: np.ndarray, dim_u: int, sigma_eigs: List[float]) -> TestStatistics:
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
    return TestStatistics(fro_norm_squared, p, 1-p)

def confidence_interval_signal(signal: np.ndarray, n_sims: int, c: float = 0.95) -> Tuple[np.ndarray,np.ndarray, np.ndarray]:
    mu = signal.mean(0)
    std = signal.std(0)
    c = t.ppf(c + (1-c)/2, df=n_sims-1)
    return mu, mu - c * std /np.sqrt(n_sims), mu + c * std / np.sqrt(n_sims)

        
class ResultsData(object):
    def __init__(self, X: np.ndarray, U: np.ndarray, W: np.ndarray, Sigma: np.ndarray, TrueAB: np.ndarray, deltas: np.ndarray):
        self.TrueAB = TrueAB
        self.X = X
        self.U = U
        self.W = W
        self.Sigma = Sigma
        
        n_deltas = len(deltas)

        self.opt_DeltaX = np.zeros((n_deltas,) + X.shape)
        self.gauss_DeltaX = np.zeros((n_deltas,) + X.shape)
        self.cem_DeltaX = np.zeros((n_deltas,) + X.shape)
        self.opt_DeltaU = np.zeros((n_deltas,) + U.shape)
        self.gauss_DeltaU = np.zeros((n_deltas,) + U.shape)
        self.cem_DeltaU = np.zeros((n_deltas,) + U.shape)
        self.deltas = deltas