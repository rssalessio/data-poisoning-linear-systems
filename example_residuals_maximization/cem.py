from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
from math import ceil
from scipy.linalg._fblas import dger, dgemm

def is_positive_definite(x: np.ndarray, atol: float = 1e-9):
    return np.all(np.linalg.eigvals(x) > atol)

def is_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def mean_cov(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    See https://groups.google.com/g/scipy-user/c/FpOU4pY8W2Y
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

class Population(ABC):
    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @abstractmethod
    def sample(self, num_points: int) -> List[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> Population:
        raise NotImplementedError

    @abstractmethod
    def update(self, elite_points: List[Tuple[float, np.ndarray]], *args):
        raise NotImplementedError


class GaussianPopulation(Population):
    means: np.ndarray
    covariance: np.ndarray

    def __init__(self, means: np.ndarray, covariance: np.ndarray):
        super().__init__(means.shape[0])
        assert len(means.shape) <= 1, 'Means should be a scalar or a vector'
        assert means.shape[0] == covariance.shape[0] and means.shape[0] == covariance.shape[1], 'Means and std_devs should have the same dimensionality'
        assert is_positive_definite(covariance) and is_symmetric(covariance), 'Covariance needs to be PSD and symmetric'

        self.means = means
        self.covariance = covariance

    def clone(self) -> GaussianPopulation:
        return GaussianPopulation(self.means.copy(), self.covariance.copy())

    def sample(self, num_points: int) -> List[np.ndarray]:
        return np.random.multivariate_normal(self.means, cov=self.covariance, size=(num_points,))

    def update(self, elite_results: List[Tuple[float, np.ndarray]], smoothed_update: float = 0.5, regularization: float = 1e-3):
        results, points = list(zip(*elite_results))
        new_mean, new_cov = mean_cov(np.array(points))
        self.means = (1 - smoothed_update) * self.means + smoothed_update * new_mean
        self.covariance = (1 - smoothed_update) * self.covariance + smoothed_update * (new_cov + regularization * np.eye(self.dim))


def evaluate_population(
    fun: Callable[[np.ndarray], float],
    population: Population,
    num_points: int,
    elite_fraction: float = 0.2):
    assert elite_fraction > 0 and elite_fraction < 1, 'Elite fraction needs to be in (0,1)'

    # Sample population
    points = population.sample(num_points)

    # Evaluate points
    results = list(map(fun, points))

    # Sort results
    sorted_results = sorted(zip(results, points), key=lambda x: x[0])

    # Compute elite population
    elite_results = sorted_results[-ceil(elite_fraction * num_points):]

    # Update population
    new_population = population.clone()
    new_population.update(elite_results)

    return new_population, sorted_results

def optimize(
    fun: Callable[[np.ndarray], float],
    population: Population,
    num_points: int,
    max_iterations: int = 1000,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-6,
    elite_fraction: float = 0.2):
    assert max_iterations > 0, 'Number of max iterations needs to be positive'

    best_result = -np.infty
    best_params = None

    for epoch in range(max_iterations):
        population, results = evaluate_population(fun, population, num_points=num_points, elite_fraction=elite_fraction)
        _best = results[-1]
        
        if _best[0] > best_result:
            prev_best_result = best_result
            best_result = _best[0]
            best_params = _best[1]

            if np.isclose(best_result, prev_best_result, rtol = rel_tol, atol = abs_tol):
                break
    
    return best_result, best_params
