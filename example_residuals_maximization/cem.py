from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
from math import ceil
from utils import is_positive_definite, is_symmetric, mean_cov

class Population(ABC):
    """
    Population class, used by the CEM method
    """
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
    """
    Define a Gaussian population used by the CEM method
    """    
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
        # Sample a number of points from the population
        return np.random.multivariate_normal(self.means, cov=self.covariance, size=(num_points,))

    def update(self, elite_results: List[Tuple[float, np.ndarray]], smoothed_update: float = 0.5, regularization: float = 1e-3):
        # Update the population according to the results
        results, points = list(zip(*elite_results))
        new_mean, new_cov = mean_cov(np.array(points))
        # Smooth update of means and covariance
        self.means = (1 - smoothed_update) * self.means + smoothed_update * new_mean
        self.covariance = (1 - smoothed_update) * self.covariance + smoothed_update * (new_cov + regularization * np.eye(self.dim))


def evaluate_population(
    fun: Callable[[np.ndarray], float],
    population: Population,
    num_points: int,
    elite_fraction: float = 0.2) -> Tuple[Population, List[Tuple[float, np.ndarray]]]:
    """Evaluate a population on a function and updates the population according to the 
    best results

    Args:
        fun (Callable[[np.ndarray], float]): function to evaluate
        population (Population): population used by cem
        num_points (int): number of points to evlauate
        elite_fraction (float, optional): Elite fraction. Defaults to 0.2.

    Returns:
        Tuple[Population, List[Tuple[float, np.ndarray]]]: first element is the new population, the second element
        are the best results (sorted)
    """    
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
    elite_fraction: float = 0.2) -> Tuple[float, np.ndarray]:
    """Optimize a function using the CEM method

    Args:
        fun (Callable[[np.ndarray], float]): function to optimize
        population (Population): Population to use to optimize the function
        num_points (int): Number of points to evaluate in each iteration
        max_iterations (int, optional): maximum number of iterations. Defaults to 1000.
        rel_tol (float, optional): relative tolerance. Defaults to 1e-4.
        abs_tol (float, optional): absolute tolerance. Defaults to 1e-6.
        elite_fraction (float, optional): Fraction of best results used by the CEM method to update
            the population. Defaults to 0.2.

    Returns:
        Tuple[float, np.ndarray]: Tuple (best results, best parameters)
    """    
    assert max_iterations > 0, 'Number of max iterations needs to be positive'

    best_result = -np.infty
    best_params = None

    for epoch in range(max_iterations):
        # Evaluate and update population
        population, results = evaluate_population(fun, population, num_points=num_points, elite_fraction=elite_fraction)
        _best = results[-1]
        
        # Check if the results have improved
        if _best[0] > best_result:
            prev_best_result = best_result
            best_result = _best[0]
            best_params = _best[1]

            if np.isclose(best_result, prev_best_result, rtol = rel_tol, atol = abs_tol):
                break
    
    return best_result, best_params
