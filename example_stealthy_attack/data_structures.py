import numpy as np
import torch
from typing import NamedTuple, List
from utils import TestStatistics


class UnpoisonedDataInfo(NamedTuple):
    # Data structure used to collect statistics from unpoisoned data
    X: np.ndarray
    U: np.ndarray
    R: np.ndarray
    num_lags: int
    SigmaW: np.ndarray
    eigs_SigmaW: List[float]
    AB_unpoisoned: np.ndarray
    residuals_unpoisoned: np.ndarray
    whiteness_statistics_test_unpoisoned: TestStatistics
    residuals_variance_test_unpoisoned: TestStatistics

    @property
    def dim_x(self):
        return self.X.shape[0]
    
    @property
    def dim_u(self):
        return self.U.shape[0]
    
    @property
    def T(self):
        return self.U.shape[1]



class AttackData(NamedTuple):
    # Information about an attack and its statistics
    unpoisoned_data: UnpoisonedDataInfo
    DeltaU: torch.Tensor
    DeltaX: torch.Tensor
    D_poisoned: torch.Tensor
    AB_poisoned: torch.Tensor
    residuals_poisoned: torch.Tensor
    DeltaAB: torch.Tensor
    R: List[torch.Tensor]


class AttackOptimizationInfo(NamedTuple):
    unpoisoned_data: UnpoisonedDataInfo
    loss: List[float] = []
    regularizer: List[float] = []
    residuals_norm: List[float] = []
    delta_norm: List[float] = []
    AB_poisoned: List[np.ndarray] = []
    DeltaX: List[np.ndarray] = []
    DeltaU: List[np.ndarray] = []
    residuals_poisoned: List[np.ndarray] = []
    whiteness_statistics_test_poisoned: List[TestStatistics] = []
    residuals_variance_test_poisoned: List[TestStatistics] = []