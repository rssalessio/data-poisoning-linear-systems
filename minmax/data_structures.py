from os import system
import numpy as np
import torch
from typing import NamedTuple, List
from utils import TestStatistics
import scipy.signal as scipysig

class SimulationParameters(NamedTuple):
    SEED: int
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    DT: float
    NUM_SIMS: int
    T: int
    STD_U: float
    STD_W: float
    MAX_ITERATIONS: int
    EPOCHS_ATTACK: int
    EPOCHS_DISCRIMINATOR: int
    LR_ATTACK: float
    LR_DISCRIMINATOR: float
    MAX_GRAD_NORM: float
    PENALTY_REGULARIZER: float
    SEQUENCE_LENGTH: int
    

    @property
    def dim_x(self):
        return self.B.shape[0]

    @property
    def dim_u(self):
        return self.B.shape[1]

class UnpoisonedDataInfo(NamedTuple):
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