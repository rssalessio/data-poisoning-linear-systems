import torch
from typing import List, Optional
from data_structures import AttackData


class ConstraintModule(torch.nn.Module):
    def __init__(self, min_val: Optional[float], max_val: Optional[float]):
        super().__init__()
        assert min_val is not None or max_val is not None, 'min val or max val need to be different from None'
        self.min_val = min_val
        self.max_val = max_val

    def _forward(self, x: AttackData) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: AttackData) -> torch.Tensor:
        max_term, min_term = 0, 0
        val = self._forward(x)
        if self.max_val is not None:
            max_term = torch.log(self.max_val - val)
        if self.min_val is not None:
            min_term = torch.log(val - self.min_val)
        return -(max_term + min_term)


class WhitenessConstraint(ConstraintModule):
    def __init__(self, T: int, num_lags: int, min_val: Optional[float], max_val: Optional[float]):
        super().__init__(min_val, max_val)
        self.T = T
        self.num_lags = num_lags

    def _forward(self, x: AttackData) -> torch.Tensor:
        R = x.R
        R0Inv = torch.linalg.inv(R[0])
        return self.T * torch.sum(torch.stack([torch.trace(R[x].T @ R0Inv @ R[x] @ R0Inv) for x in range(1, self.num_lags + 1)]))

class ResidualsVarianceConstraint(ConstraintModule):
    def __init__(self, min_val: Optional[float], max_val: Optional[float]):
        super().__init__(min_val, max_val)

    def _forward(self, x: AttackData) -> torch.Tensor:
        return torch.linalg.norm(x.residuals_poisoned, 'fro' ) ** 2
