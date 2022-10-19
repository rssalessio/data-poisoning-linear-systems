from pyexpat import features
from typing import List, Optional
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, features_dim: int, arch_net: Optional[List[int]] = [256, 32], recurrent_layer: Optional[nn.Module] = nn.GRU, activation_fn: Optional[nn.Module] = nn.SELU):
        super().__init__()
        self.features_dim = features_dim
        input_dim = features_dim

        self.gru_1 = nn.GRU(input_dim, arch_net[0], bias=True, batch_first=True, num_layers=2)
        self.act_1 = activation_fn()

        self.fc = nn.Linear(arch_net[0], 1)
        self.act_fc = nn.Sigmoid()
        # for lsize in arch_net:
        #     networks.append(
        #     networks.append(activation_fn())
        #     input_dim = lsize
        
        # networks.append(nn.Linear(input_dim, 1))
        # networks.append(nn.ReLU())
        
        # self.network = nn.Sequential(*networks)

    def forward(self, data: torch.Tensor):
        h1 = self.act_1(self.gru_1(data.float())[0])
        out = self.act_fc(self.fc(h1))
        return out

        # return logprob
        # return -self.network(data.float())