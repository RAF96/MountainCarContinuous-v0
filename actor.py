import torch
import torch.nn.functional as F
from torch import nn

L1_SIZE = 5
L2_SIZE = 5


class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.l1 = nn.Linear(state_size, L1_SIZE)
        self.l2 = nn.Linear(L1_SIZE, L2_SIZE)
        self.l3 = nn.Linear(L2_SIZE, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.l1.weight.data.uniform_(-3e-3, 3e-3)
        self.l2.weight.data.uniform_(-3e-3, 3e-3)
        self.l3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = state
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)

        return torch.tanh(x)
