import torch
import torch.nn.functional as F
from torch import nn

L1_SIZE = 20
L2_SIZE = 10


class Critic(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.l1 = nn.Linear(state_size, L1_SIZE)
        self.l2 = nn.Linear(L1_SIZE + action_size, L2_SIZE)
        self.l3 = nn.Linear(L2_SIZE, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.l1.weight.data.uniform_(-3e-3, 3e-3)
        self.l2.weight.data.uniform_(-3e-3, 3e-3)
        self.l3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = self.l1(state)
        xs = F.leaky_relu(xs)
        x = torch.cat((xs, action), dim=1)
        x = self.l2(x)
        x = F.leaky_relu(x)
        return self.l3(x)
