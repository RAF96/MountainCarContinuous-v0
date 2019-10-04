import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binom, uniform
from torch import optim

from actor import Actor
from critic import Critic
from memory import Memory
from settings import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 20
BUFFER_SIZE = 1000
PROBABILITY_RAND_STEP = 0.2
TAU = 0.001


class Agent:

    def __init__(self, state_size, action_size):
        self._state_size = state_size
        self._action_size = action_size

        # Actor network
        self._actor_local = Actor(state_size, action_size).to(device)
        self._actor_target = Actor(state_size, action_size).to(device)
        self._actor_optimizer = optim.Adam(self._actor_local.parameters())

        # Critic network
        self._critic_local = Critic(state_size, action_size).to(device)
        self._critic_target = Critic(state_size, action_size).to(device)
        self._critic_optimizer = optim.Adam(self._critic_local.parameters())

        # Memory
        self._memory = Memory(BUFFER_SIZE)

        # Do equal weights
        self.hard_update(self._actor_local, self._actor_target)
        self.hard_update(self._critic_local, self._critic_target)

    def step(self, state, action, reward, next_state, done):
        self._memory.push((state, action, reward, next_state, done))

        if len(self._memory) > BATCH_SIZE:
            samples = self._memory.sample(BATCH_SIZE)
            self.learn(samples)

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)

        if binom.rvs(1, PROBABILITY_RAND_STEP):
            action = np.ndarray((1,), buffer=np.array(uniform(-1, 1).rvs()))
        else:
            self._actor_local.eval()
            with torch.no_grad():
                action = self._actor_local(state).cpu().data.numpy()
            self._actor_local.train()

        return np.clip(action, -1, 1)

    def hard_update(self, local, target):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(local_param.data)

    def soft_update(self, local, target, tau):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def learn(self, samples):

        states, actions, rewards, next_states, dones = samples

        actions_next = self._actor_target(next_states)
        Q_targets_next = self._critic_target(next_states, actions_next)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        Q_expected = self._critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        actions_pred = self._actor_local(states)
        actor_loss = -self._critic_local(states, actions_pred).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        self.soft_update(self._critic_local, self._critic_target, TAU)
        self.soft_update(self._actor_local, self._actor_target, TAU)

    def save(self):
        torch.save(self._actor_local.state_dict(), ACTOR_PATH)
        torch.save(self._critic_local.state_dict(), CRITIC_PATH)

    def load(self):
        self._actor_local = torch.load(ACTOR_PATH)
        self._critic_local = torch.load(CRITIC_PATH)
