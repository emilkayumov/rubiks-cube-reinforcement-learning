import random
from collections import namedtuple

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import cube

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    def __init__(self, n_space, n_action):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_space, 2048)
        self.layer2 = nn.Linear(2048, 512)
        self.layer3 = nn.Linear(512, 128)
        self.head = nn.Linear(128, n_action)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.head(x)

    def get_action_probabilities(self, observation, tau=1):
        return Categorical(logits=self.forward(observation * tau).detach())

    def sample_action(self, observation):
        return self.get_action_probabilities(observation).sample()

    def select_action(self, observation):
        return self.forward(observation).detach().max(0)[1]


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ValueNetwork(nn.Module):
    def __init__(self, n_space):
        super(ValueNetwork, self).__init__()
        self.layer1 = nn.Linear(n_space, 2048)
        self.layer2 = nn.Linear(2048, 512)
        self.layer3 = nn.Linear(512, 128)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.head(x)


class SimpleValue:
    def __init__(self):
        pass

    @staticmethod
    def get_value(state):
        return np.mean(state == cube.FINAL_STATE)
