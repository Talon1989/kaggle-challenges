import numpy as np
import torch
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # this is to deal with a PyCharm - matplolib issue
# matplotlib.use('Agg')  # this is to deal with a PyCharm - matplolib issue
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, random_split


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1 + torch.exp(-x))


def one_hot_transformation(y: np.array) -> np.array:
    """
    :param y: label encoded 1D np.array
    :return:
    """
    assert len(y.shape) == 1
    n_unique = len(np.unique(y))
    one_hot = np.zeros(shape=[y.shape[0], n_unique])
    for idx, val in enumerate(y):
        one_hot[idx, int(val)] = 1
    return one_hot


class ReplayBuffer:
    def __init__(self, max_size=1_000):
        self.max_size = max_size
        self.states, self.actions, self.rewards, self.states_, self.dones = (
            [],
            [],
            [],
            [],
            [],
        )

    def get_buffer_size(self):
        assert len(self.states) == len(self.actions) == len(self.rewards)
        return len(self.actions)

    def remember(self, s, a, r, s_, done):
        if len(self.states) > self.max_size:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.states_[0]
            del self.dones[0]
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.states_.append(s_)
        self.dones.append(done)

    def clear(self):
        self.states, self.actions, self.rewards, self.states_, self.dones = (
            [],
            [],
            [],
            [],
            [],
        )

    def get_buffer(
        self, batch_size, randomized=True, cleared=False, return_bracket=False
    ):
        assert batch_size <= self.max_size + 1
        indices = np.arange(self.get_buffer_size())
        if randomized:
            np.random.shuffle(indices)
        buffer_states = np.squeeze([self.states[i] for i in indices][0:batch_size])
        buffer_actions = [self.actions[i] for i in indices][0:batch_size]
        buffer_rewards = [self.rewards[i] for i in indices][0:batch_size]
        buffer_states_ = np.squeeze([self.states_[i] for i in indices][0:batch_size])
        buffer_dones = [self.dones[i] for i in indices][0:batch_size]
        if cleared:
            self.clear()
        if return_bracket:
            for i in range(batch_size):
                buffer_actions[i] = np.array(buffer_actions[i])
                buffer_rewards[i] = np.array([buffer_rewards[i]])
                buffer_dones[i] = np.array([buffer_dones[i]])
            return (
                np.array(buffer_states),
                np.array(buffer_actions),
                np.array(buffer_rewards),
                np.array(buffer_states_),
                np.array(buffer_dones),
            )
            # return tuple(np.array(buffer_states)), tuple(np.array(buffer_actions)), tuple(np.array(buffer_rewards)), tuple(np.array(buffer_states_)), tuple(np.array(buffer_dones))
        return (
            np.array(buffer_states),
            np.array(buffer_actions),
            np.array(buffer_rewards),
            np.array(buffer_states_),
            np.array(buffer_dones),
        )

    def __len__(self):
        return len(self.actions)
