import numpy as np
import torch
import torch.nn as nn
import gym
from utilities import ReplayBuffer


# d_env = gym.make('CartPole-v1', render_mode='human')
d_env = gym.make('CartPole-v1')


def random_steps(n_episodes=10):
    for _ in range(n_episodes):
        d_env.reset()
        score = 0.
        while True:
            d_env.render()
            a = d_env.action_space.sample()
            _, r, d, t, _ = d_env.step(a)
            score += r
            if d or t:
                break
        print('Score: %.3f' % score)
    d_env.close()


# random_steps()


class DiscreteAgent(nn.Module):
    def __init__(self, in_d:int, h_shape:np.array, out_d:int, dtype=torch.float64):
        super().__init__()
        torch.set_default_dtype(dtype)
        self.layers = nn.ModuleList()
        current_in = in_d
        for h in h_shape:
            self.layers.append(nn.Linear(current_in, h))
            current_in = h
        self.output = nn.Linear(h_shape[-1], out_d)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output(x)
        return x


class Dql:
    def __init__(self, env: gym.Env, hidden_shape:np.array, alpha:float=1/1_000, gamma:float=99/100):
        self.env = env
        self.n_s = self.env.observation_space.shape[0]
        self.n_a = self.env.action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.buffer = ReplayBuffer()
        self.main_nn = DiscreteAgent(self.n_s, hidden_shape, self.n_a)
        self.target_nn = DiscreteAgent(self.n_s, hidden_shape, self.n_a)
        self.target_hard_update()

    def target_hard_update(self):
        self.target_nn.load_state_dict(self.main_nn.state_dict())


dql = Dql(gym.make('CartPole-v1'), np.array([8, 16, 32, 16]))
s = d_env.reset()[0]
t = torch.unsqueeze(torch.tensor(s, dtype=torch.float64), dim=0)

# ----------------------------------------------------------------------------


class PolicyGradient:
    pass


class ActorCriticNn:
    pass


class TD3PG:
    pass



