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
    def __init__(self, env: gym.Env, hidden_shape:np.array, batch_size=64,
                 alpha:float=1/1_000, gamma:float=99/100, e_decay:float=99/100):
        self.env = env
        self.n_s = self.env.observation_space.shape[0]
        self.n_a = self.env.action_space.n
        self.gamma = gamma
        self.epsilon = 1.
        self.e_decay = e_decay
        self.buffer = ReplayBuffer()
        self.batch_size = batch_size
        self.main_nn = DiscreteAgent(self.n_s, hidden_shape, self.n_a)
        self.optimizer = torch.optim.Adam(params=self.main_nn.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.target_nn = DiscreteAgent(self.n_s, hidden_shape, self.n_a)
        self._target_hard_update()

    def _target_hard_update(self):
        self.target_nn.load_state_dict(self.main_nn.state_dict())

    def _choose_action(self, s):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float64), dim=0)
            a = torch.argmax(self.main_nn(s)).detach().numpy()
            return a

    def _store(self, s, a, r, s_, d):
        self.buffer.remember(s, a, r, s_, d)

    def _learn(self):
        if self.buffer.get_buffer_size() < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.buffer.get_buffer(
            batch_size=self.batch_size, randomized=True, cleared=False)
        states = torch.tensor(states, dtype=torch.float64)
        actions = torch.tensor(actions).reshape([-1, 1])
        rewards = torch.tensor(rewards).reshape([-1, 1])
        states_ = torch.tensor(states_, dtype=torch.float64)
        dones = torch.tensor(dones).reshape([-1, 1])
        self.optimizer.zero_grad()
        self.main_nn.train()
        pp = self.main_nn(states)
        y_preds = torch.gather(input=pp, dim=1, index=actions)
        with torch.no_grad():
            next_, _ = torch.max(self.target_nn(states_), dim=1)
            next_ = torch.reshape(next_, [-1, 1])
        y_hat = rewards + self.gamma * next_ * (1 - dones)
        loss = self.criterion(y_preds, y_hat)
        loss.backward()
        self.optimizer.step()

    def fit(self, n_episodes=2_000):
        scores, avg_scores = [], []
        for ep in range(1, n_episodes+1):
            score = 0
            s = self.env.reset()[0]
            while True:
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                score += r
                self._store(s, a, r, s_, int(d))
                self._learn()
                if d or t:
                    break
                s = s_
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if ep % 10 == 0:
                print('Episode %d | Avg Score: %.3f | Epsilon: %.3f' % (ep, avg_scores[-1], self.epsilon))
                if self.epsilon > 1/10:
                    self.epsilon = self.epsilon * self.e_decay
                self._target_hard_update()
        return self


dql = Dql(gym.make('CartPole-v1'), np.array([8, 16, 32, 16]))
# s = d_env.reset()[0]
# t = torch.unsqueeze(torch.tensor(s, dtype=torch.float64), dim=0)
# dql.main_nn(t)

# ----------------------------------------------------------------------------


class PolicyGradient:
    pass


class ActorCriticNn:
    pass


class TD3PG:
    pass



