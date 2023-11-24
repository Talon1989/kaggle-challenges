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


class StateActionAgent(nn.Module):
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


class StateValueAgent(nn.Module):
    def __init__(self, in_d:int, h_shape:np.array, dtype=torch.float64):
        super().__init__()
        torch.set_default_dtype(dtype)
        self.layers = nn.ModuleList()
        current_in = in_d
        for h in h_shape:
            self.layers.append(nn.Linear(current_in, h))
            current_in = h
        self.output = nn.Linear(h_shape[-1], 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output(x)
        return x


class Actor(nn.Module):
    def __init__(self, in_d: int, h_shape: np.array, out_d: int, dtype=torch.float64):
        super().__init__()
        torch.set_default_dtype(dtype)
        self.layers = nn.ModuleList()
        current_in = in_d
        for h in h_shape:
            self.layers.append(nn.Linear(current_in, h))
            current_in = h
        self.output = nn.Linear(h_shape[-1], out_d)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.softmax(self.output(x))
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
        self.main_nn = StateActionAgent(self.n_s, hidden_shape, self.n_a)
        self.optimizer = torch.optim.Adam(params=self.main_nn.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.target_nn = StateActionAgent(self.n_s, hidden_shape, self.n_a)
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
    def __init__(self, env: gym.Env, hidden_shape:np.array, alpha:float=1/1_000, gamma:float=99/100):
        self.env = env
        self.n_s = self.env.observation_space.shape[0]
        self.n_a = self.env.action_space.n
        self.gamma = gamma
        self.buffer = ReplayBuffer()
        self.actor = Actor(self.n_s, hidden_shape, self.n_a)
        self.optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()

    def _custom_loss(self, states, actions, n_returns):
        self.actor.train()
        distribution = torch.distributions.Categorical(probs=self.actor(states))
        log_prob_actions = distribution.log_prob(actions)
        return torch.sum(-log_prob_actions * n_returns)

    def _choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float64).unsqueeze(dim=0)
        with torch.no_grad():
            probabilities = self.actor(s)
            distribution = torch.distributions.Categorical(probs=probabilities)
            a = distribution.sample().detach().numpy()
            return a[0]

    def _store_transition(self, s, a, r):
        self.buffer.remember(s, a, r, None, None)

    def _learn(self):
        states, actions, rewards, _, _ = self.buffer.get_buffer(
            batch_size=self.buffer.get_buffer_size(), randomized=False, cleared=True)
        returns = []
        cumulative_value = 0
        for r in rewards[::-1]:
            cumulative_value = r + self.gamma * cumulative_value
            returns.append(cumulative_value)
        returns = np.array(returns[::-1])
        norm_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        states = torch.tensor(states, dtype=torch.float64)
        actions = torch.tensor(actions, dtype=torch.float64)
        norm_returns = torch.tensor(norm_returns, dtype=torch.float64)
        self.optimizer.zero_grad()
        self.actor.eval()
        loss = self._custom_loss(states, actions, norm_returns)
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
                self._store_transition(s, a, r)
                if d or t:
                    break
                s = s_
            self._learn()
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if ep % 10 == 0:
                print('Episode %d | Avg Score: %.3f' % (ep, avg_scores[-1]))
        return self


pgm = PolicyGradient(d_env, np.array([16, 16, 32, 16]))
# pgm.fit()


class ActorCriticNn:
    def __init__(self, env: gym.Env, hidden_actor: np.array, hidden_critic: np.array,
                 alpha: float = 1 / 1_000, beta: float = 1/750, gamma: float = 99/100):
        self.env = env
        self.n_s = self.env.observation_space.shape[0]
        self.n_a = self.env.action_space.n
        self.gamma = gamma
        self.buffer = ReplayBuffer()
        self.actor = Actor(self.n_s, hidden_actor, self.n_a)
        self.critic = StateValueAgent(self.n_s, hidden_critic)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=alpha)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=beta)
        self.actor_criterion = self._custom_loss
        self.critic_criterion = nn.MSELoss()

    def _custom_loss(self, states, actions, returns):
        with torch.no_grad():
            critic_state_values = self.critic(states)
        advantages = returns - critic_state_values
        distribution = torch.distributions.Categorical(self.actor(states))
        log_prob_actions = distribution.log_prob(actions)
        return torch.sum(- log_prob_actions * advantages)

    def _choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float64).unsqueeze(dim=0)
        with torch.no_grad():
            probabilities = self.actor(s)
            distribution = torch.distributions.Categorical(probs=probabilities)
            a = distribution.sample().detach().numpy()
            return a[0]

    def _store_transition(self, s, a, r):
        self.buffer.remember(s, a, r, None, None)

    def _learn(self):
        states, actions, rewards, _, _ = self.buffer.get_buffer(
            batch_size=self.buffer.get_buffer_size(), randomized=False, cleared=True
        )
        cum_rewards = 0
        returns = []
        for r in rewards[::-1]:
            cum_rewards = r + self.gamma * cum_rewards
            returns.append(cum_rewards)
        returns = torch.tensor(returns, dtype=torch.float64).reshape([-1, 1])
        states = torch.tensor(states, dtype=torch.float64)
        actions = torch.tensor(actions, dtype=torch.float64)
        actor_loss = self._actor_learn(states, actions, returns)
        critic_loss = self._critic_learn(states, returns)
        return actor_loss, critic_loss

    def _actor_learn(self, states, actions, returns):
        self.actor_optimizer.zero_grad()
        self.actor.train()
        loss = self.actor_criterion(states, actions, returns)
        loss.backward()
        self.actor_optimizer.step()
        return loss.detach().numpy()

    def _critic_learn(self, states, returns):
        self.critic_optimizer.zero_grad()
        self.critic.train()
        critic_state_values = self.critic(states)
        loss = self.critic_criterion(critic_state_values, returns)
        loss.backward()
        self.critic_optimizer.step()
        return loss.detach().numpy()

    def fit(self, n_episodes=2_000):
        scores, avg_scores = [], []
        for ep in range(1, n_episodes+1):
            score = 0
            s = self.env.reset()[0]
            while True:
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                score += r
                self._store_transition(s, a, r)
                if d or t:
                    break
                s = s_
            self._learn()
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if ep % 10 == 0:
                print('Episode %d | Avg Score: %.3f' % (ep, avg_scores[-1]))
        return self


actor_critic = ActorCriticNn(d_env, np.array([16, 16, 32, 16]), np.array([16, 16, 32, 16]))
# actor_critic.fit()


c_env = gym.make('Pendulum-v1')


class ActorContinuous(nn.Module):
    def __init__(self, in_d: int, h_shape: np.array, out_d: int, dtype=torch.float64, scalar: float = 2.):
        super().__init__()
        # torch.set_default_dtype(dtype)
        self.layers = nn.ModuleList()
        current_in = in_d
        for h in h_shape:
            self.layers.append(nn.Linear(current_in, h))
            current_in = h
        self.output = nn.Linear(h_shape[-1], out_d)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.scalar = scalar

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.tanh(self.output(x))
        x = torch.mul(x, self.scalar).to(torch.float64)
        return x


# environment requires continuous action space
class TD3PG:
    def __init__(self, env: gym.Env, hidden_actor: np.array, hidden_critic: np.array,
                 alpha: float = 1 / 1_000, beta: float = 1/750, gamma: float = 99/100,
                 batch_size: int = 64, actor_update: int = 2, tau: float = 1/100,
                 noise: float = 3., noise_decay: float = 99/100):
        self.env = env
        self.n_s = self.env.observation_space.shape[0]
        self.n_a = self.env.action_space.shape[0]  # hardcoded
        self.max_action = self.env.action_space.high[0]    # hardcoded
        self.gamma = gamma
        self.actor_update_rate = actor_update
        self.tau = tau
        self.noise = noise
        self.noise_decay = noise_decay
        self.buffer = ReplayBuffer()
        self.batch_size = batch_size
        self.actor = ActorContinuous(self.n_s, hidden_actor, self.n_a, self.max_action)
        self.t_actor = ActorContinuous(self.n_s, hidden_actor, self.n_a, self.max_action)
        self.actor_soft_update()
        self.critic_1 = StateValueAgent((self.n_s + self.n_a), hidden_critic)
        self.critic_2 = StateValueAgent((self.n_s + self.n_a), hidden_critic)
        self.t_critic_1 = StateValueAgent((self.n_s + self.n_a), hidden_critic)
        self.t_critic_2 = StateValueAgent((self.n_s + self.n_a), hidden_critic)
        self.critic_soft_update()
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=alpha)
        self.critic_1_optimizer = torch.optim.Adam(params=self.critic_1.parameters(), lr=beta)
        self.critic_2_optimizer = torch.optim.Adam(params=self.critic_2.parameters(), lr=beta)

    def actor_soft_update(self, tau=1.):
        for param, t_param in zip(self.actor.parameters(), self.t_actor.parameters()):
            t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)

    def critic_soft_update(self, tau=1.):
        for param, t_param in zip(self.critic_1.parameters(), self.t_critic_1.parameters()):
            t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
        for param, t_param in zip(self.critic_2.parameters(), self.t_critic_2.parameters()):
            t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)

    def _choose_action(self, s):
        pass

    def _store_transition(self, s, a, r, s_, d):
        self.buffer.remember(s, a, r, s_, int(d))

    def _update_actor(self):
        pass

    def _update_critic(self):
        pass

    def _learn(self):
        pass

    def fit(self, n_episodes=2_000):
        pass


td3 = TD3PG(c_env, np.array([16, 32, 16]), np.array([16, 32, 16]))
s = c_env.reset()[0]
t = torch.tensor(s, dtype=torch.float64).unsqueeze(dim=0)
actor = td3.actor
critic = td3.critic_1
print(
    actor(t)
)
print(
    critic(torch.concat([t, actor(t)], dim=1))
)


class SAC:
    pass
