import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset


class QFunc(nn.Module):
    def __init__(self, action_space=4, observation_space=8, hidden_space_1=512, hidden_space_2=512):
        super(QFunc, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(observation_space, hidden_space_1),
            nn.ReLU(),
            nn.Linear(hidden_space_1, hidden_space_2),
            nn.ReLU(),
            nn.Linear(hidden_space_2, action_space)
        )

    def forward(self, observations):
        return self.ffn(observations)

    @property
    def device(self):
        return next(self.parameters()).device


class ReplayBufferDataset(Dataset):
    def __init__(self, buffer, device='cuda'):
        self.buffer = buffer
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        s, a, r, s_p, done = self.buffer[index]
        return (
            torch.tensor(s, dtype=torch.float32).to(self.device),
            torch.tensor(a, dtype=torch.long).to(self.device),
            torch.tensor(r, dtype=torch.float32).to(self.device),
            torch.tensor(s_p, dtype=torch.float32).to(self.device),
            torch.tensor(done, dtype=torch.bool).to(self.device)
        )


class LunarLanderAgent:
    def __init__(self, qfunc, env):
        self.qfunc = qfunc
        self.env = env

    @torch.no_grad()
    def get_action(self, observations, validation=False, eps=0.1):
        if not validation and np.random.random() <= eps:
            return self.env.action_space.sample()
        tensor = torch.tensor(observations).unsqueeze(0).to(self.qfunc.device)
        action = torch.argmax(self.qfunc(tensor), dim=1).item()
        return action

    @torch.no_grad()
    def play_episode(self, buffer, validation=False, eps=0.):
        total_reward = 0
        curr_observations, _ = self.env.reset()
        while True:
            action = self.get_action(curr_observations, validation=validation, eps=eps)
            next_observations, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            buffer.append((curr_observations.copy(), action, reward, next_observations.copy(), terminated))
            if done: break
            curr_observations = next_observations
        return total_reward


class Trainer:
    def __init__(self,
                 policy_net,
                 target_net,
                 batch_size,
                 optimizer,
                 sync_rate):
        self.policy_net = policy_net
        self.target_net = target_net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.sync_rate = sync_rate

    def run(self, replay_buffer):
        from torch.utils.data import DataLoader
        dataset = ReplayBufferDataset(replay_buffer, device=self.policy_net.device)
        xloader = DataLoader(dataset, self.batch_size, shuffle=True)
        avgloss = []
        for step, (s, a, r, sp, done) in enumerate(xloader):
            if step % self.sync_rate: self.update_target()
            self.optimizer.zero_grad()
            loss = self.dqn_loss(s, a, r, sp, done)
            loss.backward()
            self.optimizer.step()
            avgloss.append(loss.item())
        return np.mean(avgloss)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def dqn_loss(self, curr_s, curr_a, curr_r, next_s, curr_done, gamma=0.98):
        left_Q = self.policy_net(curr_s)[torch.arange(curr_s.shape[0]), curr_a]
        with torch.no_grad():
            qvalues = self.target_net(next_s).max(dim=1)[0].detach()
            qvalues[curr_done] = 0.
            right_Q = curr_r + gamma * qvalues
        return torch.nn.functional.mse_loss(left_Q, right_Q) 