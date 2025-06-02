import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from .model import DuelingQNet

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, action_size, device, config):
        # config에서 하이퍼파라미터 로드
        self.gamma = config['gamma']
        self.lr = config['learning_rate']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.sync_interval = config['sync_interval']
        
        self.epsilon = self.epsilon_start
        self.total_steps = 0
        self.action_size = action_size
        self.device = device

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = DuelingQNet(self.action_size).to(self.device)
        self.qnet_target = DuelingQNet(self.action_size).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.update_count = 0

    def get_action(self, state):
        self.total_steps += 1
        self.update_epsilon()  # 스텝 단위로 업데이트

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                qs = self.qnet(state)
            return qs.argmax(dim=1).item()

    def update_epsilon(self):
        decay_steps = self.epsilon_decay
        self.epsilon = max(self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.total_steps / decay_steps)
        )

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state, action, reward, next_state, done = state.to(self.device), action.to(self.device), reward.to(self.device), next_state.to(self.device), done.to(self.device)

        qs = self.qnet(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.qnet(next_state).argmax(1)
            next_qs = self.qnet_target(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = reward + (1 - done) * self.gamma * next_qs

        loss = F.mse_loss(qs, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.sync_interval == 0:
            self.sync_qnet()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def save(self, path, best_reward=None):
        save_dict = {
            'qnet_state_dict': self.qnet.state_dict(),
            'qnet_target_state_dict': self.qnet_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'replay_buffer': self.replay_buffer.buffer
        }
        if best_reward is not None:
            save_dict['best_reward'] = best_reward
        torch.save(save_dict, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.qnet.load_state_dict(checkpoint['qnet_state_dict'])
        self.qnet_target.load_state_dict(checkpoint['qnet_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps'] 