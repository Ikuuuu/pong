# PPO + 병렬 환경 (ALE-Pong-v5)
# 구조: PPOAgent, 모델, 학습 루프, 벡터환경 생성 및 프레임 전처리 포함

import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from gym.vector import AsyncVectorEnv
from collections import deque

# 프레임 전처리 함수 (Atari용)
def preprocess_frame(obs):
    obs = obs[:, 35:195, :, :]     # crop
    obs = obs[:, ::2, ::2, :]      # downsample
    obs = obs[:, :, :, 0]          # R channel만
    obs = np.expand_dims(obs, 1)   # (B, 1, H, W)
    return obs.astype(np.float32) / 255.0

# 환경 생성기 (for 병렬)
def make_env():
    def thunk():
        env = gym.make("ALE/Pong-v5")
        return env
    return thunk

# 모델 정의
class ActorCritic(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, 512), nn.ReLU()
        )
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

# PPO 에이전트
class PPOAgent:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        self.device = device
        self.config = config

    def get_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            probs, values = self.model(obs)
        dist = Categorical(probs)
        actions = dist.sample()
        return actions.cpu().numpy(), dist.log_prob(actions).cpu().numpy(), values.cpu().numpy(), dist.entropy().cpu().numpy()

    def compute_gae(self, rewards, values, dones, next_values):
        adv = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.config['gamma'] * next_values[t] * mask - values[t]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * mask * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    def update(self, obs, actions, old_log_probs, returns, advantages):
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        for _ in range(self.config['ppo_epochs']):
            idxs = np.arange(len(obs))
            np.random.shuffle(idxs)
            for start in range(0, len(obs), self.config['batch_size']):
                end = start + self.config['batch_size']
                mb_idx = idxs[start:end]

                probs, values = self.model(obs[mb_idx])
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.config['clip_ratio'], 1.0 + self.config['clip_ratio']) * advantages[mb_idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), returns[mb_idx])

                loss = policy_loss + self.config['value_coef'] * value_loss - self.config['entropy_coef'] * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

# 학습 루프
def train():
    config = {
        'n_envs': 4,
        'rollout_len': 128,
        'lr': 2.5e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'ppo_epochs': 4,
        'batch_size': 256,
        'episodes': 1000,
        'save_path': './ppo_pong_model.pth'
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = AsyncVectorEnv([make_env() for _ in range(config['n_envs'])])
    obs, _ = envs.reset()
    obs = preprocess_frame(obs)
    stacked_frames = deque([obs] * 4, maxlen=4)
    state = np.concatenate(stacked_frames, axis=1)

    model = ActorCritic(input_channels=4, num_actions=envs.single_action_space.n)
    agent = PPOAgent(model, config, device)

    total_rewards = []

    for episode in range(config['episodes']):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        episode_reward = 0

        for _ in range(config['rollout_len']):
            actions, log_probs, values, _ = agent.get_action(state)
            next_obs, rewards, terms, truncs, _ = envs.step(actions)
            dones = np.logical_or(terms, truncs)

            next_obs = preprocess_frame(next_obs)
            stacked_frames.append(next_obs)
            next_state = np.concatenate(stacked_frames, axis=1)

            obs_buf.append(state)
            act_buf.append(actions)
            logp_buf.append(log_probs)
            rew_buf.append(rewards)
            val_buf.append(values.squeeze())
            done_buf.append(dones)

            episode_reward += np.mean(rewards)
            state = next_state

        total_rewards.append(episode_reward)

        with torch.no_grad():
            _, next_values = agent.model(torch.FloatTensor(state).to(device))
        next_values = next_values.cpu().numpy().squeeze()

        obs_buf = np.concatenate(obs_buf)
        act_buf = np.concatenate(act_buf)
        logp_buf = np.concatenate(logp_buf)
        rew_buf = np.array(rew_buf).transpose(1, 0).reshape(-1)
        val_buf = np.array(val_buf).transpose(1, 0).reshape(-1)
        done_buf = np.array(done_buf).transpose(1, 0).reshape(-1)
        next_values = np.repeat(next_values, config['rollout_len'])

        adv_buf, ret_buf = agent.compute_gae(rew_buf, val_buf, done_buf, next_values)
        agent.update(obs_buf, act_buf, logp_buf, ret_buf, adv_buf)

        print(f"Episode {episode}: mean reward {episode_reward:.2f}")

        if (episode + 1) % 100 == 0:
            agent.save(config['save_path'])
            print(f"Model saved to {config['save_path']}")

    # 학습 완료 후 리워드 그래프 저장
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Pong Training Reward')
    plt.grid(True)
    plt.savefig('ppo_pong_training_curve.png')
    print("Reward curve saved to ppo_pong_training_curve.png")

if __name__ == "__main__":
    train()
