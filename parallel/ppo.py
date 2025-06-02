import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import envpool

# 설정
save_dir = "./ppo_models"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "final_model.pth")
best_model_path = os.path.join(save_dir, "best_model.pth")

# 하이퍼파라미터
learning_rate = 2.5e-4
gamma = 0.99
gae_lambda = 0.95
clip_coef = 0.1
value_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5
total_timesteps = 10_000_000
num_envs = 1  # CPU에서 테스트 시 1로 줄이기
num_steps = 128
update_epochs = 4
num_minibatches = 1
batch_size = num_envs * num_steps
minibatch_size = batch_size // num_minibatches
save_interval = 1_000_000

# 환경 생성
envs = envpool.make_gym("Pong-v5", num_envs=num_envs, frame_skip=4, frame_stack=4)
obs_shape = envs.observation_space.shape
n_actions = envs.action_space.n

# 모델
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU()
        )
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = self.fc(x)
        return self.policy(x), self.value(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

# 초기화
obs, _ = envs.reset()
obs = torch.tensor(obs, dtype=torch.float32, device=device)
episode_rewards = []
ep_rew_buffer = deque(maxlen=100)
global_step = 0
best_avg_reward = -float('inf')

# 학습 루프
for update in range(total_timesteps // batch_size):
    log_probs, values, rewards, dones, actions = [], [], [], [], []
    next_obs = obs

    for step in range(num_steps):
        with torch.no_grad():
            logits, value = model(next_obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        obs_cpu, info = envs.step(action.cpu().numpy())
        rew = info["reward"]
        done = info["terminated"] | info["truncated"]
        next_obs = torch.tensor(obs_cpu, dtype=torch.float32, device=device)

        log_probs.append(log_prob)
        values.append(value.squeeze())
        rewards.append(torch.tensor(rew, dtype=torch.float32, device=device))
        dones.append(torch.tensor(done, dtype=torch.float32, device=device))
        actions.append(action)

        global_step += num_envs
        if "episode" in info:
            reward = info["episode"]["r"][0]
            ep_rew_buffer.append(reward)
            episode_rewards.append(reward)
            print(f"Step {global_step}: episode reward = {reward}")

    # GAE 계산
    with torch.no_grad():
        _, next_value = model(next_obs)
    advantages, returns = [], []
    gae = 0
    for t in reversed(range(num_steps)):
        delta = rewards[t] + gamma * next_value.squeeze() * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    # PPO 업데이트
    b_log_probs = torch.cat(log_probs)
    b_values = torch.cat(values)
    b_actions = torch.cat(actions)
    b_returns = torch.stack(returns).detach()
    b_advantages = torch.stack(advantages).detach()

    for epoch in range(update_epochs):
        idx = torch.randperm(batch_size)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_idx = idx[start:end]

            logits, new_values = model(obs[mb_idx])
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(b_actions[mb_idx])

            ratio = (new_log_probs - b_log_probs[mb_idx]).exp()
            surr1 = ratio * b_advantages[mb_idx]
            surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * b_advantages[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((b_returns[mb_idx] - new_values.squeeze()) ** 2).mean()
            entropy_loss = dist.entropy().mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    # 모델 저장
    if global_step % save_interval == 0:
        print(f"Saving model at step {global_step}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"ppo_{global_step}.pth"))

    # 베스트 모델 저장
    if len(ep_rew_buffer) == ep_rew_buffer.maxlen:
        avg_rew = np.mean(ep_rew_buffer)
        if avg_rew > best_avg_reward:
            print(f"New best avg reward: {avg_rew:.2f} → saving best model.")
            best_avg_reward = avg_rew
            torch.save(model.state_dict(), best_model_path)

# 학습 종료 후 모델 저장
torch.save(model.state_dict(), model_path)

# 에피소드별 리워드 시각화 저장
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Pong - Episode Total Rewards")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "episode_rewards.png"))
plt.close()
