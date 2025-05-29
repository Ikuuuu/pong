import os
import gym
import torch
import numpy as np
import yaml
from collections import deque
import matplotlib.pyplot as plt
from .agent import DQNAgent

def preprocess_frame(frame):
    frame = frame[35:195]
    frame = frame[::2, ::2, 0]
    frame[frame == 144] = 0
    frame[frame == 109] = 0
    frame[frame != 0] = 1
    return frame.astype(np.float32)

def stack_frames(stacked_frames, new_frame, is_new):
    processed = preprocess_frame(new_frame)
    if is_new:
        stacked_frames = deque([processed] * 4, maxlen=4)
    else:
        stacked_frames.append(processed)
    return np.stack(stacked_frames, axis=0), stacked_frames

def load_config(config_path='configs/dqn_config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train(env_name=None, episodes=None):
    # 설정 파일 로드
    config = load_config()
    
    # 설정값 적용
    env_name = env_name or config['env_name']
    episodes = episodes or config['episodes']
    
    # 환경 설정
    env = gym.make(env_name, render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 에이전트 설정
    agent = DQNAgent(env.action_space.n, device)
    agent.gamma = config['gamma']
    agent.lr = config['learning_rate']
    agent.epsilon_start = config['epsilon_start']
    agent.epsilon_end = config['epsilon_end']
    agent.epsilon_decay = config['epsilon_decay']
    agent.buffer_size = config['buffer_size']
    agent.batch_size = config['batch_size']
    agent.sync_interval = config['sync_interval']

    # 모델 저장 경로 설정
    model_dir = os.path.join("models", "dqn", "pong")
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    best_model_path = os.path.join(model_dir, "best_model.pth")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 학습 기록
    rewards = []
    best_reward = float('-inf')

    for episode in range(episodes):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        state, stacked = stack_frames(None, obs, True)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.get_action(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_result
            next_state, stacked = stack_frames(stacked, next_obs, False)

            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        
        # 학습 진행 상황 출력
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.3f}")

        # 모델 저장
        if episode % config['save_interval'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode}.pth")
            agent.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # 최고 성능 모델 저장
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(best_model_path)
            print(f"New best model saved! Reward: {best_reward}")

    # 최종 모델 저장
    agent.save(best_model_path)
    print(f"Training completed. Best model saved to {best_model_path}")

    # 학습 곡선 그리기
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Dueling Double DQN on Pong")
    plt.savefig(os.path.join(model_dir, "training_curve.png"))
    plt.show()

if __name__ == "__main__":
    train() 