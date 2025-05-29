import os
import gym
import torch
import numpy as np
from collections import deque
from src.algorithms.dqn.agent import DQNAgent
from src.algorithms.dqn.train import stack_frames

def evaluate(env_name='ALE/Pong-v5', model_path=None, render=True):
    if model_path is None:
        model_path = os.path.join("models", "dqn", "pong", "best_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    # 환경 설정
    env = gym.make(env_name, render_mode='human' if render else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env.action_space.n, device)
    
    # 모델 로드
    agent.load(model_path)
    agent.qnet.eval()
    agent.epsilon = 0.0  # 평가 시에는 탐험하지 않음

    # 평가 시작
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
        state = next_state
        total_reward += reward
        steps += 1

    print(f"Evaluation completed:")
    print(f"Total Reward: {total_reward}")
    print(f"Total Steps: {steps}")
    
    env.close()
    return total_reward, steps

if __name__ == "__main__":
    evaluate() 