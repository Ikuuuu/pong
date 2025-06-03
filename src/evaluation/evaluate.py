import os
import gym
import torch
import numpy as np
import cv2
from collections import deque
from src.algorithms.dqn.agent import DQNAgent
from src.algorithms.dqn.train import stack_frames
from src.algorithms.ppo.model import ActorCritic

def preprocess_observation(obs):
    # 학습 시와 동일한 전처리 적용
    obs = obs[35:195]  # 화면 크롭
    obs = obs[::2, ::2, 0]  # 다운샘플링
    obs[(obs == 144) | (obs == 109)] = 0  # 배경
    obs[obs != 0] = 1  # 나머지
    return obs.astype(np.float32)

class FrameStack:
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = []
        
    def add_frame(self, frame):
        if len(self.frames) >= self.num_frames:
            self.frames.pop(0)
        self.frames.append(frame)
        
    def get_stacked_frames(self):
        if len(self.frames) < self.num_frames:
            # 프레임이 부족한 경우 마지막 프레임으로 채움
            while len(self.frames) < self.num_frames:
                self.frames.append(self.frames[-1])
        return np.stack(self.frames, axis=0)

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

def evaluate_best_model(env_name='ALE/Pong-v5', model_path=None, render=True):
    if model_path is None:
        model_path = os.path.join("models", "ppo", "pong", "best_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 환경 설정
    env = gym.make(env_name, render_mode='human' if render else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 초기화 및 로드
    model = ActorCritic(input_channels=4, num_actions=env.action_space.n).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['actor_critic_state_dict'])
    model.eval()

    # 프레임 스택 초기화
    frame_stack = FrameStack(num_frames=4)

    # 평가 시작
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    done = False
    total_reward = 0
    steps = 0

    while not done:
        # 상태 전처리
        processed_obs = preprocess_observation(obs)
        frame_stack.add_frame(processed_obs)
        stacked_frames = frame_stack.get_stacked_frames()
        state = torch.FloatTensor(stacked_frames).unsqueeze(0).to(device)
        
        # 행동 선택
        with torch.no_grad():
            action_probs, state_value = model(state)
            print(f"Action probabilities: {action_probs[0].cpu().numpy()}")  # 행동 확률 출력
            print(f"State value: {state_value[0].item()}")  # 상태 가치 출력
            action = torch.argmax(action_probs).item()
            print(f"Selected action: {action}")  # 선택된 행동 출력
        
        # 환경과 상호작용
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = step_result

        obs = next_obs
        total_reward += reward
        steps += 1

    print(f"평가 완료:")
    print(f"총 보상: {total_reward}")
    print(f"총 스텝: {steps}")
    
    env.close()
    return total_reward, steps

if __name__ == "__main__":
    evaluate_best_model() 