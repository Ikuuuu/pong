import os
import gym
import torch
import numpy as np
import yaml
from collections import deque
import matplotlib.pyplot as plt
from .agent import DQNAgent
import logging
from datetime import datetime

def setup_logging(model_dir):
    # 로그 디렉토리 생성
    log_dir = os.path.join(model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일명 생성 (날짜_시간 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

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

def train(env_name=None, episodes=None, model_path=None):
    # 설정 파일 로드
    logger = logging.getLogger(__name__)
    logger.info("Loading configuration...")
    config = load_config()
    logger.info("Configuration loaded successfully")
    
    # 설정값 적용
    env_name = env_name or config['env_name']
    episodes = episodes or config['episodes']
    
    # 환경 설정
    logger.info(f"Creating environment: {env_name}")
    env = gym.make(env_name, render_mode=None)
    logger.info("Environment created successfully")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 모델 저장 경로 설정
    model_dir = os.path.join("models", "dqn", "pong")
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    best_model_path = os.path.join(model_dir, "best_model.pth")
    
    logger.info("Creating model directories...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logger.info("Model directories created successfully")

    # 로깅 설정
    logger = setup_logging(model_dir)
    logger.info("="*50)
    logger.info("Training started")
    logger.info("="*50)
    logger.info(f"Using device: {device}")
    logger.info(f"Starting training with config: {config}")
    
    # 에이전트 설정
    logger.info("Initializing DQN Agent...")
    try:
        agent = DQNAgent(env.action_space.n, device, config)
        logger.info("DQN Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DQN Agent: {str(e)}")
        raise
    
    # 이전 모델이 있으면 로드
    if model_path and os.path.exists(model_path):
        logger.info(f"Attempting to load model from {model_path}")
        try:
            agent.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Current epsilon: {agent.epsilon:.3f}")
            logger.info(f"Total steps: {agent.total_steps}")
            logger.info(f"Replay buffer size: {len(agent.replay_buffer)}")
            
            # 이전 모델의 reward 정보를 로드
            checkpoint = torch.load(model_path)
            if 'best_reward' in checkpoint:
                best_reward = checkpoint['best_reward']
                logger.info(f"Loaded previous best reward: {best_reward}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    elif os.path.exists(best_model_path):
        logger.info(f"Attempting to load best model from {best_model_path}")
        try:
            agent.load(best_model_path)
            logger.info(f"Best model loaded successfully from {best_model_path}")
            logger.info(f"Current epsilon: {agent.epsilon:.3f}")
            logger.info(f"Total steps: {agent.total_steps}")
            logger.info(f"Replay buffer size: {len(agent.replay_buffer)}")
        except Exception as e:
            logger.error(f"Error loading best model: {str(e)}")
            raise
    else:
        logger.info("No existing model found. Starting fresh training.")

    # 학습 기록
    rewards = []
    best_reward = float('-inf')
    window_size = 100  # 평균을 계산할 에피소드 수
    
    # 이전 모델이 있으면 해당 모델의 reward를 best_reward로 설정
    if model_path and os.path.exists(model_path):
        try:
            # 이전 모델의 reward 정보를 로드
            checkpoint = torch.load(model_path)
            if 'best_reward' in checkpoint:
                best_reward = checkpoint['best_reward']
                logger.info(f"Loaded previous best reward: {best_reward}")
        except Exception as e:
            logger.warning(f"Could not load previous best reward: {str(e)}")
    
    logger.info(f"Starting training for {episodes} episodes")
    logger.info("="*50)

    for episode in range(episodes):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        state, stacked = stack_frames(None, obs, True)
        done = False
        total_reward = 0
        steps = 0

        logger.info(f"Starting Episode {episode}")

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

            # 500 스텝마다 진행상황 로깅
            # if steps % 500 == 0:
            #     logger.info(f"Episode {episode} - Steps: {steps}, Current Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

        rewards.append(total_reward)
        logger.info(f"Episode {episode} completed - Total Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.3f}")
        # 100 에피소드마다 로깅
        # if episode % 100 == 0:
        #     logger.info(f"Episode {episode} completed - Total Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.3f}")
        
        # 모델 저장
        if episode % config['save_interval'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode}.pth")
            agent.save(checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

        # 평균 reward 계산
        if len(rewards) >= window_size:
            avg_reward = sum(rewards[-window_size:]) / window_size
            # 최고 성능 모델 저장 (평균 reward가 이전보다 좋을 때만)
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(best_model_path, best_reward=best_reward)
                logger.info(f"New best model saved! Average Reward: {avg_reward:.2f} (over last {window_size} episodes)")

    # 최종 모델 저장
    agent.save(best_model_path, best_reward=best_reward)
    logger.info(f"Training completed. Best model saved to {best_model_path}")

    # 학습 곡선 그리기
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Dueling Double DQN on Pong")
    plt.savefig(os.path.join(model_dir, "training_curve.png"))
    logger.info("Training curve saved")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model checkpoint to continue training')
    args = parser.parse_args()
    train(model_path=args.model_path) 