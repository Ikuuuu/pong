"""
PPO (Proximal Policy Optimization) 학습 스크립트

이 파일은 PPO 알고리즘을 사용하여 Pong 게임을 학습하는 메인 스크립트입니다.
주요 기능:
1. 환경 설정 및 전처리
2. Reward shaping
3. 학습 루프
4. 모델 저장 및 로깅
"""

import os
import gym
import torch
import numpy as np
import yaml
from collections import deque
import matplotlib.pyplot as plt
from .agent import PPOAgent
import logging
from datetime import datetime

def setup_logging(model_dir):
    """
    로깅 설정
    
    Args:
        model_dir (str): 모델 저장 디렉토리
        
    Returns:
        logging.Logger: 설정된 로거
    """
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
    """
    게임 프레임 전처리
    
    Args:
        frame (np.ndarray): 원본 게임 프레임
        
    Returns:
        np.ndarray: 전처리된 프레임
    """
    frame = frame[35:195]  # 게임 영역만 추출
    frame = frame[::2, ::2, 0]  # 다운샘플링 및 그레이스케일
    frame[(frame == 144) | (frame == 109)] = 0  # 배경 제거
    frame[frame != 0] = 1  # 이진화
    return frame.astype(np.float32)

def stack_frames(stacked_frames, new_frame, is_new):
    """
    프레임 스택 생성
    
    Args:
        stacked_frames (deque): 이전 프레임 스택
        new_frame (np.ndarray): 새로운 프레임
        is_new (bool): 새로운 에피소드 여부
        
    Returns:
        tuple: (스택된 프레임, 프레임 스택)
    """
    processed = preprocess_frame(new_frame)
    if is_new:
        stacked_frames = deque([processed] * 4, maxlen=4)
    else:
        stacked_frames.append(processed)
    return np.stack(stacked_frames, axis=0), stacked_frames

def load_config(config_path='configs/ppo_config.yaml'):
    """
    설정 파일 로드
    
    Args:
        config_path (str): 설정 파일 경로
        
    Returns:
        dict: 설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_ball_position(frame):
    """
    공의 위치 추출
    
    Args:
        frame (np.ndarray): 게임 프레임
        
    Returns:
        tuple: (공의 y좌표, 공의 x좌표)
    """
    ball_pixels = np.where(frame == 1)
    if len(ball_pixels[0]) > 0:
        return np.mean(ball_pixels[0]), np.mean(ball_pixels[1])
    return None, None

def get_paddle_positions(frame):
    """
    패들의 위치 추출
    
    Args:
        frame (np.ndarray): 게임 프레임
        
    Returns:
        tuple: (상단 패들 y좌표, 하단 패들 y좌표)
    """
    paddle_pixels = np.where(frame == 1)
    if len(paddle_pixels[0]) > 0:
        sorted_y = np.sort(paddle_pixels[0])
        mid_point = len(sorted_y) // 2
        top_paddle = sorted_y[mid_point-1]
        bottom_paddle = sorted_y[mid_point]
        return top_paddle, bottom_paddle
    return None, None

def calculate_shaped_reward(frame, prev_frame, prev_ball_pos, prev_paddle_pos):
    """
    Reward shaping 계산
    
    Args:
        frame (np.ndarray): 현재 프레임
        prev_frame (np.ndarray): 이전 프레임
        prev_ball_pos (tuple): 이전 공 위치
        prev_paddle_pos (float): 이전 패들 위치
        
    Returns:
        tuple: (shaped reward, (공 위치), 패들 위치)
    """
    reward = 0
    
    # 현재 프레임에서 공과 패들 위치 찾기
    ball_y, ball_x = get_ball_position(frame)
    top_paddle, bottom_paddle = get_paddle_positions(frame)
    
    if ball_y is not None and top_paddle is not None:
        # 1. 공과 패들 사이의 거리에 따른 보상
        paddle_to_ball_dist = abs(bottom_paddle - ball_y)
        reward -= paddle_to_ball_dist * 0.05
        
        # 2. 패들이 공을 향해 움직일 때 보상
        ball_movement = 0
        if prev_paddle_pos is not None and prev_ball_pos is not None:
            paddle_movement = bottom_paddle - prev_paddle_pos
            ball_movement = ball_y - prev_ball_pos
            
            # 패들이 공의 움직임 방향으로 움직일 때 보상
            if (paddle_movement > 0 and ball_movement > 0) or \
               (paddle_movement < 0 and ball_movement < 0):
                reward += 0.3
            
            # 패들이 공을 향해 움직이기만 해도 작은 보상
            if (paddle_movement > 0 and ball_y > bottom_paddle) or \
               (paddle_movement < 0 and ball_y < bottom_paddle):
                reward += 0.1
        
        # 3. 공이 화면 중앙에 있을 때 보상
        center_y = frame.shape[0] / 2
        reward -= abs(ball_y - center_y) * 0.005
        
        # 4. 공이 우리 쪽으로 올라올 때 보상
        if ball_movement < 0:
            reward += 0.2
    
    return reward, (ball_y, ball_x), bottom_paddle

def train(env_name=None, episodes=None, model_path=None, config_updates=None):
    """
    PPO 학습 메인 함수
    
    Args:
        env_name (str): 환경 이름
        episodes (int): 학습할 에피소드 수
        model_path (str): 이어서 학습할 모델 경로
        config_updates (dict): 하이퍼파라미터 업데이트
    """
    # 설정 파일 로드
    logger = logging.getLogger(__name__)
    logger.info("Loading configuration...")
    config = load_config()
    
    # 하이퍼파라미터 업데이트
    if config_updates:
        config.update(config_updates)
        logger.info(f"Updated hyperparameters: {config_updates}")
    
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
    model_dir = os.path.join("models", "ppo", "pong")
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
    logger.info("Initializing PPO Agent...")
    try:
        agent = PPOAgent(input_channels=4, num_actions=env.action_space.n, device=device, config=config)
        logger.info("PPO Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PPO Agent: {str(e)}")
        raise
    
    # 이전 모델이 있으면 로드
    if model_path and os.path.exists(model_path):
        logger.info(f"Attempting to load model from {model_path}")
        try:
            agent.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    # 학습 기록
    episode_rewards = []
    best_reward = float('-inf')
    window_size = 100  # 평균을 계산할 에피소드 수
    
    logger.info(f"Starting training for {episodes} episodes")
    logger.info("="*50)

    for episode in range(episodes):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        state, stacked = stack_frames(None, obs, True)
        done = False
        total_reward = 0
        steps = 0
        
        # 에피소드 데이터 수집
        states = []
        actions = []
        step_rewards = []
        values = []
        log_probs = []
        dones = []
        
        # 이전 상태 저장
        prev_ball_pos = None
        prev_paddle_pos = None
        
        while not done:
            # 액션 선택
            action, log_prob, value = agent.get_action(state)
            
            # 환경과 상호작용
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked = stack_frames(stacked, next_obs, False)
            
            # Reward shaping 적용
            shaped_reward, ball_pos, paddle_pos = calculate_shaped_reward(
                next_state[0],  # 현재 프레임
                state[0],       # 이전 프레임
                prev_ball_pos,
                prev_paddle_pos
            )
            
            # 원래 보상과 shaped reward 결합
            combined_reward = reward + shaped_reward
            
            # 상태 업데이트
            prev_ball_pos = ball_pos[0]
            prev_paddle_pos = paddle_pos
            
            # 데이터 저장
            states.append(state)
            actions.append(action)
            step_rewards.append(combined_reward)  # shaped reward 사용
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            state = next_state
            total_reward += combined_reward  # shaped reward 사용
            steps += 1
        
        # 에피소드 총 보상 저장
        episode_rewards.append(total_reward)
        
        # 마지막 상태의 가치 계산
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, next_value = agent.actor_critic(state_tensor)
            next_value = next_value.item()
        
        # GAE 계산
        advantages, returns = agent.compute_gae(step_rewards, values, next_value, dones)
        
        # PPO 업데이트
        agent.update(states, actions, log_probs, returns, advantages)
        
        # 학습 진행 상황 로깅
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-window_size:])
            logger.info(f"Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}")
            
            # 최고 성능 모델 저장
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(best_model_path)
                logger.info(f"New best model saved! Average Reward: {best_reward:.2f}")
        
        # 주기적으로 체크포인트 저장
        if (episode + 1) % 1000 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode + 1}.pth")
            agent.save(checkpoint_path)
            logger.info(f"Checkpoint saved at episode {episode + 1}")
    
    # 학습 종료
    logger.info("Training completed!")
    logger.info(f"Best average reward: {best_reward:.2f}")
    
    # 최종 모델 저장
    final_model_path = os.path.join(model_dir, "final_model.pth")
    agent.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # 학습 곡선 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(model_dir, 'training_curve.png'))
    plt.close()
    
    return best_reward

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model checkpoint to continue training')
    parser.add_argument('--entropy_coef', type=float, help='Entropy coefficient for exploration')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    args = parser.parse_args()
    
    # 하이퍼파라미터 업데이트
    config_updates = {}
    if args.entropy_coef is not None:
        config_updates['entropy_coef'] = args.entropy_coef
    if args.learning_rate is not None:
        config_updates['learning_rate'] = args.learning_rate
    
    train(model_path=args.model_path, config_updates=config_updates) 