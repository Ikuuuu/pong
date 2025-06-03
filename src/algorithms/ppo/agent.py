"""
PPO (Proximal Policy Optimization) 에이전트 구현

이 파일은 PPO 알고리즘의 핵심 에이전트 클래스를 구현합니다.
PPO는 정책 기반 강화학습 알고리즘으로, 안정적인 학습을 위해 정책 업데이트를 제한합니다.

주요 기능:
1. 액션 선택 (get_action)
2. GAE(Generalized Advantage Estimation) 계산 (compute_gae)
3. PPO 업데이트 (update)
4. 모델 저장/로드 (save/load)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .model import ActorCritic

class PPOAgent:
    def __init__(self, input_channels, num_actions, device, config):
        """
        PPO 에이전트 초기화
        
        Args:
            input_channels (int): 입력 이미지의 채널 수 (프레임 스택 수)
            num_actions (int): 가능한 액션의 수
            device (torch.device): 학습에 사용할 디바이스 (CPU/GPU)
            config (dict): 하이퍼파라미터 설정
        """
        self.device = device
        self.num_actions = num_actions
        
        # PPO 하이퍼파라미터
        self.gamma = config['gamma']          # 할인율
        self.gae_lambda = config['gae_lambda']  # GAE 람다
        self.clip_ratio = config['clip_ratio']  # PPO 클리핑 비율
        self.value_coef = config['value_coef']  # 가치 함수 손실 가중치
        self.entropy_coef = config['entropy_coef']  # 엔트로피 보너스 가중치
        self.max_grad_norm = config['max_grad_norm']  # 그래디언트 클리핑
        self.ppo_epochs = config['ppo_epochs']  # PPO 업데이트 에폭 수
        self.batch_size = config['batch_size']  # 배치 크기
        
        # Actor-Critic 네트워크 및 옵티마이저 초기화
        self.actor_critic = ActorCritic(input_channels, num_actions).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config['learning_rate'])
        
    def get_action(self, state):
        """
        현재 상태에서 액션 선택
        
        Args:
            state (np.ndarray): 현재 상태 (프레임 스택)
            
        Returns:
            tuple: (선택된 액션, 로그 확률, 상태 가치)
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, state_value = self.actor_critic(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob, state_value.item()
    
    def compute_gae(self, rewards, values, next_value, dones):
        """
        GAE(Generalized Advantage Estimation) 계산
        
        Args:
            rewards (list): 보상 시퀀스
            values (list): 상태 가치 시퀀스
            next_value (float): 마지막 상태의 가치
            dones (list): 종료 상태 시퀀스
            
        Returns:
            tuple: (정규화된 어드밴티지, 반환값)
        """
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32).to(self.device)
        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return norm_advantages, returns
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        PPO 업데이트 수행
        
        Args:
            states (list): 상태 시퀀스
            actions (list): 액션 시퀀스
            old_log_probs (list): 이전 정책의 로그 확률
            returns (list): 반환값 시퀀스
            advantages (list): 어드밴티지 시퀀스
        """
        # 데이터를 numpy 배열로 변환
        states = np.array(states)
        actions = np.array(actions)
        old_log_probs = np.array([lp.cpu().numpy() for lp in old_log_probs])
        returns = np.array([r.cpu().numpy() for r in returns])
        advantages = np.array([a.cpu().numpy() for a in advantages])
        
        # 미니배치로 데이터 분할
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        
        for start_idx in range(0, len(states), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(states))
            minibatch_indices = indices[start_idx:end_idx]
            
            # 배치 데이터 준비
            batch_states = torch.FloatTensor(states[minibatch_indices]).to(self.device)
            batch_actions = torch.LongTensor(actions[minibatch_indices]).to(self.device)
            batch_old_log_probs = torch.FloatTensor(old_log_probs[minibatch_indices]).to(self.device)
            batch_returns = torch.FloatTensor(returns[minibatch_indices]).to(self.device)
            batch_advantages = torch.FloatTensor(advantages[minibatch_indices]).to(self.device)
            
            # PPO 업데이트
            for _ in range(self.ppo_epochs):
                # 현재 정책으로 액션 평가
                new_log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions)
                
                # 정책 비율 계산
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO 클리핑
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 가치 함수 손실
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 엔트로피 보너스
                entropy_loss = -entropy.mean()
                
                # 전체 손실
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 최적화
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def save(self, path):
        """
        모델 저장
        
        Args:
            path (str): 저장할 경로
        """
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """
        모델 로드
        
        Args:
            path (str): 로드할 모델 경로
        """
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 