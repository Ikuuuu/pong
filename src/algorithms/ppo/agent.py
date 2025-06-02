import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .model import ActorCritic

class PPOAgent:
    def __init__(self, input_channels, num_actions, device, config):
        self.device = device
        self.num_actions = num_actions
        
        # 하이퍼파라미터
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.clip_ratio = config['clip_ratio']
        self.value_coef = config['value_coef']
        self.entropy_coef = config['entropy_coef']
        self.max_grad_norm = config['max_grad_norm']
        self.ppo_epochs = config['ppo_epochs']
        self.batch_size = config['batch_size']
        
        # 네트워크
        self.actor_critic = ActorCritic(input_channels, num_actions).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config['learning_rate'])
        
    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, state_value = self.actor_critic(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob, state_value.item()
    
    def compute_gae(self, rewards, values, next_value, dones):
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
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        # 데이터를 numpy 배열로 변환
        states = np.array(states)
        actions = np.array(actions)
        old_log_probs = np.array([lp.cpu().numpy() for lp in old_log_probs])
        returns = np.array([r.cpu().numpy() for r in returns])
        advantages = np.array([a.cpu().numpy() for a in advantages])
        
        # 데이터를 배치로 나누기
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
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 