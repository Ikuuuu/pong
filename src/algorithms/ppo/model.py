import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(ActorCritic, self).__init__()
        
        # 공통 특징 추출기
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # CNN 출력 크기 계산
        # 입력: (4, 80, 80) -> (32, 19, 19) -> (64, 8, 8) -> (64, 6, 6)
        # 64 * 6 * 6 = 2304
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU()
        )
        
        # Actor (정책) 네트워크
        self.actor = nn.Linear(512, num_actions)
        
        # Critic (가치) 네트워크
        self.critic = nn.Linear(512, 1)
        
    def forward(self, x):
        # 입력이 (batch_size, channels, height, width) 형태라고 가정
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        logits = self.actor(x)
        action_probs = F.softmax(logits, dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value
    
    def evaluate_actions(self, x, action):
        action_probs, state_value = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_log_probs, state_value, dist_entropy 