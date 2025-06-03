# Pong PPO 강화학습 프로젝트

이 프로젝트는 Atari Pong 게임을 PPO(Proximal Policy Optimization) 알고리즘을 사용하여 학습하는 강화학습 프로젝트입니다.

## 프로젝트 구조

```
.
├── configs/
│   └── ppo_config.yaml    # PPO 하이퍼파라미터 설정
├── src/
│   ├── algorithms/
│   │   └── ppo/          # PPO 알고리즘 구현
│   │       ├── agent.py  # PPO 에이전트
│   │       ├── model.py  # Actor-Critic 모델
│   │       └── train.py  # 학습 스크립트
│   └── evaluation/       # 모델 평가 코드
├── models/               # 학습된 모델 저장
└── requirements.txt      # 프로젝트 의존성
```

## 환경 설정

1. Python 3.8 환경 생성 및 활성화:
```bash
conda create -n atari python=3.8
conda activate atari
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

3. Atari ROM 설치:
```bash
AutoROM --accept-license
```

## 학습 실행

PPO 모델 학습을 시작하려면 다음 명령어를 실행하세요:

```bash
python -m src.algorithms.ppo.train
```

하이퍼파라미터 조정:
```bash
python -m src.algorithms.ppo.train --entropy_coef 0.1 --learning_rate 0.00005
```

이어서 학습:
```bash
python -m src.algorithms.ppo.train --model_path models/ppo/pong/best_model.pth
```

백그라운드에서 실행:
```bash
nohup python -m src.algorithms.ppo.train > training_ppo.log 2>&1 &
```

## Reward Shaping

현재 구현된 reward shaping 구조:

1. 거리 보상:
   - 공과 패들 사이의 거리에 따른 보상 (-0.05 * 거리)

2. 움직임 보상:
   - 패들이 공의 움직임 방향으로 움직일 때 (+0.3)
   - 패들이 공을 향해 움직일 때 (+0.1)

3. 위치 보상:
   - 공이 화면 중앙에 있을 때 (-0.005 * 중앙으로부터의 거리)
   - 공이 우리 쪽으로 올라올 때 (+0.2)

## 학습 결과

- 초기 성능: -155 ~ -182 점
- 중간 성능: 17점
- 최종 성능: 32점

## 하이퍼파라미터 설정

`configs/ppo_config.yaml` 파일에서 다음 하이퍼파라미터를 조정할 수 있습니다:

- `gamma`: 할인율 (기본값: 0.99)
- `gae_lambda`: GAE 람다 (기본값: 0.95)
- `clip_ratio`: PPO 클리핑 비율 (기본값: 0.2)
- `value_coef`: 가치 함수 손실 가중치 (기본값: 0.5)
- `entropy_coef`: 엔트로피 보너스 가중치 (기본값: 0.05)
- `learning_rate`: 학습률 (기본값: 0.0001)
- `batch_size`: 배치 크기 (기본값: 128)
- `ppo_epochs`: PPO 업데이트 에폭 수 (기본값: 10)

## 모델 평가

학습된 모델을 평가하려면 다음 명령어를 실행하세요:

```bash
python -m src.evaluation.evaluate
```

## 학습 모니터링

학습 중 다음 사항을 모니터링하세요:
1. 10 에피소드마다 평균 보상이 출력됩니다
2. 1000 에피소드마다 체크포인트가 저장됩니다
3. 최고 성능 모델은 자동으로 저장됩니다
4. 학습 곡선이 `models/ppo/pong/training_curve.png`에 저장됩니다

## 주의사항

- CUDA가 설치된 환경에서 실행하는 것을 권장합니다
- 학습에는 상당한 시간이 소요될 수 있습니다
- 충분한 GPU 메모리가 필요합니다 (최소 4GB 권장)
- 학습 로그는 `models/ppo/pong/logs/` 디렉토리에 저장됩니다