# Pong DQN 강화학습 프로젝트

이 프로젝트는 Atari Pong 게임을 Dueling Double DQN 알고리즘을 사용하여 학습하는 강화학습 프로젝트입니다.

## 프로젝트 구조

```
.
├── configs/
│   └── dqn_config.yaml    # DQN 하이퍼파라미터 설정
├── src/
│   ├── algorithms/
│   │   └── dqn/          # DQN 알고리즘 구현
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

DQN 모델 학습을 시작하려면 다음 명령어를 실행하세요:

```bash
python -m src.algorithms.dqn.train
```

학습 중 주요 정보:
- 10 에피소드마다 현재 보상과 스텝 수가 출력됩니다
- 100 에피소드마다 체크포인트가 저장됩니다
- 최고 성능 모델은 자동으로 저장됩니다
- 학습 곡선이 `models/dqn/pong/training_curve.png`에 저장됩니다

## 모델 평가

학습된 모델을 평가하려면 다음 명령어를 실행하세요:

```bash
python -m src.evaluation.evaluate
```

기본적으로 `models/dqn/pong/best_model.pth`의 모델을 사용합니다.
다른 모델을 사용하려면 경로를 지정할 수 있습니다:

```bash
python -m src.evaluation.evaluate --model_path "models/dqn/pong/checkpoints/model_episode_1000.pth"
```

## 하이퍼파라미터 설정

`configs/dqn_config.yaml` 파일에서 다음 하이퍼파라미터를 조정할 수 있습니다:

- `gamma`: 할인율 (기본값: 0.99)
- `learning_rate`: 학습률 (기본값: 0.0001)
- `epsilon_start`: 초기 탐험률 (기본값: 1.0)
- `epsilon_end`: 최종 탐험률 (기본값: 0.1)
- `epsilon_decay`: 탐험률 감소 속도 (기본값: 20000)
- `buffer_size`: 리플레이 버퍼 크기 (기본값: 50000)
- `batch_size`: 배치 크기 (기본값: 128)
- `sync_interval`: 타겟 네트워크 동기화 주기 (기본값: 1000)
- `episodes`: 총 학습 에피소드 수 (기본값: 2000)

## 학습 모니터링

학습 중 다음 사항을 모니터링하세요:
1. 에피소드별 보상이 점진적으로 증가하는지
2. epsilon 값이 적절히 감소하는지
3. 학습 곡선이 안정적으로 상승하는지

## 주의사항

- CUDA가 설치된 환경에서 실행하는 것을 권장합니다
- 학습에는 상당한 시간이 소요될 수 있습니다
- 충분한 GPU 메모리가 필요합니다 (최소 4GB 권장)


nohup python -m src.algorithms.dqn.train --model_path "models/dqn/pong/best_model.pth" > training.log 2>&1 &

python -m src.evaluation.evaluate --model_path ""models/dqn/pong/best_model.pth"

nohup python -m src.algorithms.ppo.train > training_ppo.log 2>&1 &