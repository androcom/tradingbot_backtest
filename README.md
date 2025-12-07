# Hybrid AI Trading Bot (ML + RL)

이 프로젝트는 **지도 학습(XGBoost + LSTM)** 기반의 예측 모델과 **강화 학습(PPO)** 에이전트를 결합한 고도화된 하이브리드 트레이딩 봇입니다. 
기존의 규칙 기반(Rule-based) 로직을 넘어, 시장의 미세한 패턴을 학습한 ML 모델(Teacher)이 RL 에이전트(Student)에게 신호를 제공하고, RL 에이전트가 최종적인 매매 의사결정을 내리는 **Teacher-Student 아키텍처**를 채택하고 있습니다.

## 🚀 주요 특징 (Key Features)

1.  **하이브리드 AI 아키텍처 (Teacher-Student)**
    *   **Teacher (ML Strategy)**: XGBoost(트리 기반)와 LSTM(시계열 기반) 앙상블 모델이 시장의 방향성(Long/Short/Hold) 확률을 예측합니다.
    *   **Student (RL Strategy)**: PPO(Proximal Policy Optimization) 에이전트가 Teacher의 예측 값과 시장 데이터를 종합하여 최적의 행동(Action)을 학습합니다.

2.  **고도화된 강화 학습 (Reinforcement Learning)**
    *   **알고리즘**: Stable Baselines3의 PPO 사용.
    *   **환경 (Environment)**: `gymnasium` 기반의 커스텀 트레이딩 환경 (`CryptoTradingEnv`).
    *   **보상 (Reward)**: 자산 가치(Equity)의 변동률을 기반으로 수익 추구 및 리스크 관리 학습.

3.  **정교한 데이터 파이프라인**
    *   Binance OHLCV 데이터 수집 및 가공.
    *   기술적 지표 (RSI, ADX, BB, MACD 등) 및 파생 변수 (RVOL, 이격도, 로그 수익률) 생성.

4.  **현실적인 백테스팅 엔진**
    *   슬리피지(Slippage), 거래 수수료(Commission), 펀딩비(Funding Fee) 반영.
    *   레버리지 및 리스크 관리 로직 내장.

---

## 📂 시스템 구조 (System Structure)

```
📂 tradingbot_backtest
│
├── 📂 data/                # 수집된 데이터 및 전처리된 데이터
├── 📂 logs/                # 학습 로그 (TensorBoard) 및 백테스팅 결과
├── 📂 models/              # 학습된 모델 파일 (XGBoost, LSTM, PPO)
│
├── 📂 ml_strategy/         # [Teacher] 지도 학습 전략 모듈
│   ├── ai_models.py        # XGBoost + LSTM 모델 정의
│   ├── ga_optimizer.py     # 유전 알고리즘 (파라미터 최적화)
│   └── run_phase1_ml.py    # ML 모델 학습 및 테스트 스크립트
│
├── 📂 rl_strategy/         # [Student] 강화 학습 전략 모듈
│   ├── rl_env.py           # 강화 학습 환경 (Gymnasium)
│   └── train_rl.py         # PPO 에이전트 학습 스크립트
│
├── config.py               # 전체 설정 관리 (경로, 파라미터 등)
├── data_processor.py       # 데이터 수집 및 피처 엔지니어링
├── trading_engine.py       # 매매 실행 및 계좌 관리 엔진
└── requirements.txt        # 의존성 패키지 목록
```

---

## 🧠 상세 로직 (Detailed Logic)

### 1. ML Strategy (The Teacher)
*   **입력**: 기술적 지표, 캔들 패턴, 거래량 변화 등.
*   **모델**: 
    *   **XGBoost**: 정형 데이터 패턴 분류에 강점.
    *   **LSTM**: 시계열 데이터의 순차적 패턴 학습.
*   **출력**: 다음 구간의 포지션 확률 (Long / Short / Hold).
*   이 예측 확률은 RL 에이전트의 **Observation(관측 상태)** 중 하나로 제공되어, 에이전트가 더 나은 판단을 하도록 돕습니다.

### 2. RL Strategy (The Student)
*   **Observation Space**: 
    *   시장 데이터 (OHLCV, 지표).
    *   계좌 상태 (잔고, 보유 포지션, 미실현 손익).
    *   **Teacher's Signal (ML 예측 확률)**.
*   **Action Space (Discrete 4)**:
    *   `0`: Hold (관망)
    *   `1`: Open Long (롱 진입/추가)
    *   `2`: Open Short (숏 진입/추가)
    *   `3`: Close (포지션 청산)
*   **Reward Function**:
    *   포트폴리오 가치(Equity)의 상승률에 비례하여 보상.
    *   파산(Bankruptcy) 시 큰 페널티 부여.

---

## 🛠 설치 및 실행 (Installation & Usage)

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 수집 및 ML 모델 학습 (Phase 1)
먼저 지도 학습 모델을 학습시켜야 합니다.
```bash
python ml_strategy/run_phase1_ml.py
```

### 3. 강화 학습 에이전트 학습 (Phase 2)
ML 모델의 예측값을 바탕으로 PPO 에이전트를 학습시킵니다.
```bash
python rl_strategy/train_rl.py
```

### 4. 학습 모니터링
TensorBoard를 통해 학습 진행 상황(보상, 손실 등)을 실시간으로 확인할 수 있습니다.
```bash
tensorboard --logdir=logs/tensorboard
```

---

## ⚠️ 주의사항 (Disclaimer)
*   이 프로젝트는 학습 및 연구 목적으로 개발되었습니다.
*   실제 트레이딩에 사용할 경우 발생할 수 있는 금전적 손실에 대해 책임지지 않습니다.
*   과거의 데이터가 미래의 수익을 보장하지 않습니다.
