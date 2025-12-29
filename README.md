# Hierarchical AI Trading System (Model Factory & Strategy Modularization)

이 프로젝트는 **계층적 강화학습(Hierarchical RL)**을 지향하는 차세대 알고리즘 트레이딩 시스템입니다.
하위 레벨에서는 **지도 학습(XGBoost + LSTM)**과 **규칙 기반(Rule-based)** 전략들이 개별적인 매매를 수행하고, 상위 레벨에서는 **강화 학습(RL Manager)**이 시장 국면(Trend/Range/Panic)을 판단하여 최적의 전략을 선택합니다.

또한, **Model Factory**를 통해 모델 학습, 파라미터 튜닝, 성능 검증, 챔피언 승격까지의 전 과정이 **완전 자동화(Full-Automation)**되어 있습니다.

---

## 🚀 핵심 경쟁력 (Key Features)

### 1. 전략 모듈화 (Modular Strategy Architecture)
단일 모델에 의존하지 않고, 시장 상황에 맞는 특화된 전략을 부품처럼 교체할 수 있습니다.
*   **Trend Strategy**: `Hybrid Learner (XGB+LSTM)`로 추세를 예측하고 `Trailing Stop`으로 수익을 극대화합니다.
*   **Range Strategy**: RSI 및 볼린저 밴드를 활용하여 횡보장에서 역추세 매매를 수행합니다.
*   **Defense Strategy**: 하락장 및 급변동 시 현금 비중을 늘리거나 숏 포지션으로 방어합니다.

### 2. 완전 자동화된 모델 공장 (The Model Factory)
버튼 하나로 다음의 과정이 자동으로 수행됩니다.
1.  **Teacher Tuning**: AI 모델(XGB/LSTM)의 구조(Layer, Node 등)를 유전 알고리즘(TPE)으로 최적화.
2.  **Training**: 최적화된 구조로 대용량 데이터를 학습하여 추세 예측 모델 생성.
3.  **Strategy Tuning**: 학습된 모델을 바탕으로 실전 매매 규칙(TP/SL, 진입 임계값)을 시뮬레이션하여 최적화.
4.  **Championship**: 기존 챔피언 모델과 새로운 도전자 모델을 경쟁(Battle)시켜, 승리한 모델을 실전 봇에 자동 배포.

### 3. 데이터 무결성 및 고성능 (Integrity & Performance)
*   **Strict Walk-Forward**: 미래 참조 편향(Look-ahead Bias)을 원천 차단하는 검증 방식을 사용합니다.
*   **Mixed Precision & Batch Optimization**: 대용량 데이터 학습 시 GPU 메모리 효율과 속도를 극대화했습니다.
*   **Robust Logging**: 모든 프로세스는 정규화된 로그와 CSV 리포트를 남겨 투명하게 관리됩니다.

---

## 📂 시스템 구조 (Directory Structure)

```text
tradingbot/
├── 📁 configs/                 # [설정] 전략별 파라미터(JSON) 자동 관리
├── 📁 core/                    # [핵심] 공통 모듈 (Data, Config, Constants)
├── 📁 execution/               # [실행] 전략을 선택하고 매매를 수행하는 엔진
├── 📁 learning/                # [두뇌] AI 모델 구조, 학습 로직, RL 환경
├── 📁 operations/              # [운영] 공장(Factory), 생애주기(Lifecycle), 스케줄러
├── 📁 strategies/              # [전략] Trend, Range, Defense 등 개별 전략 구현체
├── 📁 tuning/                  # [최적화] Optuna 기반의 모델 및 전략 튜너
├── 📁 data/                    # [데이터] OHLCV 및 가공 데이터 (Parquet)
├── 📁 logs/                    # [기록] 실행 로그 및 튜닝 결과(CSV)
└── 📁 models_saved/            # [저장소] 챔피언 모델 및 아카이브
```

---

## 🗺️ 프로젝트 로드맵 (Roadmap)

### ✅ Phase 1: 기반 구축 및 안정화 (완료)
*   데이터 파이프라인(Binance API, Feature Engineering) 구축.
*   GPU 가속(Mixed Precision) 및 로깅 시스템 표준화.
*   기본 AI 모델(XGBoost + LSTM) 설계 및 학습 파이프라인 완성.

### ✅ Phase 2: 모듈화 및 자동화 (완료)
*   **Strategy Pattern 도입**: `TradingCore`를 분해하여 확장 가능한 전략 구조 수립.
*   **Model Factory 구축**: 튜닝 → 학습 → 검증 → 배포의 완전 자동화 구현.
*   **Lifecycle Manager**: 모델 간 경쟁 시스템 및 챔피언 자동 승격 로직 구현.
*   **Config Manager**: JSON 기반의 동적 파라미터 관리 시스템 구축.

### 🔜 Phase 3: 지능형 관리자 (RL Manager) (진행 예정)
*   **Manager Environment**: 하위 전략을 Action으로 선택하는 강화학습 환경 구축.
*   **Meta-Learning**: 시장 국면(변동성, 추세 강도 등)에 따라 전략을 스위칭하는 상위 에이전트 학습.

### 🔜 Phase 4: 실전 투입 (Deployment)
*   **Live Bot Implementation**: 바이낸스 API 연동 및 실시간 구동 스크립트 작성.
*   **Paper Trading**: Testnet 환경에서의 모의 투자 및 슬리피지/수수료 검증.

---

## ⚠️ Disclaimer
본 프로젝트는 알고리즘 트레이딩 연구 및 학습 목적으로 개발되었습니다. 실제 투자는 본인의 책임 하에 신중하게 진행해야 하며, 개발자는 이 프로그램 사용으로 인한 금전적 손실에 대해 책임을 지지 않습니다.