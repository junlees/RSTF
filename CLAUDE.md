# CLAUDE.md

This file provides context and conventions for Claude Code when working in this repository.

---

## Project Overview

CNC 밀링 머신의 진동 데이터에서 **이상 탐지(Anomaly Detection)**를 수행하는 PyTorch 프로젝트.

- **참조 논문**: Çekik & Turan (2025) — *RoughLSTM-Based Anomaly Detection in CNC Vibration Data*, Appl. Sci.
- **변경 사항**: 논문의 LSTM 구조를 **Transformer Autoencoder**로 교체
- **데이터셋**: [Bosch CNC Machining Dataset](https://github.com/boschresearch/CNC_Machining) (M01/M02/M03, 15개 공정 작업)
- **베이스 템플릿**: [victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)

---

## Repository Structure

```
├── base/                        # 템플릿 추상 기반 클래스 (수정 금지)
│   ├── base_data_loader.py      #   BaseDataLoader
│   ├── base_model.py            #   BaseModel (nn.Module 상속)
│   └── base_trainer.py          #   BaseTrainer (체크포인트/로깅 담당)
│
├── model/
│   ├── transformer_model.py     # ★ TransformerAnomalyDetector (핵심 모델)
│   ├── loss.py                  # ★ CombinedAnomalyLoss + nll_loss
│   ├── metric.py                # ★ accuracy / f1_score / FPR / FNR 등
│   ├── model.py                 #   원본 MnistModel (유지)
│   └── __init__.py
│
├── data_loader/
│   ├── cnc_data_loaders.py      # ★ CNCVibrationDataset + CNCDataLoader
│   └── data_loaders.py          #   원본 MnistDataLoader (유지)
│
├── trainer/
│   ├── anomaly_trainer.py       # ★ AnomalyTrainer (combined loss 처리)
│   ├── trainer.py               #   원본 Trainer (유지)
│   └── __init__.py
│
├── logger/                      # TensorboardWriter (수정 금지)
├── utils/                       # MetricTracker, prepare_device 등 (수정 금지)
│
├── config_cnc_transformer.json  # ★ CNC 실험용 설정 파일
├── train_cnc.py                 # ★ CNC 학습 진입점
├── test_cnc.py                  # ★ CNC 평가 + Rough Set 후처리
├── train.py                     #   원본 MNIST 학습 스크립트 (유지)
└── parse_config.py              # ConfigParser (수정 금지)
```

`★` 표시 파일이 이번 프로젝트에서 추가/수정된 파일입니다.

---

## Core Architecture

### TransformerAnomalyDetector (`model/transformer_model.py`)

```
Input (B, W=50, F=3)
  → Linear(3 → 64)              # Input Projection
  → PositionalEncoding
  → TransformerEncoder × 3      # Pre-LayerNorm, nhead=4, FFN dim=256
  → Linear(64×50 → 32)          # Bottleneck (encoder_to_latent)
  → Linear(32 → 64×50)          # Latent Expansion (latent_to_decoder)
  → TransformerDecoder × 3      # Cross-attention with encoder memory
  → Linear(64 → 3)              # Reconstruction head  → (B, W, F)
  → Linear(32 → 2)              # Classifier head      → (B, 2) logits
```

**Forward 반환값은 항상 tuple**: `(logits, reconstruction)`
- `logits`       : `(B, num_classes)`
- `reconstruction`: `(B, window_size, n_features)`

모든 loss 함수와 metric 함수는 이 tuple을 처리할 수 있어야 합니다.

### CombinedAnomalyLoss (`model/loss.py`)

```
L = alpha * CrossEntropy(logits, labels)
  + (1 - alpha) * MSE(reconstruction, input)

기본값: alpha = 0.6
설정:   config["trainer"]["loss_alpha"]
```

### Rough Set 후처리 (`test_cnc.py`)

```python
# 불확실 구간: [0.5 - ε, 0.5 + ε]  (기본 ε = 0.1)
upper_approach : p > 0.5 + ε  → 확실한 이상 (anomaly)
lower_approach : p < 0.5 - ε  → 확실한 정상 (normal)
boundary_region: 그 사이       → 최근접 클래스로 분류
```

### Temporal Context Correction (`apply_context_correction`)

boundary 샘플을 시간적 이웃(upper/lower 확정 판정)의 **다수결**로 재판정.

```python
# 각 boundary 인덱스 i 에 대해
neighborhood = regions[i-k : i] + regions[i+1 : i+k+1]  # i 자신 제외
n_upper = sum(neighborhood == "upper")
n_lower = sum(neighborhood == "lower")

n_upper > n_lower           → 이상 (1)
n_lower >= n_upper (동점)   → 정상 (0)  # 보수적
둘 다 0 (전부 boundary)     → fallback: P >= 0.5
```

`require_both_sides=True` 옵션: 왼쪽·오른쪽 **양쪽**에 upper가 있어야 이상 → 더 엄격.

- **권장 설정**: `--context-k 5` (FPR 8.67%↓, FNR 0.43%↓, Accuracy 99.01%↑)
- `--require-both-sides`는 FPR 개선 효과 감소로 비권장

---

## Data

### 입력 형식

| 항목 | 값 |
|---|---|
| 센서 | Bosch CISS 3축 가속도계 (x, y, z) |
| 샘플링 레이트 | 2 kHz |
| 윈도우 크기 | 50 샘플 |
| 오버랩 | 25 샘플 (stride = 25) |
| 정규화 | window 단위 z-score |
| 레이블 | 0 = NOK (이상), 1 = OK (정상) |

### 데이터셋 설치

```bash
git clone https://github.com/boschresearch/CNC_Machining data/
```

### CSV 파일 명명 규칙

- `*_OK.csv`  → label = 1 (정상)
- `*_NOK.csv` → label = 0 (이상)

---

## Commands

### 학습

```bash
# 기본 실행 (M01, OP02+OP05+OP08+OP11+OP14)
python train_cnc.py -c config_cnc_transformer.json

# 하이퍼파라미터 오버라이드
python train_cnc.py -c config_cnc_transformer.json --lr 0.0001 --bs 64 --alpha 0.7

# 체크포인트에서 재개
python train_cnc.py -r saved/models/CNC_TransformerAutoencoder/<run_id>/checkpoint-epoch10.pth
```

### 평가

```bash
# 기본 (RST only)
python test_cnc.py -r saved/models/CNC_TransformerAutoencoder/<run_id>/model_best.pth

# ε 조정
python test_cnc.py -r <checkpoint> --epsilon 0.05

# Temporal context correction (권장: k=5)
python test_cnc.py -r <checkpoint> --context-k 5

# 엄격 조건 (양쪽 upper 이웃 필요)
python test_cnc.py -r <checkpoint> --context-k 5 --require-both-sides

# 시각화 포함
python test_cnc.py -r <checkpoint> --context-k 5 --plot --boundary-dist
```

### TensorBoard

```bash
tensorboard --logdir saved/log/
```

### 원본 MNIST 예제 (템플릿 검증용)

```bash
python train.py -c config.json
```

---

## Configuration

설정은 모두 JSON으로 관리됩니다 (`config_cnc_transformer.json`).
CLI 오버라이드 키는 `;`으로 경로를 구분합니다.

```json
{
  "arch.args.d_model"          : 64,
  "arch.args.latent_dim"       : 32,
  "arch.args.num_encoder_layers": 3,
  "trainer.loss_alpha"         : 0.6,
  "trainer.early_stop"         : 10,
  "trainer.monitor"            : "min val_loss"
}
```

CLI 예시:
```bash
--lr          → optimizer;args;lr
--bs          → data_loader;args;batch_size
--alpha       → trainer;loss_alpha
--machines    → data_loader;args;machines
--ops         → data_loader;args;operations
```

---

## Conventions

### 모델 출력 처리

모델 출력이 `(logits, reconstruction)` tuple이므로, metric과 loss 함수 작성 시 반드시 처리:

```python
# metric 함수 패턴
def my_metric(output, target):
    if isinstance(output, (tuple, list)):
        logits = output[0]        # classification
        recon  = output[1]        # reconstruction (필요 시)
    else:
        logits = output
    ...
```

### 새 모델 추가

1. `model/` 에 파일 생성, `BaseModel` 상속
2. `model/__init__.py` 에 import 추가
3. `config.json` 의 `arch.type` 에 클래스명 등록

### 새 Trainer 추가

1. `trainer/` 에 파일 생성, `BaseTrainer` 상속, `_train_epoch()` 구현
2. `trainer/__init__.py` 에 import 추가

### 새 DataLoader 추가

1. `data_loader/` 에 파일 생성, `BaseDataLoader` 상속
2. `config.json` 의 `data_loader.type` 에 클래스명 등록

### 코드 스타일

- Python 3.10+, type hint 사용
- 텐서 shape은 주석으로 명시: `# (B, W, d_model)`
- flake8 준수 (`.flake8` 설정: max-line-length=120)

---

## Key Design Decisions

| 결정 | 이유 |
|---|---|
| Pre-LayerNorm (`norm_first=True`) | Post-LN보다 학습 안정성 우수 |
| Xavier 초기화 | Transformer 수렴 속도 개선 |
| Gradient clipping (max_norm=1.0) | 긴 시퀀스의 gradient 폭발 방지 |
| AdamW + CosineAnnealingLR | weight decay 분리, 학습률 안정적 감소 |
| alpha=0.6 (CE > Recon) | 분류 정확도 우선, 재구성은 보조 역할 |
| window_size=50, overlap=25 | 논문 RoughLSTM과 동일 설정 유지 |

---

## Dependencies

```
torch >= 2.0
numpy
pandas
scikit-learn   # roc_auc_score (test_cnc.py)
tqdm
tensorboard >= 1.14
```
