# CNC TransformerAutoencoder — 평가 보고서 2
## Temporal Context Correction for RST Boundary Region

> **실험 일시**: 2026-03-12
> **체크포인트**: `saved/models/CNC_TransformerAutoencoder/0312_040936/model_best.pth` (Epoch 5)
> **데이터**: Bosch CNC Machining Dataset — M01 / OP02, OP05, OP08, OP11, OP14
> **기준 보고서**: `evaluation_report.md`

---

## 1. 배경 및 동기

`evaluation_report.md` (baseline)에서 Rough Set Theory(RST) 경계 영역(boundary region) 처리 방식의
한계가 시각화를 통해 확인되었다.

### 식별된 두 가지 오판 패턴

| 패턴 | 위치 | 현상 | 올바른 판정 |
|------|------|------|-------------|
| **이상 덩어리 내 경계** | 확정 이상(upper) 윈도우 다수 사이 | boundary로 분류 → P≥0.5 기준 정상 판정 | 이상 |
| **정상 구간 고립 스파이크** | 확정 정상(lower) 구간에서 1–2개만 경계 | boundary로 분류 → P≥0.5 기준 이상 판정 | 정상 |

기존 방식: `P >= 0.5 → 이상` (컨텍스트 무시)
개선 방식: **시간적 이웃(k개 앞뒤 윈도우)의 다수결**로 재판정

---

## 2. 방법론 — `apply_context_correction`

### 알고리즘

```
각 boundary 인덱스 i 에 대해:
  neighborhood = regions[max(0, i-k) : i]   ←── 이전 k개
               + regions[i+1 : min(N, i+k+1)] ──→ 이후 k개
  (i 자신 제외, boundary 이웃은 카운트 안 됨)

  n_upper = Σ(neighborhood == "upper")
  n_lower = Σ(neighborhood == "lower")

  판정:
    n_upper > n_lower          → 이상 (1)
    n_lower >= n_upper (동점)  → 정상 (0)  [보수적]
    둘 다 0 (전부 boundary)    → fallback: P >= 0.5
```

### `require_both_sides` 옵션

```
왼쪽(i-k:i)에 upper 존재 AND 오른쪽(i+1:i+k+1)에 upper 존재
  → 이상 (1)   # 이상 덩어리에 둘러싸인 경우만
else
  → 정상 (0)
```

### 파라미터 의미

| 파라미터 | 의미 | 단위 변환 (stride=25, 2kHz) |
|----------|------|-----------------------------|
| k=5 | 앞뒤 각 5개 윈도우 참조 | ±62.5 ms |
| k=10 | 앞뒤 각 10개 윈도우 참조 | ±125 ms |

---

## 3. 실험 설정

| 항목 | 값 |
|------|----|
| 체크포인트 | model_best.pth (Epoch 5, val_loss=0.02807) |
| RST ε | 0.1 (고정) |
| 총 평가 윈도우 | 502,463 |
| Boundary 샘플 수 | 2,679 (0.53%) |
| 테스트 구성 | Baseline / k=5 / k=10 / k=5+both-sides |

---

## 4. 실험 결과

### 4.1 주요 지표 비교

| 구성 | Accuracy | Precision | Recall | F1 | FPR | FNR | AUC-ROC |
|------|----------|-----------|--------|----|-----|-----|---------|
| **Baseline** (k=0) | 98.81% | 99.31% | 99.41% | 99.36% | 9.64% | 0.59% | 99.38% |
| **k=5** | 99.01% | 99.36% | 99.57% | 99.47% | 8.67% | 0.43% | 99.38% |
| **k=10** | **99.03%** | **99.38%** | **99.57%** | **99.48%** | **8.51%** | **0.43%** | 99.38% |
| k=5 + both-sides | 98.98% | 99.34% | 99.57% | 99.46% | 9.17% | 0.43% | 99.38% |

### 4.2 Confusion Matrix 비교

**Baseline (k=0)**
```
                    Predicted
                 Normal   Anomaly
Actual Normal  [ 465,919 |   2,744 ]   총 468,663
Actual Anomaly [   3,259 |  30,541 ]   총  33,800
```

**k=5**
```
                    Predicted
                 Normal   Anomaly
Actual Normal  [ 466,634 |   2,029 ]   총 468,663   (FN  -715 ▼)
Actual Anomaly [   2,931 |  30,869 ]   총  33,800   (FP  -328 ▼)
```

**k=10**
```
                    Predicted
                 Normal   Anomaly
Actual Normal  [ 466,659 |   2,004 ]   총 468,663   (FN  -740 ▼)
Actual Anomaly [   2,878 |  30,922 ]   총  33,800   (FP  -381 ▼)
```

**k=5 + require-both-sides**
```
                    Predicted
                 Normal   Anomaly
Actual Normal  [ 466,647 |   2,016 ]   총 468,663   (FN  -728 ▼)
Actual Anomaly [   3,098 |  30,702 ]   총  33,800   (FP   -161 ▼)
```

### 4.3 Baseline 대비 절대 개선량

| 구성 | ΔAccuracy | ΔFPR | ΔFNR | ΔTP | ΔTN | ΔFP | ΔFN |
|------|-----------|------|------|-----|-----|-----|-----|
| k=5 | +0.20%p | **-0.97%p** | -0.16%p | +715 | +328 | -328 | -715 |
| k=10 | +0.22%p | **-1.13%p** | -0.16%p | +740 | +381 | -381 | -740 |
| k=5 + both | +0.17%p | -0.47%p | -0.16%p | +728 | +161 | -161 | -728 |

### 4.4 경계 샘플 재분류 현황

| 구성 | 재분류 수 | → 이상 | → 정상 | 재분류율 |
|------|-----------|--------|--------|----------|
| k=5 | 1,199 | 793 | 406 | 44.8% |
| k=10 | 1,213 | 786 | 427 | 45.3% |
| k=5 + both-sides | 1,185 | 876 | 309 | 44.2% |

> 전체 boundary 2,679개 중 약 45%가 재판정됨.
> 이상 방향 재분류(793–876)가 정상 방향(309–427)의 약 2배 → 이상 덩어리 내부 boundary가 고립 스파이크보다 더 빈번

---

## 5. 분석

### 5.1 FPR·FNR 동시 감소

context correction이 FPR과 FNR을 **동시에** 개선한다.
- FPR 감소 (→이상 방향 재분류): 이상 덩어리 내 경계 샘플이 이상으로 올바르게 판정
- FNR 감소 (→정상 방향 재분류): 정상 구간 고립 스파이크가 정상으로 올바르게 판정
- 이는 시각화에서 관찰한 두 가지 오판 패턴 모두를 교정함을 실증

### 5.2 k 값별 수확 체감

```
FPR:  baseline 9.64%  →  k=5: 8.67%  →  k=10: 8.51%
개선: (기준)          →  -0.97%p     →  -0.16%p (추가 개선)
```

k=5 → k=10 추가 개선이 k=0 → k=5 대비 약 1/6 수준.
더 넓은 이웃을 참조해도 추가 이득이 작으므로 **k=5가 실용적 균형점**.

### 5.3 `require-both-sides` 효과

FPR 개선이 k=5 (8.67%) 대비 k=5+both (9.17%)로 오히려 악화.
원인: 이상 덩어리의 **끝 부분** boundary 샘플은 한쪽(한방향)만 upper 이웃을 가짐 →
      양쪽 조건 미충족 → 정상 판정 오류.

```
... [upper][upper][BOUNDARY][lower][lower] ...
         ↑ 왼쪽 upper 있음, 오른쪽 upper 없음 → both-sides=False → 정상(오판)
```

결론: `require-both-sides` 옵션은 실용적이지 않음.

### 5.4 AUC-ROC 불변

AUC-ROC=99.38%는 모든 구성에서 동일 — context correction은 고정 임계값
이후 후처리이므로 확률 분포 자체를 바꾸지 않음. 예상된 결과.

---

## 6. 설정별 권장 사항

| 목적 | 권장 설정 |
|------|-----------|
| **최소 누락 이상** (FPR 최소화) | `--context-k 10` (FPR 8.51%) |
| **균형 운영** | `--context-k 5` (FPR 8.67%, 속도 우선) |
| **기존 호환** | 옵션 없음 (baseline) |
| `--require-both-sides` | 비권장 |

---

## 7. 종합 결론

RST boundary region에 대한 **temporal context correction** 적용으로:

- Accuracy: **98.81% → 99.01%** (+0.20%p, k=5 기준)
- FPR: **9.64% → 8.67%** (-0.97%p) — 이상 누락 감소
- FNR: **0.59% → 0.43%** (-0.16%p) — 허위 경보 감소
- F1: **99.36% → 99.47%** (+0.11%p)

전체 502,463개 샘플 중 **단 2,679개(0.53%)** 에 대한 후처리만으로 전반적 지표가 개선되었다.
이는 RST 경계 샘플이 소수이지만, 시간적 맥락을 무시한 단순 임계값 판정이 오류를 유발함을 시사한다.

`model_best.pth`의 save_period 제약 문제 해결(실제 최적 에폭 저장) 시 기저 성능 추가 향상 가능.

---

*Generated by `test_cnc.py --context-k 5` with `model_best.pth` (Epoch 5, val_loss=0.02807)*
