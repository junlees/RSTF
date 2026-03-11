# CNC Vibration Anomaly Detection — Transformer Autoencoder

Pytorch implementation that replaces the **RoughLSTM** architecture from the
reference paper with a **Transformer Autoencoder**, applied to the
[Bosch CNC Machining Dataset](https://github.com/boschresearch/CNC_Machining).

---

## Architecture

```
Input (B, W=50, F=3)
        │
  Input Projection  (F → d_model=64)
        │
  Positional Encoding
        │
  ┌─────────────────────────────────┐
  │   Transformer Encoder × 3      │  ← Multi-head Self-Attention (4 heads)
  │   (Pre-LayerNorm, d_model=64)  │    + Feed-Forward (dim=256)
  └──────────────┬──────────────────┘
                 │  memory (B, W, 64)
        ┌────────┴────────┐
        │                 │
   Bottleneck         Cross-attention
  (64×50→latent=32)      memory
        │                 │
   Latent Expand      ┌───┴──────────────────────┐
  (32→64×50)          │  Transformer Decoder × 3 │
        │              └───────────┬──────────────┘
        └────────┬─────────────────┘
                 │
  ┌──────────────┴──────────────────┐
  │  Output Projection (64 → F=3)  │  → Reconstruction (B, W, F)
  └─────────────────────────────────┘
        │
  ┌─────────────────┐
  │  Classifier     │  latent → 2-class logits  (normal / anomaly)
  └─────────────────┘

Loss = α·CrossEntropy(logits, labels) + (1−α)·MSE(reconstruction, input)
       α = 0.6  (configurable via "trainer.loss_alpha")
```

### Why Transformer Autoencoder over LSTM?

| Property | LSTM (paper) | Transformer AE (this work) |
|---|---|---|
| Temporal modelling | Sequential, recurrent | Parallel, global self-attention |
| Long-range dependency | Vanishing gradient risk | Full attention window coverage |
| Reconstruction signal | None (classification only) | Explicit decoder + MSE loss |
| Uncertainty handling | Rough-Set post-processing | Same Rough-Set ε-threshold |
| Training speed | Slow (sequential) | Fast (parallelisable) |

---

## Repository Layout

```
cnc_transformer/
├── config_cnc_transformer.json   ← main configuration
├── train_cnc.py                  ← training entry point
├── test_cnc.py                   ← evaluation + Rough-Set post-processing
│
├── model/
│   ├── transformer_model.py      ← TransformerAnomalyDetector
│   ├── loss.py                   ← CombinedAnomalyLoss + helpers
│   └── metric.py                 ← accuracy / F1 / FPR / FNR / ...
│
├── data_loader/
│   └── cnc_data_loaders.py       ← CNCVibrationDataset + CNCDataLoader
│
└── trainer/
    └── anomaly_trainer.py        ← AnomalyTrainer (inherits BaseTrainer)
```

> The `base/`, `logger/`, `utils/`, and `parse_config.py` files are re-used
> unchanged from the original PyTorch template.

---

## Dataset Setup

Download the Bosch CNC Machining Dataset:

```bash
git clone https://github.com/boschresearch/CNC_Machining data/
```

Expected structure after download:

```
data/
  M01/  OP00/  OP01/  ...  OP14/
  M02/  ...
  M03/  ...
```

Each CSV contains three columns: `x_axis`, `y_axis`, `z_axis` (m/s²).
Files named `*_OK.csv` → label=1 (normal), `*_NOK.csv` → label=0 (anomaly).

---

## Quick Start

### Training

```bash
# Replicate M01 experiment from the paper (OP02+OP05+OP08+OP11+OP14)
python train_cnc.py -c config_cnc_transformer.json

# Override hyperparameters via CLI
python train_cnc.py -c config_cnc_transformer.json --lr 0.0001 --alpha 0.7

# Multi-machine training
python train_cnc.py -c config_cnc_transformer.json \
    --machines '["M01","M02","M03"]' \
    --ops '["OP02","OP05","OP08","OP11","OP14"]'

# Resume from checkpoint
python train_cnc.py -r saved/models/CNC_TransformerAutoencoder/XXXX/model_best.pth
```

### Evaluation

```bash
python test_cnc.py \
    -r saved/models/CNC_TransformerAutoencoder/XXXX/model_best.pth \
    --epsilon 0.1
```

Sample output:
```
==================================================
  CNC TransformerAutoencoder — Test Results
==================================================
  accuracy                      : 0.9312
  precision                     : 0.9187
  recall                        : 0.9541
  specificity                   : 0.8934
  f1_score                      : 0.9361
  false_positive_rate           : 0.1066
  false_negative_rate           : 0.0459
  auc_roc                       : 0.9724
  avg_reconstruction_mse        : 0.002341
  boundary_samples_pct          : 8.23
  rough_epsilon                 : 0.1
==================================================
```

### TensorBoard

```bash
tensorboard --logdir saved/log/
```

---

## Configuration Reference

```jsonc
{
  "arch.args": {
    "n_features"         : 3,       // x, y, z accelerometer axes
    "window_size"        : 50,      // sliding window size (samples)
    "d_model"            : 64,      // Transformer embedding dimension
    "nhead"              : 4,       // number of attention heads
    "num_encoder_layers" : 3,       // Transformer encoder depth
    "num_decoder_layers" : 3,       // Transformer decoder depth
    "dim_feedforward"    : 256,     // FFN hidden dimension
    "dropout"            : 0.1,
    "latent_dim"         : 32,      // bottleneck size
    "num_classes"        : 2        // normal / anomaly
  },
  "trainer": {
    "loss_alpha"  : 0.6,            // CE weight; (1-α) for reconstruction
    "epochs"      : 50,
    "early_stop"  : 10,
    "monitor"     : "min val_loss"
  }
}
```

---

## Key Implementation Details

### Rough-Set Post-processing (`test_cnc.py`)
Mirrors the RoughLSTM paper:

```
Δ_ur = [0.5 − ε, 0.5 + ε]          (uncertainty region)

Upper approach : p > 0.5 + ε  → definite anomaly
Lower approach : p < 0.5 − ε  → definite normal
Boundary       : 0.5−ε ≤ p ≤ 0.5+ε → uncertain, resolved by nearest side
```

### Combined Loss
```
L = α · CrossEntropy(logits, y) + (1−α) · MSE(x̂, x)
```
- Classification head learns discriminative features
- Reconstruction head forces the encoder to preserve all signal structure
- Together they provide richer representations than classification alone

### Training Stability
- Pre-LayerNorm (norm_first=True) for stable Transformer training
- Gradient clipping (max_norm=1.0)
- AdamW + CosineAnnealingLR
- Data augmentation: Gaussian noise on NOK windows during training

---

## Dependencies

```
torch>=2.0
numpy
pandas
scikit-learn
tqdm
tensorboard>=1.14
```
