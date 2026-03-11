"""
Evaluation metrics for anomaly detection.
All functions accept model output (tuple of logits, reconstruction) and labels.
"""

import torch
import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def _get_preds(output):
    """Extract class predictions from (logits, reconstruction) or plain logits."""
    if isinstance(output, (tuple, list)):
        logits = output[0]
    else:
        logits = output
    return torch.argmax(logits, dim=1)


def _confusion_components(output, target):
    pred = _get_preds(output)
    tp = ((pred == 1) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    return tp, tn, fp, fn


# ─── metrics ──────────────────────────────────────────────────────────────────

def accuracy(output, target):
    """Overall classification accuracy."""
    pred = _get_preds(output)
    correct = pred.eq(target).sum().item()
    return correct / len(target)


def precision(output, target):
    """Precision = TP / (TP + FP)."""
    tp, _, fp, _ = _confusion_components(output, target)
    return tp / (tp + fp + 1e-8)


def recall(output, target):
    """Recall (TPR / sensitivity) = TP / (TP + FN)."""
    tp, _, _, fn = _confusion_components(output, target)
    return tp / (tp + fn + 1e-8)


def specificity(output, target):
    """Specificity (TNR) = TN / (TN + FP)."""
    _, tn, fp, _ = _confusion_components(output, target)
    return tn / (tn + fp + 1e-8)


def f1_score(output, target):
    """F1 = 2 * Precision * Recall / (Precision + Recall)."""
    p = precision(output, target)
    r = recall(output, target)
    return 2 * p * r / (p + r + 1e-8)


def false_positive_rate(output, target):
    """FPR = FP / (FP + TN)."""
    _, tn, fp, _ = _confusion_components(output, target)
    return fp / (fp + tn + 1e-8)


def false_negative_rate(output, target):
    """FNR = FN / (FN + TP)."""
    tp, _, _, fn = _confusion_components(output, target)
    return fn / (fn + tp + 1e-8)


def reconstruction_error(output, input_data):
    """
    Mean per-sample MSE reconstruction error.
    output     : (logits, reconstruction)  or just reconstruction tensor
    input_data : original input (B, W, F)
    """
    if isinstance(output, (tuple, list)):
        reconstruction = output[1]
    else:
        reconstruction = output
    mse = ((reconstruction - input_data) ** 2).mean(dim=(1, 2))  # (B,)
    return mse.mean().item()


def top_k_acc(output, target, k=2):
    """Top-k accuracy (meaningful when num_classes > 2)."""
    if isinstance(output, (tuple, list)):
        logits = output[0]
    else:
        logits = output
    with torch.no_grad():
        pred = torch.topk(logits, k, dim=1)[1]
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
