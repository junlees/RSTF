"""
Loss functions for TransformerAutoencoder anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Simple losses ────────────────────────────────────────────────────────────

def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    """Standard cross-entropy for classification logits."""
    if isinstance(output, (tuple, list)):
        output = output[0]  # (logits, reconstruction) → take logits
    return F.cross_entropy(output, target)


def reconstruction_loss(output, input_data, reduction="mean"):
    """
    MSE reconstruction loss.
    output     : (B, window_size, n_features)  ← decoder output
    input_data : (B, window_size, n_features)  ← original input
    """
    if isinstance(output, (tuple, list)):
        output = output[1]  # take reconstruction part
    return F.mse_loss(output, input_data, reduction=reduction)


# ─── Combined loss ────────────────────────────────────────────────────────────

class CombinedAnomalyLoss(nn.Module):
    """
    Weighted combination of cross-entropy classification loss and
    MSE reconstruction loss.

        L = alpha * CE(logits, labels) + (1 - alpha) * MSE(reconstruction, input)

    Parameters
    ──────────
    alpha : weight for classification loss  (0 → pure reconstruction, 1 → pure CE)
    """

    def __init__(self, alpha: float = 0.6):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target, input_data):
        """
        output     : tuple (logits, reconstruction)
        target     : (B,) long tensor with class labels
        input_data : (B, window_size, n_features) original input
        """
        logits, reconstruction = output

        ce_loss   = F.cross_entropy(logits, target)
        recon_loss = F.mse_loss(reconstruction, input_data)

        loss = self.alpha * ce_loss + (1.0 - self.alpha) * recon_loss
        return loss, ce_loss, recon_loss


def combined_anomaly_loss(output, target, input_data=None, alpha=0.6):
    """
    Functional wrapper for CombinedAnomalyLoss.
    If input_data is None only CE loss is used (fallback).
    """
    logits, reconstruction = output
    ce_loss = F.cross_entropy(logits, target)

    if input_data is not None:
        recon_loss = F.mse_loss(reconstruction, input_data)
        return alpha * ce_loss + (1.0 - alpha) * recon_loss
    return ce_loss
