"""
test_cnc.py — Evaluation script for CNC TransformerAnomalyDetector
───────────────────────────────────────────────────────────────────
Includes Rough Set post-processing (as in the reference paper) applied
to the classification probability instead of LSTM output.

Usage:
    python test_cnc.py -r saved/models/CNC_TransformerAutoencoder/XXXX/model_best.pth
    python test_cnc.py -r <checkpoint> -c config_cnc_transformer.json
"""

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix

import data_loader.cnc_data_loaders as module_data
import model.transformer_model as module_arch
from parse_config import ConfigParser


# ─────────────────────────────────────────────────────────────────────────────
# Rough-Set Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def rough_set_decision(prob_anomaly: np.ndarray, epsilon: float = 0.1):
    """
    Apply Rough Set Theory boundary-region analysis to classification
    probabilities, mirroring the RoughLSTM paper.

    Regions
    ───────
    Upper approach  (definite anomaly) : prob > 0.5 + ε
    Lower approach  (definite normal)  : prob < 0.5 - ε
    Boundary region (uncertain)        : 0.5 - ε ≤ prob ≤ 0.5 + ε

    Boundary samples are resolved by proximity: whichever side of 0.5
    they are closer to (uncertainty score U < 1).

    Returns
    ───────
    predictions : (N,) int array  — 0 = normal, 1 = anomaly
    uncertainty : (N,) float array — normalised uncertainty score
    regions     : (N,) str array  — 'upper' | 'lower' | 'boundary'
    """
    upper_thresh = 0.5 + epsilon
    lower_thresh = 0.5 - epsilon

    predictions = np.zeros(len(prob_anomaly), dtype=int)
    uncertainty = np.abs(prob_anomaly - 0.5) / epsilon          # U_Δ(x)
    regions     = np.full(len(prob_anomaly), "boundary", dtype=object)

    # Upper approach — definite anomaly
    upper_mask = prob_anomaly > upper_thresh
    predictions[upper_mask] = 1
    regions[upper_mask]     = "upper"

    # Lower approach — definite normal
    lower_mask = prob_anomaly < lower_thresh
    predictions[lower_mask] = 0
    regions[lower_mask]     = "lower"

    # Boundary — resolved by nearest class
    boundary_mask = ~(upper_mask | lower_mask)
    predictions[boundary_mask] = (prob_anomaly[boundary_mask] >= 0.5).astype(int)

    return predictions, uncertainty, regions


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(config, checkpoint_path, epsilon: float = 0.1):
    logger = config.get_logger("test")

    # ── Data ─────────────────────────────────────────────────────────────
    # Use test split (no shuffling, no augmentation, full dataset)
    data_loader_args = config["data_loader"]["args"].copy()
    data_loader_args.update({
        "batch_size"       : 256,
        "shuffle"          : False,
        "validation_split" : 0.0,
        "training"         : False,
    })
    from data_loader.cnc_data_loaders import CNCDataLoader
    data_loader = CNCDataLoader(**data_loader_args)

    # ── Model ────────────────────────────────────────────────────────────
    model = config.init_obj("arch", module_arch)
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    # ── Inference ────────────────────────────────────────────────────────
    all_probs    = []
    all_labels   = []
    total_recon  = 0.0
    n_samples    = 0

    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluating"):
            data   = data.to(device)
            logits, reconstruction = model(data)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(anomaly)
            all_probs.extend(probs.tolist())
            all_labels.extend(target.numpy().tolist())

            # Reconstruction error per sample
            recon_mse = ((reconstruction.cpu() - data.cpu()) ** 2).mean(dim=(1, 2))
            total_recon += recon_mse.sum().item()
            n_samples   += len(data)

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ── Rough-Set post-processing ─────────────────────────────────────────
    predictions, uncertainty, regions = rough_set_decision(all_probs, epsilon)

    # ── Metrics ───────────────────────────────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()

    acc         = (tp + tn) / len(all_labels)
    prec        = tp / (tp + fp + 1e-8)
    rec         = tp / (tp + fn + 1e-8)
    spec        = tn / (tn + fp + 1e-8)
    f1          = 2 * prec * rec / (prec + rec + 1e-8)
    fpr         = fp / (fp + tn + 1e-8)
    fnr         = fn / (fn + tp + 1e-8)
    auc         = roc_auc_score(all_labels, all_probs)
    avg_recon   = total_recon / n_samples

    # Boundary region stats
    n_boundary  = np.sum(regions == "boundary")
    pct_boundary = 100.0 * n_boundary / len(all_labels)

    results = {
        "accuracy"              : round(acc,  4),
        "precision"             : round(prec, 4),
        "recall"                : round(rec,  4),
        "specificity"           : round(spec, 4),
        "f1_score"              : round(f1,   4),
        "false_positive_rate"   : round(fpr,  4),
        "false_negative_rate"   : round(fnr,  4),
        "auc_roc"               : round(auc,  4),
        "avg_reconstruction_mse": round(avg_recon, 6),
        "boundary_samples_pct"  : round(pct_boundary, 2),
        "rough_epsilon"         : epsilon,
        "confusion_matrix": {
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
        },
    }

    # ── Log ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("  CNC TransformerAutoencoder — Test Results")
    logger.info("=" * 50)
    for k, v in results.items():
        logger.info(f"  {k:<30s}: {v}")
    logger.info(f"\n  Boundary region : {n_boundary}/{len(all_labels)} "
                f"({pct_boundary:.1f}%) samples (ε={epsilon})")
    logger.info("=" * 50)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="CNC Transformer Autoencoder — Test")
    args.add_argument("-c", "--config", default=None, type=str)
    args.add_argument("-r", "--resume", required=True,  type=str,
                      help="path to model checkpoint")
    args.add_argument("-d", "--device", default=None,   type=str)
    args.add_argument("--epsilon",      default=0.1,    type=float,
                      help="Rough Set uncertainty threshold ε  (default: 0.1)")

    config = ConfigParser.from_args(args)
    results = evaluate(config, config.resume, epsilon=0.1)
