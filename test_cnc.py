"""
test_cnc.py — Evaluation script for CNC TransformerAnomalyDetector
───────────────────────────────────────────────────────────────────
Includes Rough Set post-processing (as in the reference paper) applied
to the classification probability instead of LSTM output.

Usage:
    python test_cnc.py -r saved/models/CNC_TransformerAutoencoder/XXXX/model_best.pth
    python test_cnc.py -r <checkpoint> -c config_cnc_transformer.json
    python test_cnc.py -r <checkpoint> --epsilon 0.05 --plot
    python test_cnc.py -r <checkpoint> --plot --plot-dir results/figures
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
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert boolean mask to list of (start, end) index intervals."""
    intervals = []
    in_interval = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_interval:
            start = i
            in_interval = True
        elif not v and in_interval:
            intervals.append((start, i))
            in_interval = False
    if in_interval:
        intervals.append((start, len(mask)))
    return intervals


def save_reconstruction_plot(
    all_inputs: np.ndarray,       # (N, W, F) — z-scored window data
    all_recon_errors: np.ndarray, # (N,)       — per-window MSE
    all_probs: np.ndarray,        # (N,)       — P(anomaly)
    all_labels: np.ndarray,       # (N,)       — ground-truth labels
    regions: np.ndarray,          # (N,)       — 'upper'|'lower'|'boundary'
    epsilon: float,
    stride: int,
    save_path: Path,
    max_windows: int = 5000,
) -> None:
    """
    Save a multi-panel figure showing:
      • Original time series (3 channels) coloured by per-step reconstruction error
      • RST boundary-region intervals shaded in orange
      • Ground-truth label strip (thin coloured bar)
      • Probability panel with ε thresholds

    Windows are mapped back to a continuous time axis using the given stride.
    When N > max_windows the sequence is uniformly subsampled before plotting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch

    N, W, F = all_inputs.shape
    ch_names = ["X-axis", "Y-axis", "Z-axis"][:F]

    # ── optional subsampling ─────────────────────────────────────────────
    if N > max_windows:
        idx = np.linspace(0, N - 1, max_windows, dtype=int)
        all_inputs       = all_inputs[idx]
        all_recon_errors = all_recon_errors[idx]
        all_probs        = all_probs[idx]
        all_labels       = all_labels[idx]
        regions          = regions[idx]
        N = max_windows

    # ── reconstruct continuous signal from overlapping windows ───────────
    T = (N - 1) * stride + W

    signal    = np.zeros((T, F), dtype=np.float32)
    error_map = np.zeros(T, dtype=np.float64)   # averaged reconstruction MSE
    prob_map  = np.zeros(T, dtype=np.float64)   # averaged P(anomaly)
    count_map = np.zeros(T, dtype=np.int32)

    for i in range(N):
        s, e = i * stride, i * stride + W
        signal[s:e]    += all_inputs[i]
        error_map[s:e] += all_recon_errors[i]
        prob_map[s:e]  += all_probs[i]
        count_map[s:e] += 1

    valid = count_map > 0
    signal[valid]    /= count_map[valid, None]
    error_map[valid] /= count_map[valid]
    prob_map[valid]  /= count_map[valid]

    # ── per-timestep RST region (majority vote over overlapping windows) ─
    region_votes = {"upper": np.zeros(T), "lower": np.zeros(T), "boundary": np.zeros(T)}
    for i in range(N):
        s, e = i * stride, i * stride + W
        region_votes[regions[i]][s:e] += 1

    region_stack = np.stack([region_votes["upper"],
                              region_votes["lower"],
                              region_votes["boundary"]], axis=1)
    region_idx   = region_stack.argmax(axis=1)          # 0=upper,1=lower,2=boundary
    ts_boundary  = region_idx == 2                       # boolean mask

    # ── per-timestep ground-truth label strip (window centre → timestep) ─
    label_strip = np.full(T, -1, dtype=int)
    for i in range(N):
        centre = i * stride + W // 2
        if centre < T:
            label_strip[centre] = all_labels[i]
    # forward-fill gaps
    last = -1
    for t in range(T):
        if label_strip[t] >= 0:
            last = label_strip[t]
        elif last >= 0:
            label_strip[t] = last

    t = np.arange(T)

    # ── colour normalisation (shared across channels) ────────────────────
    err_min, err_max = error_map.min(), error_map.max()
    if err_max - err_min < 1e-12:           # flat error — avoid div-by-zero
        err_max = err_min + 1e-6
    norm = mcolors.Normalize(vmin=err_min, vmax=err_max)
    cmap = plt.cm.RdYlGn_r                  # red = high error, green = low

    boundary_color = "#FF8C00"              # dark-orange
    label_colors   = {0: "#EF5350", 1: "#42A5F5", -1: "#BDBDBD"}

    # ── layout: F signal panels + 1 probability panel ────────────────────
    n_rows   = F + 1
    row_h    = [3.0] * F + [2.5]
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(20, sum(row_h) + 1.0),
        gridspec_kw={"height_ratios": row_h},
        sharex=True,
    )

    # ─── signal panels ────────────────────────────────────────────────────
    for f_idx in range(F):
        ax = axes[f_idx]
        y  = signal[:, f_idx]

        # Colored line via LineCollection
        pts  = np.stack([t, y], axis=1).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)  # (T-1, 2, 2)
        lc   = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.0, alpha=0.9)
        lc.set_array(error_map[:-1])
        ax.add_collection(lc)

        # RST boundary shading
        for s_t, e_t in _get_intervals(ts_boundary):
            ax.axvspan(s_t, e_t, color=boundary_color, alpha=0.18, linewidth=0)

        # Ground-truth label strip (very thin bar at the bottom of the axes)
        strip_y_min = y.min() - 0.8
        strip_y_max = strip_y_min + 0.25
        for lval, lcolor in [(0, label_colors[0]), (1, label_colors[1])]:
            mask = label_strip == lval
            for s_t, e_t in _get_intervals(mask):
                ax.fill_between(
                    [s_t, e_t], strip_y_min, strip_y_max,
                    color=lcolor, alpha=0.6, linewidth=0,
                )

        ax.set_xlim(0, T - 1)
        ax.set_ylim(strip_y_min - 0.1, y.max() + 0.5)
        ax.set_ylabel(ch_names[f_idx], fontsize=10)
        ax.set_title(f"Channel {ch_names[f_idx]}  —  coloured by reconstruction error",
                     fontsize=10, pad=4)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.01)
        cb.set_label("Recon. MSE", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # ─── probability panel ────────────────────────────────────────────────
    ax_p = axes[F]
    ax_p.plot(t, prob_map, color="steelblue", linewidth=0.9, label="P(anomaly)", zorder=3)

    upper_th = 0.5 + epsilon
    lower_th = 0.5 - epsilon
    ax_p.axhline(upper_th, color="#D32F2F", linestyle="--", linewidth=1.0,
                 label=f"upper  {upper_th:.2f}")
    ax_p.axhline(lower_th, color="#388E3C", linestyle="--", linewidth=1.0,
                 label=f"lower  {lower_th:.2f}")
    ax_p.axhspan(lower_th, upper_th, color=boundary_color, alpha=0.12, linewidth=0,
                 label=f"boundary zone  (ε={epsilon})")

    # RST boundary shading in probability panel too
    for s_t, e_t in _get_intervals(ts_boundary):
        ax_p.axvspan(s_t, e_t, color=boundary_color, alpha=0.22, linewidth=0)

    ax_p.set_xlim(0, T - 1)
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.set_ylabel("P(anomaly)", fontsize=10)
    ax_p.set_xlabel("Time step (sample)", fontsize=10)
    ax_p.set_title("Classification probability with RST boundary region", fontsize=10, pad=4)
    ax_p.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # ─── shared legend for ground-truth strip + boundary ─────────────────
    legend_handles = [
        Patch(facecolor=label_colors[0], alpha=0.6, label="GT: NOK (anomaly)"),
        Patch(facecolor=label_colors[1], alpha=0.6, label="GT: OK  (normal)"),
        Patch(facecolor=boundary_color,  alpha=0.4, label=f"RST boundary region (ε={epsilon})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "CNC Anomaly Detection — Reconstruction Error & RST Boundary Analysis",
        fontsize=13, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    config,
    checkpoint_path,
    epsilon: float = 0.1,
    plot: bool = False,
    plot_dir: Path | None = None,
    plot_max_windows: int = 5000,
):
    logger = config.get_logger("test")

    # ── Data ─────────────────────────────────────────────────────────────
    data_loader_args = config["data_loader"]["args"].copy()
    data_loader_args.update({
        "batch_size"       : 256,
        "shuffle"          : False,
        "validation_split" : 0.0,
        "training"         : False,
    })
    from data_loader.cnc_data_loaders import CNCDataLoader
    data_loader = CNCDataLoader(**data_loader_args)

    stride = (
        data_loader_args["window_size"] - data_loader_args.get("overlap", 25)
    )

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
    all_probs         = []
    all_labels        = []
    all_recon_errors  = []
    all_inputs_list   = [] if plot else None   # only collect when needed

    total_recon = 0.0
    n_samples   = 0

    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluating"):
            data   = data.to(device)
            logits, reconstruction = model(data)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(anomaly)
            all_probs.extend(probs.tolist())
            all_labels.extend(target.numpy().tolist())

            # Per-sample reconstruction MSE  — (B,)
            recon_mse = ((reconstruction.cpu() - data.cpu()) ** 2).mean(dim=(1, 2))
            all_recon_errors.extend(recon_mse.numpy().tolist())
            total_recon += recon_mse.sum().item()
            n_samples   += len(data)

            if plot:
                all_inputs_list.append(data.cpu().numpy())

    all_probs        = np.array(all_probs)
    all_labels       = np.array(all_labels)
    all_recon_errors = np.array(all_recon_errors)

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
    n_boundary   = np.sum(regions == "boundary")
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

    # ── Visualization ─────────────────────────────────────────────────────
    if plot:
        all_inputs = np.concatenate(all_inputs_list, axis=0)  # (N, W, F)

        if plot_dir is None:
            plot_dir = Path(checkpoint_path).parent
        else:
            plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        ckpt_stem = Path(checkpoint_path).stem
        save_path = plot_dir / f"{ckpt_stem}_recon_rst_eps{epsilon:.3f}.png"

        save_reconstruction_plot(
            all_inputs       = all_inputs,
            all_recon_errors = all_recon_errors,
            all_probs        = all_probs,
            all_labels       = all_labels,
            regions          = regions,
            epsilon          = epsilon,
            stride           = stride,
            save_path        = save_path,
            max_windows      = plot_max_windows,
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNC Transformer Autoencoder — Test")
    parser.add_argument("-c", "--config", default=None, type=str)
    parser.add_argument("-r", "--resume", required=True,  type=str,
                        help="path to model checkpoint")
    parser.add_argument("-d", "--device", default=None,   type=str)
    parser.add_argument("--epsilon",      default=0.1,    type=float,
                        help="Rough Set uncertainty threshold ε  (default: 0.1)")
    parser.add_argument("--plot",         action="store_true",
                        help="save reconstruction-error + RST boundary figure")
    parser.add_argument("--plot-dir",     default=None,   type=str,
                        help="directory to save figures (default: alongside checkpoint)")
    parser.add_argument("--plot-max-windows", default=5000, type=int,
                        help="max windows to plot (uniformly subsampled, default: 5000)")

    parsed = parser.parse_args()
    config  = ConfigParser.from_args(parser)

    evaluate(
        config,
        config.resume,
        epsilon          = parsed.epsilon,
        plot             = parsed.plot,
        plot_dir         = parsed.plot_dir,
        plot_max_windows = parsed.plot_max_windows,
    )
