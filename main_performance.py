"""
main_performance.py — Step 3 of the AttentionGraph pipeline.

Loads the prediction outputs (pred.npy / true.npy) from the most recent
iTransformer experiment, computes regression metrics (R², MSE, MAE) per
sensor, prints a ranked report, and saves a 4-panel performance dashboard.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis.result_loader import find_latest_result_folder, load_sensor_names
from src.visualization.styles import set_style


# ---------------------------------------------------------------------------
# Data loading helpers (performance-specific — not in result_loader.py)
# ---------------------------------------------------------------------------

def _load_prediction_arrays(folder_path: str):
    """
    Load pred.npy and true.npy from a result folder.

    :returns: ``(preds, trues)`` — both shaped ``[Samples, Pred_Len, Features]``.
    :raises FileNotFoundError: If either file is missing.
    """
    pred_path = os.path.join(folder_path, 'pred.npy')
    true_path = os.path.join(folder_path, 'true.npy')

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            "pred.npy or true.npy not found in the result folder.\n"
            "Did you run training with '--do_predict'?"
        )

    preds = np.load(pred_path)
    trues = np.load(true_path)

    print(f"📊 Prediction shape: {preds.shape}  (Samples, SeqLen, Features)")
    return preds, trues


# ---------------------------------------------------------------------------
# Metrics computation & reporting
# ---------------------------------------------------------------------------

def _compute_sensor_metrics(preds_flat: np.ndarray, trues_flat: np.ndarray,
                             feature_names: list) -> list:
    """Compute per-sensor R², MSE, MAE; return list of dicts sorted by R²."""
    n_features = preds_flat.shape[1]
    records = []
    for i in range(n_features):
        name = feature_names[i] if i < len(feature_names) else f"Sensor_{i}"
        records.append({
            'id':   i,
            'name': name,
            'r2':   r2_score(trues_flat[:, i], preds_flat[:, i]),
            'mse':  mean_squared_error(trues_flat[:, i], preds_flat[:, i]),
            'mae':  mean_absolute_error(trues_flat[:, i], preds_flat[:, i]),
        })
    return sorted(records, key=lambda m: m['r2'])


def _print_performance_report(sorted_metrics: list,
                               global_r2: float,
                               global_mse: float,
                               global_mae: float) -> None:
    print("\n" + "=" * 80)
    print("🏆 Model Performance Report")
    print(f"   Global R²  : {global_r2:.4f}  (1.0 is perfect)")
    print(f"   Global MSE : {global_mse:.4f}  (original scale)")
    print(f"   Global MAE : {global_mae:.4f}  (original scale)")
    print("-" * 80)
    print(f"{'Feature Name':<25} | {'R² Score':<10} | {'MSE':<12} | {'MAE':<8}")
    print("-" * 80)
    for m in sorted_metrics:
        label = m['name'][:22] + '..' if len(m['name']) > 22 else m['name']
        print(f"{label:<25} | {m['r2']:>8.4f}   | {m['mse']:>12.4f} | {m['mae']:>8.4f}")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Dashboard plotting
# ---------------------------------------------------------------------------

def _plot_dashboard(preds: np.ndarray, trues: np.ndarray,
                    sensor_metrics: list, feature_names: list,
                    global_r2: float, global_mse: float,
                    save_dir: str) -> None:
    """Render and save a 4-panel performance dashboard."""
    N, L, F = preds.shape
    preds_flat = preds.reshape(-1, F)
    trues_flat = trues.reshape(-1, F)

    fig = plt.figure(figsize=(22, 14))
    plt.suptitle(
        f"Model Performance Dashboard\n"
        f"Global $R^2$={global_r2:.4f} | MSE={global_mse:.4f}",
        fontsize=22, weight='bold',
    )

    # ── Panel 1: Scatter – Predicted vs. Ground Truth ─────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    sample_idx = np.random.choice(preds_flat.shape[0],
                                   size=min(5000, preds_flat.shape[0]),
                                   replace=False)
    y_pred_s = preds_flat[sample_idx].flatten()
    y_true_s = trues_flat[sample_idx].flatten()
    plot_idx = np.random.choice(len(y_pred_s), size=min(2000, len(y_pred_s)), replace=False)

    sns.scatterplot(x=y_true_s[plot_idx], y=y_pred_s[plot_idx],
                    alpha=0.6, ax=ax1, color='#4C72B0', edgecolor=None)
    lo, hi = min(y_true_s.min(), y_pred_s.min()), max(y_true_s.max(), y_pred_s.max())
    ax1.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Fit')
    ax1.set_title("Prediction vs. Ground Truth", weight='bold', fontsize=16)
    ax1.set_xlabel("Ground Truth (Original Value)")
    ax1.set_ylabel("Predicted Value")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ── Panel 2: Error Distribution ───────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    residuals = (preds_flat - trues_flat).flatten()
    res_sample = np.random.choice(residuals, size=min(10_000, len(residuals)), replace=False)
    sns.histplot(res_sample, kde=True, bins=50, ax=ax2,
                 color='#55A868', stat='density', alpha=0.6)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_title("Error Distribution (Residuals)", weight='bold', fontsize=16)
    ax2.set_xlabel("Error (Pred − True)")

    # ── Panel 3: Per-Sensor R² Bar Chart ──────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    # sensor_metrics is sorted by R²; restore original feature order for bar chart
    ordered = sorted(sensor_metrics, key=lambda m: m['id'])
    r2_vals = [m['r2'] for m in ordered]
    colors = ['#C44E52' if v < 0 else '#55A868' if v > 0.8 else '#4C72B0'
              for v in r2_vals]
    sns.barplot(x=list(range(F)), y=r2_vals, ax=ax3,
                palette=colors, hue=list(range(F)), legend=False)
    ax3.set_title("Predictability per Sensor ($R^2$ Score)", weight='bold', fontsize=16)
    ax3.set_xlabel("Sensor Index")
    ax3.set_ylabel("$R^2$ Score (Max=1.0)")
    ax3.set_ylim(bottom=max(-1.0, min(r2_vals) - 0.1), top=1.05)
    ax3.axhline(0, color='black', linewidth=1)

    # ── Panel 4: Forecast Showcase ────────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    good_ids = [m['id'] for m in sensor_metrics if m['r2'] > 0.5] or list(range(F))
    feat_id = np.random.choice(good_ids)
    sample_id = np.random.randint(0, N)
    feat_name = feature_names[feat_id] if feat_id < len(feature_names) else f"S_{feat_id}"

    ax4.plot(trues[sample_id, :, feat_id], label='Ground Truth',
             marker='o', markersize=5, linewidth=2)
    ax4.plot(preds[sample_id, :, feat_id], label='Prediction',
             marker='x', markersize=5, linestyle='--', linewidth=2)
    ax4.set_title(f"Forecast Example: {feat_name} (Sample {sample_id})",
                  weight='bold', fontsize=16)
    ax4.set_xlabel("Time Step (Prediction Window)")
    ax4.set_ylabel("Value")
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.5)

    # Save
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    save_path = os.path.join(save_dir, "performance_dashboard.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Dashboard saved to: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    set_style()

    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    results_root = config['paths']['results_dir']
    data_path = config['paths']['processed_csv']

    try:
        latest_folder = find_latest_result_folder(results_root)
        print(f"🚀 Analyzing experiment: {os.path.basename(latest_folder)}")

        preds, trues = _load_prediction_arrays(latest_folder)

        # Use sensor names from the processed CSV (same order as model features)
        feature_names = load_sensor_names(data_path)

        N, L, F = preds.shape
        preds_flat, trues_flat = preds.reshape(-1, F), trues.reshape(-1, F)

        global_r2  = r2_score(trues_flat, preds_flat)
        global_mse = mean_squared_error(trues_flat, preds_flat)
        global_mae = mean_absolute_error(trues_flat, preds_flat)

        sensor_metrics = _compute_sensor_metrics(preds_flat, trues_flat, feature_names)
        _print_performance_report(sensor_metrics, global_r2, global_mse, global_mae)

        figures_dir = os.path.join(latest_folder, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        _plot_dashboard(preds, trues, sensor_metrics, feature_names,
                        global_r2, global_mse, figures_dir)

    except Exception as exc:
        print(f"❌ Critical error: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
