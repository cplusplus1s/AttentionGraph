import os
import glob
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def set_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    sns.set_context("talk")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

def load_config():
    """Load settings.yaml"""
    if not os.path.exists('config/settings.yaml'):
        raise FileNotFoundError("❌ Configuration file not found: config/settings.yaml")

    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_feature_names(config, num_features):
    """
    Load the sensor name mapping; if not found, generate a default name (Sensor_0, Sensor_1...)
    """
    mapping_path = 'config/sensor_mapping.json'

    # Try to load mapping file
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)

            if isinstance(mapping, dict):
                names = [mapping.get(str(i), f"Sensor_{i}") for i in range(num_features)]
                print(f"✅ Loaded {len(names)} feature names from {mapping_path}")
                return names
            elif isinstance(mapping, list):
                return mapping[:num_features]
        except Exception as e:
            print(f"⚠️ Failed to load sensor_mapping.json: {e}")

    return [f"Sensor_{i}" for i in range(num_features)]

def find_latest_result_folder(results_root):
    patterns = [
        os.path.join(results_root, "sensor_analysis*"),
        os.path.join(results_root, "sensor_analysis_merged*"),
        os.path.join(results_root, "Exp*"),
        os.path.join(results_root, "*")
    ]

    candidates = []
    for p in patterns:
        folders = [f for f in glob.glob(p) if os.path.isdir(f)]
        candidates.extend(folders)

    if not candidates:
        raise FileNotFoundError(f"❌ No result folders found in {results_root}")

    # Return the one with the latest modification time
    latest = max(candidates, key=os.path.getmtime)
    return latest

def load_metrics_data(folder_path):
    print(f"📂 Loading results from: {folder_path}")

    pred_path = os.path.join(folder_path, 'pred.npy')
    true_path = os.path.join(folder_path, 'true.npy')

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        raise FileNotFoundError("❌ pred.npy or true.npy not found. Did you run training with --do_predict?")

    preds = np.load(pred_path)
    trues = np.load(true_path)

    # Shape [Samples, Pred_Len, Features]
    print(f"📊 Data Shape: {preds.shape} (Samples, SeqLen, Features)")
    return preds, trues

def plot_performance_dashboard(preds, trues, feature_names, save_dir):
    """
    Core function: calculate metrics and plot
    """
    # [Samples, Pred_Len, Features] -> [Total_Steps, Features]
    N, L, F = preds.shape
    preds_flat = preds.reshape(-1, F)
    trues_flat = trues.reshape(-1, F)

    # Calculate global metrics
    # Note: MSE/MAE is based on the original scale.
    total_r2 = r2_score(trues_flat, preds_flat)
    total_mse = mean_squared_error(trues_flat, preds_flat)
    total_mae = mean_absolute_error(trues_flat, preds_flat)

    # Print Performance Report
    print("\n" + "="*80)
    print(f"🏆 Model Performance Report")
    print(f"   Global R²  : {total_r2:.4f} (1.0 is perfect)")
    print(f"   Global MSE : {total_mse:.4f} (Original Scale)")
    print(f"   Global MAE : {total_mae:.4f} (Original Scale)")
    print("-" * 80)
    print(f"{'Feature Name':<25} | {'R² Score':<10} | {'MSE':<12} | {'MAE':<10}")
    print("-" * 80)

    sensor_metrics = []
    for i in range(F):
        r2 = r2_score(trues_flat[:, i], preds_flat[:, i])
        mse = mean_squared_error(trues_flat[:, i], preds_flat[:, i])
        mae = mean_absolute_error(trues_flat[:, i], preds_flat[:, i])

        sensor_metrics.append({
            'name': feature_names[i] if i < len(feature_names) else f"Sensor_{i}",
            'id': i,
            'r2': r2,
            'mse': mse,
            'mae': mae
        })

    # Sort in ascending order of R² (worst result at the top)
    sorted_metrics = sorted(sensor_metrics, key=lambda x: x['r2'])

    for m in sorted_metrics:
        # Truncating overly long names
        name_display = (m['name'][:22] + '..') if len(m['name']) > 22 else m['name']
        print(f"{name_display:<25} | {m['r2']:>8.4f}   | {m['mse']:>12.4f} | {m['mae']:>8.4f}")

    print("="*80 + "\n")

    fig = plt.figure(figsize=(22, 14))
    plt.suptitle(
        f"Model Performance Dashboard\nGlobal $R^2$={total_r2:.4f} | MSE={total_mse:.4f}",
        fontsize=22, weight='bold'
    )

    # Subplot 1: Prediction vs Truth (Scatter)
    ax1 = fig.add_subplot(2, 2, 1)
    # Randomly sample 5000 points to avoid the graph being too large
    sample_size = min(5000, preds_flat.shape[0])
    sample_idx = np.random.choice(preds_flat.shape[0], size=sample_size, replace=False)

    # Plot by combining points from all sensors
    y_p_sample = preds_flat[sample_idx, :].flatten()
    y_t_sample = trues_flat[sample_idx, :].flatten()

    # Only plot 2000 points
    plot_idx = np.random.choice(len(y_p_sample), size=min(2000, len(y_p_sample)), replace=False)

    sns.scatterplot(x=y_t_sample[plot_idx], y=y_p_sample[plot_idx], alpha=0.6, ax=ax1, color='#4C72B0', edgecolor=None)

    # Draw diagonal
    min_val = min(y_t_sample.min(), y_p_sample.min())
    max_val = max(y_t_sample.max(), y_p_sample.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')

    ax1.set_title("Prediction vs. Ground Truth (Regression Fit)", weight='bold', fontsize=16)
    ax1.set_xlabel("Ground Truth (Original Value)")
    ax1.set_ylabel("Predicted Value")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Error Distribution (Residuals)
    ax2 = fig.add_subplot(2, 2, 2)
    residuals = (preds_flat - trues_flat).flatten()

    res_sample = np.random.choice(residuals, size=min(10000, len(residuals)), replace=False)

    sns.histplot(res_sample, kde=True, bins=50, ax=ax2, color='#55A868', stat='density', alpha=0.6)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_title(f"Error Distribution (Residuals)", weight='bold', fontsize=16)
    ax2.set_xlabel("Error (Pred - True)")

    # Subplot 3: Per-Sensor R2 Score (Bar Chart)
    ax3 = fig.add_subplot(2, 2, 3)

    r2_values = [m['r2'] for m in sensor_metrics] # sensor_metrics order by Feature ID 0..N

    # R2 < 0 in red，R2 > 0.8 in green
    colors = ['#C44E52' if v < 0 else '#55A868' if v > 0.8 else '#4C72B0' for v in r2_values]

    sns.barplot(x=list(range(F)), y=r2_values, ax=ax3, palette=colors, hue=list(range(F)), legend=False)
    ax3.set_title("Predictability per Sensor ($R^2$ Score)", weight='bold', fontsize=16)
    ax3.set_xlabel("Sensor Index")
    ax3.set_ylabel("$R^2$ Score (Max=1.0)")
    ax3.set_ylim(bottom=max(-1.0, min(r2_values) - 0.1), top=1.05)
    ax3.axhline(0, color='black', linewidth=1)

    # Subplot 4: Forecast Showcase
    ax4 = fig.add_subplot(2, 2, 4)
    # Randomly select a sensor with decent performance (R2 > 0.5) to showcase; if none are available, use a random sensor.
    good_sensors = [m['id'] for m in sensor_metrics if m['r2'] > 0.5]
    if not good_sensors: good_sensors = list(range(F))

    showcase_feat_id = np.random.choice(good_sensors)
    showcase_sample_id = np.random.randint(0, N)

    feat_name = feature_names[showcase_feat_id] if showcase_feat_id < len(feature_names) else f"S_{showcase_feat_id}"

    # Plot
    ax4.plot(trues[showcase_sample_id, :, showcase_feat_id], label='Ground Truth', marker='o', markersize=5, linewidth=2)
    ax4.plot(preds[showcase_sample_id, :, showcase_feat_id], label='Prediction', marker='x', markersize=5, linestyle='--', linewidth=2)

    ax4.set_title(f"Forecast Example: {feat_name} (Sample {showcase_sample_id})", weight='bold', fontsize=16)
    ax4.set_xlabel("Time Step (Prediction Window)")
    ax4.set_ylabel("Value")
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.5)

    # Save
    save_path = os.path.join(save_dir, "performance_dashboard.png")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Dashboard saved to: {save_path}")

def main():
    set_style()

    try:
        # Config path
        config = load_config()
        results_root = config['paths']['results_dir']

        # Find newest result
        latest_folder = find_latest_result_folder(results_root)
        print(f"🚀 Analyzing experiment: {os.path.basename(latest_folder)}")

        # Load metrics data
        preds, trues = load_metrics_data(latest_folder)

        # Load feature names
        feature_names = load_feature_names(config, preds.shape[2])

        # Create save directory
        figures_dir = os.path.join(latest_folder, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Performance analysis and plot
        plot_performance_dashboard(preds, trues, feature_names, figures_dir)

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()