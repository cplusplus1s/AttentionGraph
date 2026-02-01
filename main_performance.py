import os
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    plt.rcParams['font.family'] = 'sans-serif'

def find_latest_result_folder(results_root):
    """
    智能查找最新的实验结果文件夹
    支持 sensor_analysis_short, sensor_analysis_merged 等各种命名
    """
    # 优先找 sensor_analysis 开头的
    patterns = [
        os.path.join(results_root, "sensor_analysis*"),
        os.path.join(results_root, "Exp*"), # 兼容 iTransformer 原生命名
        os.path.join(results_root, "*")     # 保底
    ]

    for p in patterns:
        folders = [f for f in glob.glob(p) if os.path.isdir(f)]
        if folders:
            # 返回修改时间最新的那个
            return max(folders, key=os.path.getmtime)

    raise FileNotFoundError(f"❌ No result folders found in {results_root}")

def load_metrics_data(folder_path):
    print(f"📂 Loading results from: {folder_path}")

    pred_path = os.path.join(folder_path, 'pred.npy')
    true_path = os.path.join(folder_path, 'true.npy')

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        raise FileNotFoundError("❌ pred.npy or true.npy not found. Did you run the training with --do_predict?")

    preds = np.load(pred_path)
    trues = np.load(true_path)

    # Shape 通常是 [Samples, Pred_Len, Features]
    print(f"📊 Data Shape: {preds.shape}")
    return preds, trues

def plot_performance_dashboard(preds, trues, save_dir):
    """
    绘制类似 ROC 的综合性能评估图
    """
    # 1. 数据扁平化 (用于计算全局指标)
    # 只需要最后一个维度的特征分开，前面的 Time Steps 全部展平
    # [Samples, Pred_Len, Features] -> [N, Features]
    N, L, F = preds.shape
    preds_flat = preds.reshape(-1, F)
    trues_flat = trues.reshape(-1, F)

    # 计算全局 R2
    total_r2 = r2_score(trues_flat, preds_flat)

    fig = plt.figure(figsize=(20, 14))
    plt.suptitle(f"Model Performance Dashboard (Global $R^2$ = {total_r2:.4f})", fontsize=20, weight='bold')

    # --- Subplot 1: Global Prediction vs Truth (Scatter) ---
    ax1 = fig.add_subplot(2, 2, 1)
    # 为了避免点太多卡死，随机采样 5000 个点
    sample_idx = np.random.choice(preds_flat.shape[0], size=min(5000, preds_flat.shape[0]), replace=False)
    # 随机选一个 Feature 来看，或者混合看。这里混合看所有 Feature 的采样
    y_p = preds_flat[sample_idx, :].flatten() # 这里稍微暴力一点，把Feature也展平采样
    y_t = trues_flat[sample_idx, :].flatten()

    # 再次采样以适应绘图
    plot_idx = np.random.choice(len(y_p), size=min(2000, len(y_p)), replace=False)

    sns.scatterplot(x=y_t[plot_idx], y=y_p[plot_idx], alpha=0.5, ax=ax1, color='#4C72B0')

    # 画对角线
    min_val = min(y_t.min(), y_p.min())
    max_val = max(y_t.max(), y_p.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    ax1.set_title("Prediction vs. Ground Truth (Regression Fit)", weight='bold')
    ax1.set_xlabel("Ground Truth")
    ax1.set_ylabel("Prediction")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- Subplot 2: Error Distribution (Residuals) ---
    ax2 = fig.add_subplot(2, 2, 2)
    residuals = (preds_flat - trues_flat).flatten()
    # 采样
    res_sample = np.random.choice(residuals, size=min(10000, len(residuals)), replace=False)

    sns.histplot(res_sample, kde=True, bins=50, ax=ax2, color='#55A868', stat='density')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_title("Error Distribution (Residuals)", weight='bold')
    ax2.set_xlabel("Prediction Error (Pred - True)")

    # --- Subplot 3: Per-Sensor R2 Score (Bar Chart) ---
    ax3 = fig.add_subplot(2, 2, 3)
    # 计算每个 Feature 的 R2
    r2_scores = []
    for i in range(F):
        r2 = r2_score(trues_flat[:, i], preds_flat[:, i])
        r2_scores.append(r2)

    # 简单的 Feature ID (0~F)
    sns.barplot(x=list(range(F)), y=r2_scores, ax=ax3, palette="viridis")
    ax3.set_title("Predictability per Sensor ($R^2$ Score)", weight='bold')
    ax3.set_xlabel("Sensor Feature Index")
    ax3.set_ylabel("$R^2$ Score (Max=1.0)")
    ax3.set_ylim(bottom=max(0, min(r2_scores) - 0.1), top=1.05) # 动态调整Y轴

    # --- Subplot 4: Time Series Forecast Showcase ---
    ax4 = fig.add_subplot(2, 2, 4)
    # 随机选一个样本 (Sample) 和一个特征 (Feature)
    sample_id = np.random.randint(0, N)
    feat_id = np.random.randint(0, F)

    # 画出这一段的预测
    ax4.plot(trues[sample_id, :, feat_id], label='Ground Truth', marker='o', markersize=4)
    ax4.plot(preds[sample_id, :, feat_id], label='Prediction', marker='x', markersize=4, linestyle='--')

    ax4.set_title(f"Forecast Example (Sample {sample_id}, Sensor {feat_id})", weight='bold')
    ax4.set_xlabel("Time Step (Prediction Window)")
    ax4.set_ylabel("Normalized Value")
    ax4.legend()

    # 保存
    save_path = os.path.join(save_dir, "performance_dashboard.png")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Performance dashboard saved to: {save_path}")

def main():
    set_style()

    # 1. 加载配置
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    results_root = config['paths']['results_dir']

    # 2. 找到最新结果
    try:
        latest_folder = find_latest_result_folder(results_root)
        print(f"🚀 Analyzing experiment: {os.path.basename(latest_folder)}")

        # 3. 加载数据
        preds, trues = load_metrics_data(latest_folder)

        # 4. 创建保存目录
        figures_dir = os.path.join(latest_folder, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # 5. 绘图
        plot_performance_dashboard(preds, trues, figures_dir)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()