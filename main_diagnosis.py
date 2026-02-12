import yaml
import os
import numpy as np
import pandas as pd
from src.diagnosis.attention_drift import AttentionDriftDiagnoser
from src.diagnosis.root_cause_analysis import GraphRCADiagnoser

def load_dummy_data():
    """load dummy data for test"""
    features = [f"Sensor_{i}" for i in range(35)]

    normal_maps = np.random.rand(100, 35, 35)

    # Simulate one fault sample
    anomaly_map = np.mean(normal_maps, axis=0).copy()
    anomaly_map[0, 1] += 0.8 # Sensor_1 -> Sensor_0 weight sharp increase
    anomaly_map[5, 6] -= 0.5 # Sensor_6 -> Sensor_5 weight drop

    return features, normal_maps, anomaly_map

def load_real_attention(folder_path, n_sensors):
    npy_path = os.path.join(folder_path, 'attention_weights.npy')
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Attention weights files not found: {npy_path}")

    raw_attn = np.load(npy_path, allow_pickle=True)

    # raw_attn [Layers, Samples, Heads, N, N_key]
    # clean_attn [Samples, N, N_key]
    avg_attn = np.mean(raw_attn, axis=(0, 2))

    clean_attn = avg_attn[..., :n_sensors, :n_sensors]

    return clean_attn

def main():
    print("🚀 Starting Diagnosis Pipeline...")

    # Refactor!
    HEALTHY_DIR = "./results/MSD_v1_healthy_iTransformer_custom_M_ft96_sl48_ll96_pl64_dm8_nh2_el1_dl128_df1_fctimeF_ebTrue_dtMSD_Exp_projection_0"
    UNHEALTHY_DIR = "./results/MSD_v1_unhealthy_iTransformer_custom_M_ft96_sl48_ll96_pl64_dm8_nh2_el1_dl128_df1_fctimeF_ebTrue_dtMSD_Exp_projection_0"
    DATA_PATH = "./data/processed/matlab_filtered_aligned.csv"

    df = pd.read_csv(DATA_PATH, nrows=1)
    feature_names = [c for c in df.columns if c not in ['date', 'OT']]
    n_sensors = len(feature_names)

    print("📥 Loading real data...")
    normal_maps = load_real_attention(HEALTHY_DIR, n_sensors)

    unhealthy_all = load_real_attention(UNHEALTHY_DIR, n_sensors)

    if normal_maps.ndim > 3:
        normal_maps = normal_maps.reshape(-1, n_sensors, n_sensors)
        unhealthy_all = unhealthy_all.reshape(-1, n_sensors, n_sensors)

    test_sample_map = np.mean(unhealthy_all, axis=0)
    print(f"📊 Debug: test_sample_map shape: {test_sample_map.shape}")

    config = {
        'feature_names': feature_names,
        'drift_threshold': 0.01,
        'top_k': 20
    }

    # Factory pattern to instantiate diagnosers
    # Execute diagnosers sequentially
    diagnosers = [
        AttentionDriftDiagnoser(config),
        # GraphRCADiagnoser(config) # No implement yet
    ]

    # Fit
    for diagnoser in diagnosers:
        diagnoser.fit(normal_maps)

    # Inference
    print("\n🔍 Diagnosing test sample...")

    for diagnoser in diagnosers:
        result = diagnoser.diagnose(test_sample_map)

        print(f"DEBUG: Current Drift Score: {result.description}")

        print(f"\n--- Report from {diagnoser.__class__.__name__} ---")
        print(str(result))

        if result.is_anomaly:
            print("⚠️ Detailed Evidence:")
            for item in result.evidence:
                print(f"   🔴 {item['source']} -> {item['target']}: "
                      f"{item['type']} (Mag: {item['change_magnitude']:.4f})")
        else:
            print("✅ System looks healthy based on this metric.")

if __name__ == "__main__":
    main()