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

def main():
    print("🚀 Starting Diagnosis Pipeline...")

    config = {
        'feature_names': [f"Sensor_{i}" for i in range(35)],
        'drift_threshold': 2.0,
        'top_k': 3
    }

    # Factory pattern to instantiate diagnosers
    # Execute diagnosers sequentially
    diagnosers = [
        AttentionDriftDiagnoser(config),
        GraphRCADiagnoser(config) # No implement yet
    ]

    # Load attention_weights.npy in the integration testing
    print("📥 Loading training data for baseline...")
    features, normal_maps, test_sample_map = load_dummy_data()

    # Fit
    for diagnoser in diagnosers:
        diagnoser.fit(normal_maps)

    # 5. Inference
    print("\n🔍 Diagnosing test sample...")

    for diagnoser in diagnosers:
        result = diagnoser.diagnose(test_sample_map)

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