"""
main_diagnosis.py — Step 5 of the AttentionGraph pipeline.

Compares attention maps from a 'healthy' experiment run against a 'faulty'
run to detect anomalies and identify the most affected sensor pairs.

Paths to both experiment result folders are read from ``config/settings.yaml``
under the ``diagnosis`` section so no code changes are needed when switching
datasets.

Expected settings.yaml structure::

    diagnosis:
      healthy_result_dir: "./results/<healthy_experiment_id>"
      faulty_result_dir:  "./results/<faulty_experiment_id>"
      drift_threshold: 0.01
      top_k: 20
"""

import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis.result_loader import load_sensor_names, load_attention_weights
from src.diagnosis.attention_drift import AttentionDriftDiagnoser


def main() -> None:
    print("🚀 Starting Diagnosis Pipeline...")

    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    diag_cfg = config.get('diagnosis', {})
    healthy_dir = diag_cfg.get('healthy_result_dir')
    faulty_dir  = diag_cfg.get('faulty_result_dir')
    data_path   = config['paths']['processed_csv']

    if not healthy_dir or not faulty_dir:
        raise ValueError(
            "Please set 'diagnosis.healthy_result_dir' and "
            "'diagnosis.faulty_result_dir' in config/settings.yaml."
        )

    # Load sensor names from the processed CSV header
    feature_names = load_sensor_names(data_path)
    n_sensors = len(feature_names)
    print(f"📋 Detected {n_sensors} sensor features.")

    # Load per-sample attention maps: [Samples, N, N]
    print("📥 Loading healthy attention maps...")
    normal_maps = load_attention_weights(healthy_dir, n_sensors)

    print("📥 Loading faulty attention maps...")
    faulty_maps = load_attention_weights(faulty_dir, n_sensors)

    # Ensure both are 3-D [Samples, N, N]
    if normal_maps.ndim != 3:
        normal_maps = normal_maps.reshape(-1, n_sensors, n_sensors)
    if faulty_maps.ndim != 3:
        faulty_maps = faulty_maps.reshape(-1, n_sensors, n_sensors)

    # Represent the faulty scenario as the mean of its samples
    test_map = np.mean(faulty_maps, axis=0)
    print(f"📊 Test map shape: {test_map.shape}")

    # Build diagnoser config from YAML
    diagnoser_config = {
        'feature_names':   feature_names,
        'drift_threshold': diag_cfg.get('drift_threshold', 0.5),
        'top_k':           diag_cfg.get('top_k', 20),
    }

    # Instantiate and run diagnosers
    # Additional diagnosers (FingerprintDiagnoser, GraphRCADiagnoser) can be
    # added to this list once their logic is implemented.
    diagnosers = [
        AttentionDriftDiagnoser(diagnoser_config),
    ]

    print("\n🧠 Fitting diagnosers on normal data...")
    for diagnoser in diagnosers:
        diagnoser.fit(normal_maps)

    print("\n🔍 Diagnosing test (faulty) sample...")
    for diagnoser in diagnosers:
        result = diagnoser.diagnose(test_map)

        print(f"\n{'─' * 60}")
        print(f"Report from {diagnoser.__class__.__name__}")
        print(f"{'─' * 60}")
        print(str(result))

        if result.is_anomaly:
            print("⚠️  Detailed Evidence (Top-K changed edges):")
            for item in result.evidence:
                direction = "↑" if item['type'] == 'weight_increase' else "↓"
                print(
                    f"   {direction} {item['source']} → {item['target']}: "
                    f"Δ={item['change_magnitude']:.4f} ({item['type']})"
                )
        else:
            print("✅ System looks healthy according to this metric.")


if __name__ == "__main__":
    main()
