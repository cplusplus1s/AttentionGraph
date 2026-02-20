"""
main_diagnosis.py — Enhanced fault diagnosis with path tracing.

Compares attention maps from healthy vs. faulty experiments and uses
both drift detection and path tracing to identify root causes.

Usage::

    python main_diagnosis.py
"""

import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis.result_loader import load_sensor_names, load_attention_weights
from src.diagnosis.attention_drift import AttentionDriftDiagnoser
from src.diagnosis.path_tracing import PathTracingDiagnoser


def main() -> None:
    print("🚀 Enhanced Fault Diagnosis Pipeline\n")

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
    print(f"📋 Detected {n_sensors} sensor features.\n")

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
    print(f"📊 Test map shape: {test_map.shape}\n")

    # Build diagnoser configs
    drift_config = {
        'feature_names':   feature_names,
        'drift_threshold': diag_cfg.get('drift_threshold', 0.5),
        'top_k':           diag_cfg.get('top_k', 20),
    }

    path_config = {
        'feature_names':   feature_names,
        'path_threshold':  0.05,  # Minimum edge strength to follow
        'max_depth':       4,     # Maximum hops to trace back
    }

    # Instantiate diagnosers
    diagnosers = [
        AttentionDriftDiagnoser(drift_config),
        PathTracingDiagnoser(path_config),
    ]

    print("🧠 Fitting diagnosers on normal data...\n")
    for diagnoser in diagnosers:
        diagnoser.fit(normal_maps)

    print("🔍 Diagnosing test (faulty) sample...\n")

    # Run diagnosis
    results = {}
    for diagnoser in diagnosers:
        result = diagnoser.diagnose(test_map)
        results[diagnoser.__class__.__name__] = result

        print("─" * 70)
        print(f"Report from {diagnoser.__class__.__name__}")
        print("─" * 70)
        print(str(result))

        if result.is_anomaly:
            print("\n⚠️  Detailed Evidence:")

            if diagnoser.__class__.__name__ == 'AttentionDriftDiagnoser':
                # Show top changed edges
                for item in result.evidence[:10]:  # Top 10
                    direction = "↑" if item['type'] == 'weight_increase' else "↓"
                    print(
                        f"   {direction} {item['source']} → {item['target']}: "
                        f"Δ={item['change_magnitude']:.4f} ({item['type']})"
                    )

            elif diagnoser.__class__.__name__ == 'PathTracingDiagnoser':
                # Show traced propagation paths
                for item in result.evidence:
                    path_str = " → ".join(item['path'])
                    print(f"\n   🔗 Propagation Path:")
                    print(f"      {path_str}")
                    print(f"      Root Cause Candidate: {item['root_cause_candidate']}")
                    print(f"      Path Strengths: {[f'{s:.3f}' for s in item['path_strength']]}")
        else:
            print("✅ System looks healthy according to this metric.")

        print("\n")

    # Save path visualization
    if 'PathTracingDiagnoser' in results:
        path_result = results['PathTracingDiagnoser']
        path_diagnoser = [d for d in diagnosers if isinstance(d, PathTracingDiagnoser)][0]

        output_dir = os.path.join(faulty_dir, "figures")
        os.makedirs(output_dir, exist_ok=True)

        print("🎨 Generating path visualization...")
        path_diagnoser.visualize_paths(
            test_map,
            path_result,
            save_path=os.path.join(output_dir, "fault_propagation_paths.png")
        )

    print("="*70)
    print("✅ Diagnosis Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
