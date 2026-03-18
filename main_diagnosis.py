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
from src.diagnosis.attention_drift import AttentionDriftDiagnoser, SpectralAttentionDriftDiagnoser
from src.diagnosis.path_tracing import PathTracingDiagnoser


def _print_report(name: str, result, feature_names=None):
    """Helper function to print detailed evidence for each diagnoser."""
    print("─" * 70)
    print(f"Report from {name}")
    print("─" * 70)
    print(str(result))

    if result.is_anomaly:
        print("\n⚠️  Detailed Evidence:")

        if name == 'AttentionDriftDiagnoser':
            # Show top changed edges
            for item in result.evidence[:10]:  # Top 10
                direction = "↑" if item['type'] == 'weight_increase' else "↓"
                print(
                    f"   {direction} {item['source']} → {item['target']}: "
                    f"Δ={item['change_magnitude']:.4f} ({item['type']})"
                )

        elif name == 'SpectralAttentionDriftDiagnoser':
            baseline = result.details.get('baseline_gap', 0)
            current = result.details.get('current_gap', 0)
            print(f"   📊 Spectral Gap Drift: {abs(current - baseline):.4f} (Baseline: {baseline:.4f} → Current: {current:.4f})")
            print("   🎯 Root Cause Candidates (based on global TokenRank shift):")
            for item in result.evidence[:10]:  # Top 10
                direction = "↑ Rank increase" if item['type'] == 'rank_increased' else "↓ Rank decrease"
                print(
                    f"      {direction} Sensor [{item['sensor']}]: "
                    f"Global Importance Change Δ={item['importance_change_magnitude']:.4f}"
                )

        elif name == 'PathTracingDiagnoser':
            # Show traced propagation paths
            for item in result.evidence:
                path_str = " → ".join(item['path'])
                print(f"\n   🔗 [{item.get('trace_type', 'Path')}]")
                print(f"      Path: {path_str}")
                print(f"      Root Cause Candidate: {item['root_cause_candidate']}")
                print(f"      Path Strengths: {[f'{s:.3f}' for s in item['path_strength']]}")
    else:
        print("✅ System looks healthy according to this metric.")

    print("\n")


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

    # Batch results
    if not os.path.exists(data_path):
        base, ext = os.path.splitext(data_path)
        data_path_fallback = f"{base}_1{ext}"
        if os.path.exists(data_path_fallback):
            data_path = data_path_fallback
        else:
            raise FileNotFoundError(f"Neither {config['paths']['processed_csv']} nor {data_path_fallback} exists.")

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
        'top_k':           diag_cfg.get('top_k', 10),
    }

    spectral_config = {
        'feature_names': feature_names,
        'spectral_drift_threshold': diag_cfg.get('spectral_drift_threshold', 0.15),
        'top_k': diag_cfg.get('top_k', 10),
    }

    path_config = {
        'feature_names':   feature_names,
        'path_threshold':  diag_cfg.get('path_threshold', 0.05),  # Minimum edge strength to follow
        'max_depth':       diag_cfg.get('max_depth', 4),     # Maximum hops to trace back
        'top_k_starts':     diag_cfg.get('tracing_top_k', 3),  # Number of backward start points
        'max_bfs_branches': diag_cfg.get('max_bfs_branches', 3)  # BFS max branches per node
    }

    # Instantiate diagnosers
    diagnosers = [
        AttentionDriftDiagnoser(drift_config),
        SpectralAttentionDriftDiagnoser(spectral_config),
        PathTracingDiagnoser(path_config),
    ]

    print("🧠 Fitting diagnosers on normal data...\n")
    for diagnoser in diagnosers:
        diagnoser.fit(normal_maps)

    print("\n🔍 Diagnosing test (faulty) sample...\n")

    results = {}

    # 1. Run AttentionDriftDiagnoser
    res_drift = diagnosers[0].diagnose(test_map)
    results['AttentionDriftDiagnoser'] = res_drift
    _print_report(diagnosers[0].__class__.__name__, res_drift, feature_names)

    # 2. Run SpectralAttentionDriftDiagnoser
    res_spectral = diagnosers[1].diagnose(test_map)
    results['SpectralAttentionDriftDiagnoser'] = res_spectral
    _print_report(diagnosers[1].__class__.__name__, res_spectral, feature_names)

    # 3. Extract Top-K Root Causes to PathTracer
    root_candidate_indices = []
    if res_spectral.is_anomaly:
        top_k_roots = diag_cfg.get('tracing_top_k', 3)
        for item in res_spectral.evidence[:top_k_roots]:
            sensor_name = item['sensor']
            if sensor_name in feature_names:
                root_candidate_indices.append(feature_names.index(sensor_name))
        print(f"🔗 [PathTracer] Intercepted Top-{top_k_roots} Root Causes from Spectral: {[feature_names[i] for i in root_candidate_indices]}")

    # 4. Run PathTracingDiagnoser (backward vs forward)
    res_path = diagnosers[2].diagnose(test_map, root_candidates=root_candidate_indices)
    results['PathTracingDiagnoser'] = res_path
    _print_report(diagnosers[2].__class__.__name__, res_path, feature_names)


    # Save path visualization
    if 'PathTracingDiagnoser' in results:
        path_result = results['PathTracingDiagnoser']
        path_diagnoser = [d for d in diagnosers if isinstance(d, PathTracingDiagnoser)][0]

        output_dir = os.path.join(faulty_dir, "figures", "paths")
        os.makedirs(output_dir, exist_ok=True)

        path_diagnoser.visualize_paths(
            test_map,
            path_result,
            save_dir=output_dir
        )

    print("\n" + "═"*70)
    print("🏆 Diagnoser Performance Evaluation (Z-Score)")
    print("═"*70)

    for diagnoser in diagnosers:
        name = diagnoser.__class__.__name__

        # PathTracing to be continued...
        if name == 'PathTracingDiagnoser':
            continue

        healthy_scores = []

        # Calculate heathy baseline score
        for h_map in normal_maps:
            if name == 'AttentionDriftDiagnoser':
                # AttentionDrift
                diff_matrix = np.abs(h_map - diagnoser.baseline_map)
                score = np.linalg.norm(diff_matrix)
            elif name == 'SpectralAttentionDriftDiagnoser':
                # SpectralAttentionDrift
                curr_rank, curr_gap = diagnoser._get_spectral_features(h_map)
                gap_drift = np.abs(curr_gap - diagnoser.baseline_spectral_gap)
                rank_drift = np.sum(np.abs(curr_rank - diagnoser.baseline_token_rank))
                score = float(gap_drift + rank_drift)

            healthy_scores.append(score)

        mu_h = np.mean(healthy_scores)
        sigma_h = np.std(healthy_scores)

        # Calculate testset score
        if name == 'AttentionDriftDiagnoser':
            faulty_score = np.linalg.norm(np.abs(test_map - diagnoser.baseline_map))
        elif name == 'SpectralAttentionDriftDiagnoser':
            curr_rank, curr_gap = diagnoser._get_spectral_features(test_map)
            gap_drift = np.abs(curr_gap - diagnoser.baseline_spectral_gap)
            rank_drift = np.sum(np.abs(curr_rank - diagnoser.baseline_token_rank))
            faulty_score = float(gap_drift + rank_drift)

        # Calculate Z-Score
        # Add 1e-9 to prevent the denominator from being 0
        z_score = (faulty_score - mu_h) / (sigma_h + 1e-9)

        print(f"🎯 {name}:")
        print(f"   • Healthy Mean (μ) : {mu_h:.4f}")
        print(f"   • Healthy Std (σ)  : {sigma_h:.4f}")
        print(f"   • Faulty Score (X) : {faulty_score:.4f}")
        print(f"   ⭐ Z-Score         : {z_score:.2f}\n")

    print("="*70)
    print("✅ Diagnosis Complete!")
    print("="*70)


if __name__ == "__main__":
    main()