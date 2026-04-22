"""
main_diagnosis.py — Enhanced fault diagnosis with CoExBO preference learning.

Adds a fourth diagnosis stage powered by CoExBO's pairwise preference learning
(CoExBODiagnoser) alongside the existing three diagnosers.

CoExBO contribution
-------------------
Instead of pure threshold-based anomaly scoring, CoExBO learns a GP preference
model that captures *which sensor patterns* experts (or the drift objective)
would rank as the most likely root cause.  The soft-Copeland score then provides
a global ranking from pairwise comparisons — mathematically equivalent to the
"probability of winning a tournament" — which is fused with the attention drift
evidence via a Bayesian prior update (CoExBO_UCB posterior, Eq. 7).

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
from src.diagnosis.coexbo_diagnoser import CoExBODiagnoser


def _print_report(name: str, result, feature_names=None):
    """Helper function to print detailed evidence for each diagnoser."""
    print("─" * 70)
    print(f"Report from {name}")
    print("─" * 70)
    print(str(result))

    if result.is_anomaly:
        print("\n⚠️  Detailed Evidence:")

        if name == 'AttentionDriftDiagnoser':
            for item in result.evidence[:10]:
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
            for item in result.evidence[:10]:
                direction = "↑ Rank increase" if item['type'] == 'rank_increased' else "↓ Rank decrease"
                print(
                    f"      {direction} Sensor [{item['sensor']}]: "
                    f"Global Importance Change Δ={item['importance_change_magnitude']:.4f}"
                )

        elif name == 'PathTracingDiagnoser':
            for item in result.evidence:
                path_str = " → ".join(item['path'])
                print(f"\n   🔗 [{item.get('trace_type', 'Path')}]")
                print(f"      Path: {path_str}")
                print(f"      Root Cause Candidate: {item['root_cause_candidate']}")
                print(f"      Path Strengths: {[f'{s:.3f}' for s in item['path_strength']]}")

        elif name == 'CoExBODiagnoser':
            # ── CoExBO preference-learning output ──────────────────────────
            print(f"\n   🤖 CoExBO Soft-Copeland Ranking (γ={result.details.get('gamma', '?')}):")
            print(f"      Trained on {result.details.get('n_pairs_used', '?')} pairwise comparisons")
            print()
            print(f"   {'Rank':<5} {'Sensor':<28} {'Copeland':>9} {'±Uncert':>9} {'Drift':>8} {'Fused':>8}")
            print(f"   {'-'*5} {'-'*28} {'-'*9} {'-'*9} {'-'*8} {'-'*8}")
            for item in result.evidence:
                sensor_str = item['sensor'][:26]
                print(
                    f"   {item['rank']:<5} {sensor_str:<28} "
                    f"{item['copeland_score']:>9.4f} "
                    f"{item['copeland_uncertainty']:>9.4f} "
                    f"{item['drift_score']:>8.4f} "
                    f"{item['fused_score']:>8.4f}"
                )
    else:
        print("✅ System looks healthy according to this metric.")

    print("\n")


def main() -> None:
    print("🚀 Enhanced Fault Diagnosis Pipeline (with CoExBO Preference Learning)\n")

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

    if not os.path.exists(data_path):
        base, ext = os.path.splitext(data_path)
        data_path_fallback = f"{base}_1{ext}"
        if os.path.exists(data_path_fallback):
            data_path = data_path_fallback
        else:
            raise FileNotFoundError(f"Neither {config['paths']['processed_csv']} nor {data_path_fallback} exists.")

    feature_names = load_sensor_names(data_path)
    n_sensors = len(feature_names)
    print(f"📋 Detected {n_sensors} sensor features.\n")

    print("📥 Loading healthy attention maps...")
    normal_maps = load_attention_weights(healthy_dir, n_sensors)

    print("📥 Loading faulty attention maps...")
    faulty_maps = load_attention_weights(faulty_dir, n_sensors)

    if normal_maps.ndim != 3:
        normal_maps = normal_maps.reshape(-1, n_sensors, n_sensors)
    if faulty_maps.ndim != 3:
        faulty_maps = faulty_maps.reshape(-1, n_sensors, n_sensors)

    test_map = np.mean(faulty_maps, axis=0)
    print(f"📊 Test map shape: {test_map.shape}\n")

    # ── Shared config dicts ─────────────────────────────────────────────────
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
        'feature_names':    feature_names,
        'path_threshold':   diag_cfg.get('path_threshold', 0.05),
        'max_depth':        diag_cfg.get('max_depth', 4),
        'top_k_starts':     diag_cfg.get('tracing_top_k', 3),
        'max_bfs_branches': diag_cfg.get('max_bfs_branches', 3),
    }
    # ── CoExBO config ───────────────────────────────────────────────────────
    coexbo_config = {
        'feature_names':    feature_names,
        'drift_threshold':  diag_cfg.get('drift_threshold', 0.5),
        'top_k':            diag_cfg.get('top_k', 10),
        # Soft-Copeland Monte Carlo samples (higher = more accurate, slower)
        'n_mc_quadrature':  diag_cfg.get('n_mc_quadrature', 256),
        # Synthetic pairwise preferences for initial training
        'n_init_pref':      diag_cfg.get('n_init_pref', 400),
        # Noise on the synthetic human response (0 = noiseless oracle)
        'pref_noise_std':   diag_cfg.get('pref_noise_std', 0.05),
        # Prior inflation factor γ — higher → more weight to drift evidence
        'gamma':            diag_cfg.get('gamma', 0.01),
        'n_gp_restarts':    diag_cfg.get('n_gp_restarts', 3),
    }

    # ── Instantiate diagnosers ───────────────────────────────────────────────
    diagnosers = [
        AttentionDriftDiagnoser(drift_config),
        SpectralAttentionDriftDiagnoser(spectral_config),
        PathTracingDiagnoser(path_config),
        CoExBODiagnoser(coexbo_config),
    ]

    print("🧠 Fitting diagnosers on normal data...\n")
    for diagnoser in diagnosers:
        diagnoser.fit(normal_maps)

    print("\n🔍 Diagnosing test (faulty) sample...\n")

    results = {}

    # 1. AttentionDriftDiagnoser
    res_drift = diagnosers[0].diagnose(test_map)
    results['AttentionDriftDiagnoser'] = res_drift
    _print_report(diagnosers[0].__class__.__name__, res_drift, feature_names)

    # 2. SpectralAttentionDriftDiagnoser
    res_spectral = diagnosers[1].diagnose(test_map)
    results['SpectralAttentionDriftDiagnoser'] = res_spectral
    _print_report(diagnosers[1].__class__.__name__, res_spectral, feature_names)

    # 3. PathTracingDiagnoser (uses Spectral root candidates)
    root_candidate_indices = []
    if res_spectral.is_anomaly:
        top_k_roots = diag_cfg.get('tracing_top_k', 3)
        for item in res_spectral.evidence[:top_k_roots]:
            sensor_name = item['sensor']
            if sensor_name in feature_names:
                root_candidate_indices.append(feature_names.index(sensor_name))
        print(f"🔗 [PathTracer] Intercepted Top-{top_k_roots} Root Causes from Spectral: "
              f"{[feature_names[i] for i in root_candidate_indices]}")

    res_path = diagnosers[2].diagnose(test_map, root_candidates=root_candidate_indices)
    results['PathTracingDiagnoser'] = res_path
    _print_report(diagnosers[2].__class__.__name__, res_path, feature_names)

    # Save path visualization
    path_diagnoser = diagnosers[2]
    output_dir = os.path.join(faulty_dir, "figures", "paths")
    os.makedirs(output_dir, exist_ok=True)
    path_diagnoser.visualize_paths(test_map, res_path, save_dir=output_dir)

    # 4. CoExBO Preference Learning Diagnoser
    print("🤖 Running CoExBO preference-learning diagnosis...")
    print("   (Building pairwise preference model from drift evidence...)")
    res_coexbo = diagnosers[3].diagnose(test_map)
    results['CoExBODiagnoser'] = res_coexbo
    _print_report('CoExBODiagnoser', res_coexbo, feature_names)

    # ── Consensus report ─────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("🏆 Consensus Root Cause Report")
    print("═" * 70)

    # Collect top-1 candidates from each method that detected an anomaly
    candidates: dict[str, int] = {}
    if res_drift.is_anomaly and res_drift.evidence:
        src = res_drift.evidence[0].get("source", "")
        if src in feature_names:
            candidates["AttentionDrift"] = feature_names.index(src)
    if res_spectral.is_anomaly and res_spectral.evidence:
        sn = res_spectral.evidence[0].get("sensor", "")
        if sn in feature_names:
            candidates["SpectralDrift"] = feature_names.index(sn)
    if res_path.is_anomaly and res_path.evidence:
        rc = res_path.evidence[0].get("root_cause_candidate", "")
        if rc in feature_names:
            candidates["PathTracing"] = feature_names.index(rc)
    if res_coexbo.is_anomaly and res_coexbo.evidence:
        candidates["CoExBO"] = res_coexbo.evidence[0]["sensor_idx"]

    if candidates:
        from collections import Counter
        vote_counts = Counter(candidates.values())
        consensus_idx = vote_counts.most_common(1)[0][0]
        consensus_name = feature_names[consensus_idx]
        print(f"   🎯 Consensus Root Cause Candidate: [{consensus_name}]")
        print(f"      Voted by: {[m for m, idx in candidates.items() if idx == consensus_idx]}")
        print()
        for method, idx in candidates.items():
            print(f"   {method:<28} → {feature_names[idx]}")
    else:
        print("   ✅ No anomalies detected by any diagnoser.")

    # ── Z-Score Evaluation ───────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("🏆 Diagnoser Performance Evaluation (Z-Score)")
    print("═" * 70)

    for diagnoser in diagnosers:
        name = diagnoser.__class__.__name__
        if name in ('PathTracingDiagnoser', 'CoExBODiagnoser'):
            continue

        healthy_scores = []
        for h_map in normal_maps:
            if name == 'AttentionDriftDiagnoser':
                diff_matrix = np.abs(h_map - diagnoser.baseline_map)
                score = np.linalg.norm(diff_matrix)
            elif name == 'SpectralAttentionDriftDiagnoser':
                curr_rank, curr_gap = diagnoser._get_spectral_features(h_map)
                gap_drift = np.abs(curr_gap - diagnoser.baseline_spectral_gap)
                rank_drift = np.sum(np.abs(curr_rank - diagnoser.baseline_token_rank))
                score = float(gap_drift + rank_drift)
            healthy_scores.append(score)

        mu_h = np.mean(healthy_scores)
        sigma_h = np.std(healthy_scores)

        if name == 'AttentionDriftDiagnoser':
            faulty_score = np.linalg.norm(np.abs(test_map - diagnoser.baseline_map))
        elif name == 'SpectralAttentionDriftDiagnoser':
            curr_rank, curr_gap = diagnoser._get_spectral_features(test_map)
            gap_drift = np.abs(curr_gap - diagnoser.baseline_spectral_gap)
            rank_drift = np.sum(np.abs(curr_rank - diagnoser.baseline_token_rank))
            faulty_score = float(gap_drift + rank_drift)

        z_score = (faulty_score - mu_h) / (sigma_h + 1e-9)
        print(f"🎯 {name}:")
        print(f"   • Healthy Mean (μ) : {mu_h:.4f}")
        print(f"   • Healthy Std (σ)  : {sigma_h:.4f}")
        print(f"   • Faulty Score (X) : {faulty_score:.4f}")
        print(f"   ⭐ Z-Score         : {z_score:.2f}\n")

    print("=" * 70)
    print("✅ Diagnosis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()