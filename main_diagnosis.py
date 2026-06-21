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

Z-score evaluation (UPDATED)
----------------------------
Now uses per-run fault scores (not averaged) compared against the healthy
distribution. Reports per-run z-scores plus the fraction of fault runs that
exceed |z| > 2 for a proper distribution-vs-distribution test.

Usage::

    python main_diagnosis.py
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis.result_loader import load_sensor_names, load_attention_weights
from src.diagnosis.attention_drift import AttentionDriftDiagnoser, SpectralAttentionDriftDiagnoser
from src.diagnosis.path_tracing import PathTracingDiagnoser
from src.diagnosis.coexbo_diagnoser import CoExBODiagnoser


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_report(name: str, result, feature_names=None):
    """Print detailed evidence for a single diagnoser."""
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
            print(f"   📊 Spectral Gap Drift: {abs(current - baseline):.4f} "
                  f"(Baseline: {baseline:.4f} → Current: {current:.4f})")
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


def _score_attention_drift(diagnoser, amap):
    """Frobenius drift score for one attention map."""
    return float(np.linalg.norm(np.abs(amap - diagnoser.baseline_map)))


def _score_spectral_drift(diagnoser, amap):
    """Combined spectral-gap + TokenRank drift score for one attention map."""
    curr_rank, curr_gap = diagnoser._get_spectral_features(amap)
    gap_drift  = np.abs(curr_gap - diagnoser.baseline_spectral_gap)
    rank_drift = np.sum(np.abs(curr_rank - diagnoser.baseline_token_rank))
    return float(gap_drift + rank_drift)


def _evaluate_zscore_distribution(name, score_fn, normal_maps, faulty_maps):
    """
    Per-run distribution-vs-distribution z-score evaluation.

    Returns a dict with healthy/fault score distributions, z-scores,
    and the detection fraction (|z| > 2).
    """
    healthy_scores = np.array([score_fn(m) for m in normal_maps])
    fault_scores   = np.array([score_fn(m) for m in faulty_maps])

    mu_h    = float(healthy_scores.mean())
    sigma_h = float(healthy_scores.std(ddof=1)) + 1e-9  # avoid div-by-zero
    z_scores = (fault_scores - mu_h) / sigma_h
    n_detected_2 = int(np.sum(np.abs(z_scores) > 2))
    n_detected_3 = int(np.sum(np.abs(z_scores) > 3))

    return {
        "name":              name,
        "healthy_scores":    healthy_scores,
        "fault_scores":      fault_scores,
        "healthy_mean":      mu_h,
        "healthy_std":       sigma_h,
        "z_scores":          z_scores,
        "z_mean":            float(z_scores.mean()),
        "z_std":             float(z_scores.std(ddof=1)) if len(z_scores) > 1 else 0.0,
        "n_detected_2sigma": n_detected_2,
        "n_detected_3sigma": n_detected_3,
        "n_fault_runs":      len(fault_scores),
    }


def _print_zscore_report(metrics):
    """Print a per-run z-score table for one diagnoser."""
    print(f"🎯 {metrics['name']}:")
    print(f"   Healthy distribution : mean={metrics['healthy_mean']:.4f}, "
          f"std={metrics['healthy_std']:.4f} (n={len(metrics['healthy_scores'])})")
    print(f"   Fault distribution   : mean={metrics['fault_scores'].mean():.4f}, "
          f"std={metrics['fault_scores'].std(ddof=1):.4f} (n={metrics['n_fault_runs']})")
    print(f"   Per-run z-scores     : {[f'{z:+.2f}' for z in metrics['z_scores']]}")
    print(f"   Mean fault z-score   : {metrics['z_mean']:+.2f}")
    n = metrics['n_fault_runs']
    print(f"   Detection @ |z| > 2  : {metrics['n_detected_2sigma']}/{n} "
          f"({100*metrics['n_detected_2sigma']/n:.0f}%)")
    print(f"   Detection @ |z| > 3  : {metrics['n_detected_3sigma']}/{n} "
          f"({100*metrics['n_detected_3sigma']/n:.0f}%)")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("🚀 Enhanced Fault Diagnosis Pipeline (with CoExBO Preference Learning)\n")

    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    diag_cfg    = config.get('diagnosis', {})
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
            raise FileNotFoundError(
                f"Neither {config['paths']['processed_csv']} nor {data_path_fallback} exists."
            )

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

    print(f"   Healthy maps: {normal_maps.shape}")
    print(f"   Fault maps  : {faulty_maps.shape}")

    # The averaged fault map is still used for the descriptive single-shot
    # diagnose() calls (which expect ONE matrix). The z-score evaluation below
    # uses per-run scores from faulty_maps directly — no averaging.
    test_map = np.mean(faulty_maps, axis=0)
    print(f"📊 Averaged test map shape: {test_map.shape}\n")

    # ── Shared config dicts ────────────────────────────────────────────────
    drift_config = {
        'feature_names':   feature_names,
        'drift_threshold': diag_cfg.get('drift_threshold', 0.5),
        'top_k':           diag_cfg.get('top_k', 10),
    }
    spectral_config = {
        'feature_names':            feature_names,
        'spectral_drift_threshold': diag_cfg.get('spectral_drift_threshold', 0.15),
        'top_k':                    diag_cfg.get('top_k', 10),
    }
    path_config = {
        'feature_names':    feature_names,
        'path_threshold':   diag_cfg.get('path_threshold', 0.05),
        'max_depth':        diag_cfg.get('max_depth', 4),
        'top_k_starts':     diag_cfg.get('tracing_top_k', 3),
        'max_bfs_branches': diag_cfg.get('max_bfs_branches', 3),
    }
    coexbo_config = {
        'feature_names':   feature_names,
        'drift_threshold': diag_cfg.get('drift_threshold', 0.5),
        'top_k':           diag_cfg.get('top_k', 10),
        'n_mc_quadrature': diag_cfg.get('n_mc_quadrature', 256),
        'n_init_pref':     diag_cfg.get('n_init_pref', 100),  # reduced from 400
        'pref_noise_std':  diag_cfg.get('pref_noise_std', 0.05),
        'gamma':           diag_cfg.get('gamma', 0.01),
        'n_gp_restarts':   diag_cfg.get('n_gp_restarts', 3),
    }

    # ── Instantiate diagnosers ──────────────────────────────────────────────
    diagnosers = [
        AttentionDriftDiagnoser(drift_config),
        SpectralAttentionDriftDiagnoser(spectral_config),
        PathTracingDiagnoser(path_config),
        CoExBODiagnoser(coexbo_config),
    ]

    print("🧠 Fitting diagnosers on normal data...\n")
    for diagnoser in diagnosers:
        diagnoser.fit(normal_maps)

    print("\n🔍 Diagnosing test (averaged faulty) sample...\n")

    # 1. AttentionDriftDiagnoser
    res_drift = diagnosers[0].diagnose(test_map)
    _print_report(diagnosers[0].__class__.__name__, res_drift, feature_names)

    # 2. SpectralAttentionDriftDiagnoser
    res_spectral = diagnosers[1].diagnose(test_map)
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
    _print_report(diagnosers[2].__class__.__name__, res_path, feature_names)

    # Save path visualization
    output_dir = os.path.join(faulty_dir, "figures", "paths")
    os.makedirs(output_dir, exist_ok=True)
    diagnosers[2].visualize_paths(test_map, res_path, save_dir=output_dir)

    # 4. CoExBO Preference Learning Diagnoser
    print("🤖 Running CoExBO preference-learning diagnosis...")
    print("   (Building pairwise preference model from drift evidence...)")
    res_coexbo = diagnosers[3].diagnose(test_map)
    _print_report('CoExBODiagnoser', res_coexbo, feature_names)

    # ── Consensus report ────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("🏆 Consensus Root Cause Report")
    print("═" * 70)

    candidates: dict = {}
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

    # ── Per-run Z-Score Distribution Test ───────────────────────────────────
    # Each fault run is scored individually against the healthy distribution.
    # Reports: per-run z-scores, mean z-score, and detection rates at 2σ and 3σ.
    print("\n" + "═" * 70)
    print("🏆 Diagnoser Performance Evaluation (Per-Run Z-Score Distribution Test)")
    print("═" * 70)
    print(f"   Healthy runs: {len(normal_maps)}  |  Fault runs: {len(faulty_maps)}\n")

    drift_metrics = _evaluate_zscore_distribution(
        name="AttentionDriftDiagnoser",
        score_fn=lambda m: _score_attention_drift(diagnosers[0], m),
        normal_maps=normal_maps,
        faulty_maps=faulty_maps,
    )
    _print_zscore_report(drift_metrics)

    spectral_metrics = _evaluate_zscore_distribution(
        name="SpectralAttentionDriftDiagnoser",
        score_fn=lambda m: _score_spectral_drift(diagnosers[1], m),
        normal_maps=normal_maps,
        faulty_maps=faulty_maps,
    )
    _print_zscore_report(spectral_metrics)

    # ── Decomposed spectral-gap-only test (the original "spectral gap test") ──
    # Spectral gap alone, without TokenRank — to isolate the pure gap signal.
    print("─" * 70)
    print("🔬 Spectral-Gap-Only Test (decomposed from SpectralDrift)")
    print("─" * 70)

    def _score_gap_only(diagnoser, amap):
        _, curr_gap = diagnoser._get_spectral_features(amap)
        return float(np.abs(curr_gap - diagnoser.baseline_spectral_gap))

    gap_metrics = _evaluate_zscore_distribution(
        name="SpectralGapOnly",
        score_fn=lambda m: _score_gap_only(diagnosers[1], m),
        normal_maps=normal_maps,
        faulty_maps=faulty_maps,
    )
    _print_zscore_report(gap_metrics)

    print("=" * 70)
    print("✅ Diagnosis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()