"""
fingerprint_matching.py — Fault Fingerprint Matching via CoExBO Preference Learning.

Previously a stub ("TODO: Implement").  Now powered by CoExBODiagnoser:

    - A "fault fingerprint" is a pattern of attention-map drift that the system
      has learned to associate with a particular fault type.
    - The fingerprint database is built from labelled healthy/faulty attention
      maps; each entry is the mean attention map for a fault class.
    - During diagnosis, the unknown test map is compared against every
      fingerprint using the CoExBO soft-Copeland preference ranking:
        * For each fingerprint class, we treat the class drift as the
          "expert preference signal" and compute how strongly the test map
          resembles it.
    - The best-matching class (highest fused score) is returned as the
      diagnosis, together with a confidence estimate.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseDiagnoser, DiagnosisResult
from .coexbo_diagnoser import (
    CoExBODiagnoser,
    _build_synthetic_pairwise_data,
    _extract_sensor_features,
    PreferenceLearner,
    SoftCopelandScorer,
    PriorAugmentedRanker,
)


class FingerprintDiagnoser(BaseDiagnoser):
    """
    Fault fingerprint database matching powered by CoExBO preference learning.

    Workflow
    --------
    1. ``fit(normal_attention_maps)``
       Stores the healthy baseline map (same as other diagnosers).

    2. ``register_fingerprint(label, faulty_maps)``
       Add a named fault class to the database.  The class fingerprint is the
       mean attention map of the provided faulty samples.  Per-sensor drift
       features and simulated pairwise preferences are computed automatically.

    3. ``diagnose(current_attention_map)``
       Computes per-class CoExBO soft-Copeland scores and returns the best
       matching fault class together with a ranked sensor evidence list.

    Example
    -------
    ::

        diagnoser = FingerprintDiagnoser(config)
        diagnoser.fit(normal_maps)
        diagnoser.register_fingerprint("bearing_fault", bearing_faulty_maps)
        diagnoser.register_fingerprint("coupling_fault", coupling_faulty_maps)
        result = diagnoser.diagnose(test_map)
        # result.details["matched_class"] -> "bearing_fault"

    Expert-in-the-loop refinement
    ------------------------------
    After an initial diagnosis, domain experts can correct the matched class or
    provide sensor-level pairwise labels to refine the per-class preference
    models::

        diagnoser.add_expert_correction(
            class_label="bearing_fault",
            sensor_labels=[(3, 7, 3), (12, 5, 12)],
        )
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get("drift_threshold", 0.5)
        self.top_k = config.get("top_k", 5)

        # Fingerprint database: {label -> dict}
        self._fingerprints: Dict[str, Dict] = {}

        # Baseline (set by fit)
        self.baseline_map: Optional[np.ndarray] = None

        # Per-class CoExBO diagnosers
        self._class_diagnosers: Dict[str, CoExBODiagnoser] = {}

    # ------------------------------------------------------------------
    # fit (BaseDiagnoser interface)
    # ------------------------------------------------------------------

    def fit(self, normal_attention_maps: np.ndarray) -> None:
        """
        Record the healthy baseline.

        Args:
            normal_attention_maps: [S, N, N] healthy attention maps.
        """
        print(
            f"🧠 [FingerprintDiagnoser] Learning baseline from "
            f"{len(normal_attention_maps)} normal samples..."
        )
        self.baseline_map = np.mean(normal_attention_maps, axis=0)
        print("✅ [FingerprintDiagnoser] Baseline set.")

    # ------------------------------------------------------------------
    # Fingerprint registration
    # ------------------------------------------------------------------

    def register_fingerprint(
        self,
        label: str,
        faulty_maps: np.ndarray,
        n_pairs: int = 300,
        noise_std: float = 0.05,
        gamma: float = 0.01,
    ) -> "FingerprintDiagnoser":
        """
        Add a labelled fault class to the fingerprint database.

        Args:
            label:       Human-readable fault class name.
            faulty_maps: [S, N, N] attention maps from this fault type.
            n_pairs:     Synthetic pairwise samples for preference learning.
            noise_std:   Noise on the synthetic human preference signal.
            gamma:       Prior inflation factor (CoExBO Eq. 7).

        Returns:
            self (for chaining).
        """
        if self.baseline_map is None:
            raise RuntimeError("Call fit() before register_fingerprint().")

        fingerprint = np.mean(faulty_maps, axis=0)   # [N, N]
        diff_map = np.abs(fingerprint - self.baseline_map)
        drift_scores = np.linalg.norm(diff_map, axis=1)

        # Instantiate a dedicated CoExBODiagnoser for this class
        class_config = {
            "feature_names": self.feature_names,
            "drift_threshold": self.threshold,
            "top_k": self.top_k,
            "n_mc_quadrature": 128,
            "n_init_pref": n_pairs,
            "pref_noise_std": noise_std,
            "gamma": gamma,
            "n_gp_restarts": 2,
        }
        coexbo = CoExBODiagnoser(class_config)
        coexbo.fit(faulty_maps)   # trains preference model on this fault class

        self._fingerprints[label] = {
            "fingerprint": fingerprint,
            "diff_map": diff_map,
            "drift_scores": drift_scores,
        }
        self._class_diagnosers[label] = coexbo

        print(
            f"✅ [FingerprintDiagnoser] Registered fingerprint '{label}' "
            f"from {len(faulty_maps)} samples."
        )
        return self

    # ------------------------------------------------------------------
    # Expert correction
    # ------------------------------------------------------------------

    def add_expert_correction(
        self,
        class_label: str,
        sensor_labels: List[Tuple[int, int, int]],
    ) -> "FingerprintDiagnoser":
        """
        Refine the preference model for a specific fault class using expert labels.

        Args:
            class_label:   The fault class to refine.
            sensor_labels: List of (sensor_A_idx, sensor_B_idx, winner_idx).

        Returns:
            self (for chaining).
        """
        if class_label not in self._class_diagnosers:
            raise ValueError(
                f"Unknown class '{class_label}'. "
                f"Available: {list(self._class_diagnosers.keys())}"
            )
        self._class_diagnosers[class_label].add_expert_preferences(sensor_labels)
        self._class_diagnosers[class_label].update_preference_model()
        return self

    # ------------------------------------------------------------------
    # Diagnose (BaseDiagnoser interface)
    # ------------------------------------------------------------------

    def diagnose(
        self,
        current_attention_map: np.ndarray,
        prediction_error: Optional[np.ndarray] = None,
    ) -> DiagnosisResult:
        """
        Match the test attention map against the fingerprint database.

        Args:
            current_attention_map: [N, N] test attention matrix.
            prediction_error:      optional [N] per-sensor prediction errors.

        Returns:
            DiagnosisResult with the matched class and ranked sensor evidence.
        """
        if self.baseline_map is None:
            raise RuntimeError("Call fit() before diagnose().")

        # Overall anomaly check
        diff_map = np.abs(current_attention_map - self.baseline_map)
        overall_drift = float(np.linalg.norm(diff_map))
        is_anomaly = overall_drift > self.threshold
        severity = min(overall_drift / (self.threshold * 2 + 1e-9), 1.0)

        if not self._fingerprints:
            # No fingerprints registered — fall back to plain drift report
            return DiagnosisResult(
                is_anomaly=is_anomaly,
                severity=severity,
                diagnosis_type="FingerprintMatching",
                description=(
                    "No fault fingerprints registered. "
                    "Call register_fingerprint() to add fault classes."
                ),
            )

        # --- Compute per-class match score ---
        class_scores: Dict[str, float] = {}
        class_evidence: Dict[str, List] = {}

        for label, fp in self._fingerprints.items():
            # Similarity = negative Frobenius distance between current diff and class diff
            class_diff = np.abs(current_attention_map - fp["fingerprint"])
            frobenius_sim = -np.linalg.norm(class_diff - fp["diff_map"])

            # Run CoExBO diagnosis within this fault class context
            class_result = self._class_diagnosers[label].diagnose(
                current_attention_map, prediction_error
            )
            # CoExBO fused score of the top-1 sensor as proxy for class match
            top_sensor_fused = (
                class_result.details["fused_scores"][
                    np.argmax(class_result.details["fused_scores"])
                ]
            )
            # Combined class score: geometric mean of similarity and CoExBO ranking
            class_scores[label] = 0.5 * frobenius_sim + 0.5 * top_sensor_fused
            class_evidence[label] = class_result.evidence

        # Best matching class
        best_class = max(class_scores, key=class_scores.__getitem__)
        best_evidence = class_evidence[best_class]

        description = (
            f"Fingerprint matching result: '{best_class}' "
            f"(score: {class_scores[best_class]:.4f}, "
            f"overall drift: {overall_drift:.4f}). "
            f"Ranked by CoExBO soft-Copeland preference scores."
        )

        return DiagnosisResult(
            is_anomaly=is_anomaly,
            severity=severity,
            diagnosis_type="FingerprintMatching",
            description=description,
            evidence=best_evidence[: self.top_k],
            details={
                "matched_class": best_class,
                "class_scores": class_scores,
                "all_class_evidence": {
                    lbl: evs[: self.top_k] for lbl, evs in class_evidence.items()
                },
                "overall_drift": overall_drift,
            },
        )
