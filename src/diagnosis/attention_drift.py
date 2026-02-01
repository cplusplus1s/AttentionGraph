import numpy as np
from typing import Dict, Any, Optional
from .base import BaseDiagnoser, DiagnosisResult

class AttentionDriftDiagnoser(BaseDiagnoser):
    """
    Detect the drift of the Attention Map relative to the baseline.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.baseline_map: Optional[np.ndarray] = None
        self.threshold = config.get('drift_threshold', 0.5)

    def fit(self, normal_attention_maps: np.ndarray):
        """
        Calculate the average Attention Map of normal data as a baseline.
        """
        # Input shape: [Samples, Features, Features]
        # Output shape: [Features, Features]
        print(f"🧠 [DriftDiagnoser] Learning baseline from {len(normal_attention_maps)} samples...")
        self.baseline_map = np.mean(normal_attention_maps, axis=0)

    def diagnose(self, current_attention_map: np.ndarray, prediction_error=None) -> DiagnosisResult:
        if self.baseline_map is None:
            raise ValueError("Diagnoser not fitted! Call fit() with normal data first.")

        # Calculate the overall distance (Matrix Distance), using the Frobenius norm
        diff_matrix = np.abs(current_attention_map - self.baseline_map)
        drift_score = np.linalg.norm(diff_matrix)

        # Normalized score
        is_anomaly = drift_score > self.threshold
        severity = min(drift_score / (self.threshold * 2), 1.0) # Mapping to 0-1

        # Find the Top-K edges with the greatest differences (Root Cause Hint)
        k = self.config.get('top_k', 5)
        flat_indices = np.argsort(diff_matrix.flatten())[::-1][:k]

        evidence = []
        rows, cols = diff_matrix.shape
        for idx in flat_indices:
            r, c = divmod(idx, cols)
            # Use index if feature name does not exist
            src_name = self.feature_names[c] if self.feature_names else str(c)
            tgt_name = self.feature_names[r] if self.feature_names else str(r)

            evidence.append({
                "source": src_name,
                "target": tgt_name,
                "change_magnitude": float(diff_matrix[r, c]),
                "type": "weight_increase" if current_attention_map[r, c] > self.baseline_map[r, c] else "weight_decrease"
            })

        return DiagnosisResult(
            is_anomaly=is_anomaly,
            severity=severity,
            diagnosis_type="AttentionDrift",
            description=f"Attention structure drifted by score {drift_score:.4f} (Threshold: {self.threshold})",
            evidence=evidence
        )