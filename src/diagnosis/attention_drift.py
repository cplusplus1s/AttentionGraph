import numpy as np
from typing import Dict, Any, Optional
from scipy.linalg import eig
from .base import BaseDiagnoser, DiagnosisResult

class AttentionDriftDiagnoser(BaseDiagnoser):
    """
    Detect the drift of the Attention Map relative to the baseline.
    Use traditional Frobenius norm to calculate the drift.
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


class SpectralAttentionDriftDiagnoser(BaseDiagnoser):
    """
    [Advanced solution]
    Detect anomaly using Eigenvalue (Spectral Gap) and Eigenvector (TokenRank) drift.
    Treat the Attention Map as a transition probability graph of a Markov chain and
    calculate the collapse of its spectral topology and the drift of node importance.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.baseline_token_rank: Optional[np.ndarray] = None
        self.baseline_spectral_gap: Optional[float] = None
        # Spectral features typically range from 0 to 2, the threshold should be lower than Frobenius norm.
        self.threshold = config.get('spectral_drift_threshold', 0.15)

    def _get_spectral_features(self, attention_map: np.ndarray):
        """
        Calculate TokenRank and Spectral Gap of attention amp
        """
        # Row sums should be 1
        row_sums = attention_map.sum(axis=1, keepdims=True)
        P = attention_map / (row_sums + 1e-9)

        eigenvalues, eigenvectors = eig(P.T)

        # Sort by eigenvalue magnitude from largest to smallest.
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Extract TokenRank, the eigenvector corresponding to the largest eigenvalue
        token_rank = np.abs(np.real(eigenvectors[:, 0]))
        token_rank = token_rank / np.sum(token_rank) # make sure sum is 1

        # Extract Spectral Gap, the difference between the 1st and 2nd largest eigenvalues
        if len(eigenvalues) > 1:
            spectral_gap = np.abs(eigenvalues[0]) - np.abs(eigenvalues[1])
        else:
            spectral_gap = 1.0 # Theoretical lower bound

        return token_rank, spectral_gap

    def fit(self, normal_attention_maps: np.ndarray):
        """
        Fit the TokenRank and Spectral Gap of the baseline using Attention Maps from normal data.
        """
        print(f"🧠 [SpectralDiagnoser] Learning topological baseline from {len(normal_attention_maps)} samples...")
        ranks = []
        gaps = []
        for amap in normal_attention_maps:
            rank, gap = self._get_spectral_features(amap)
            ranks.append(rank)
            gaps.append(gap)

        self.baseline_token_rank = np.mean(ranks, axis=0)
        self.baseline_spectral_gap = float(np.mean(gaps))

    def diagnose(self, current_attention_map: np.ndarray, prediction_error=None) -> DiagnosisResult:
        if self.baseline_token_rank is None:
            raise ValueError("Diagnoser not fitted! Call fit() with normal data first.")

        curr_rank, curr_gap = self._get_spectral_features(current_attention_map)

        # Spectral Gap Drift
        # A sharp shrankingof the gap often indicates that some sensors have become isolated.
        gap_drift = np.abs(curr_gap - self.baseline_spectral_gap)

        # Node global importance drift score (TokenRank Manhattan distance)
        rank_drift_vector = np.abs(curr_rank - self.baseline_token_rank)
        rank_drift_score = np.sum(rank_drift_vector)

        # Overall score
        drift_score = float(gap_drift + rank_drift_score)

        is_anomaly = drift_score > self.threshold
        severity = min(drift_score / (self.threshold * 2), 1.0)

        # Root Cause Analysis：
        # Top-k TokenRank drift
        k = self.config.get('top_k', 5)
        k = min(k, len(curr_rank))
        top_drift_indices = np.argsort(rank_drift_vector)[::-1][:k]

        evidence = []
        for idx in top_drift_indices:
            sensor_name = self.feature_names[idx] if self.feature_names else f"Sensor_{idx}"
            change_val = float(curr_rank[idx] - self.baseline_token_rank[idx])
            evidence.append({
                "sensor": sensor_name,
                "importance_change_magnitude": abs(change_val),
                "type": "rank_increased" if change_val > 0 else "rank_decreased",
                "reason": "TokenRank Topological Shift"
            })

        return DiagnosisResult(
            is_anomaly=is_anomaly,
            severity=severity,
            diagnosis_type="SpectralAttentionDrift",
            description=(f"Topology drifted by score {drift_score:.4f} "
                         f"(Gap drift: {gap_drift:.4f}, Rank drift: {rank_drift_score:.4f})"),
            evidence=evidence,
            details={
                "baseline_gap": self.baseline_spectral_gap,
                "current_gap": curr_gap
            }
        )