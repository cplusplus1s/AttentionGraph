from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

@dataclass
class DiagnosisResult:
    is_anomaly: bool
    severity: float  # 0.0 - 1.0
    diagnosis_type: str  # e.g., "AttentionDrift", "RootCauseAnalysis"
    description: str = ""

    # Top-K signals which result in anomaly
    # format: [{"source": "Gas_Flow", "target": "Pressure", "score": 0.8}, ...]
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Extended fields, reserved for storing special data for future algorithms.
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"[{self.diagnosis_type}] Anomaly: {self.is_anomaly} (Severity: {self.severity:.2f}) - {self.description}"


class BaseDiagnoser(ABC):
    """Diagnoser abstract base class"""

    def __init__(self, config: Dict[str, Any]):
        """
        :param config: specific configuration dictionary of this dignoser
        """
        self.config = config
        self.feature_names = config.get('feature_names', [])

    @abstractmethod
    def fit(self, normal_attention_maps: np.ndarray):
        """
        (Optional) Load or learn a baseline for the "normal state".
        For attention drift detection, calculate the mean of the normal Attention here.
        """
        pass

    @abstractmethod
    def diagnose(self,
                 current_attention_map: np.ndarray,
                 prediction_error: Optional[np.ndarray] = None) -> DiagnosisResult:
        """
        The core method for performing diagnosis.

        :param current_attention_map: the Attention matrix for the current time window
        :param prediction_error: (optional) the current prediction error (MSE/MAE)
        """
        pass