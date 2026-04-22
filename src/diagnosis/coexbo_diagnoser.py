"""
coexbo_diagnoser.py — CoExBO-based Fault Diagnosis via Preference Learning.

Ports the key mathematical methods from CoExBO (Collaborative Exploration
Bayesian Optimization) into AttentionGraph's diagnosis pipeline:

    1. PreferenceLearner      — GP-based pairwise preference model (Dirichlet GP
                                ported to sklearn GPC for zero extra dependencies)
    2. SoftCopelandScorer     — Monte Carlo Quadrature that converts pairwise
                                preference probabilities into global rankings
    3. PriorAugmentedRanker   — Combines attention drift (likelihood GP) with
                                the expert preference prior in a Bayesian update
                                (mirrors CoExBO_UCB._acquisition_function.py)
    4. CoExBODiagnoser        — Full BaseDiagnoser implementation that wires
                                everything together

Reference: "CoExBO: Collaborative Exploration in Bayesian Optimization with
           Pairwise Preference Feedback" (CoExBO_repomix-output.xml)

Design notes
------------
- No new heavy dependencies: uses torch, sklearn, scipy (already in requirements.txt).
- Automatic "synthetic human" mode: when no expert labels exist, preferences are
  simulated from the drift score (higher drift = preferred root cause), mirroring
  CoExBO's CoExBOwithSimulation class.
- Interactive mode: experts can supply explicit pairwise labels (list of
  (sensor_A_idx, sensor_B_idx, winner_idx) triples) via ``fit()``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit          # sigmoid, for preference probability
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from .base import BaseDiagnoser, DiagnosisResult


# =============================================================================
# 1. Feature Extraction  (attention matrix → per-sensor feature vector)
# =============================================================================

def _extract_sensor_features(
    attention_map: np.ndarray,
    diff_map: np.ndarray,
    baseline_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a feature matrix X of shape [N, d] where each row describes one
    sensor (potential root-cause candidate).

    Features per sensor i:
      - out-degree drift : mean attention drift on outgoing edges  (row i of diff)
      - in-degree drift  : mean attention drift on incoming edges  (col i of diff)
      - max outgoing     : max attention value in row i
      - max incoming     : max attention value in col i
      - self-attention   : attention_map[i, i]
      - diff_norm_row    : L2 norm of diff_map row i
      - current_row_max  : max of attention_map row i (excl. self)
      - baseline_row_max : max of baseline_map row i (excl. self) — 0 if None

    Adapted from CoExBO's convention where each candidate is described by
    a continuous feature vector before pairwise GP classification.
    """
    N = attention_map.shape[0]
    feats = []
    for i in range(N):
        row_diff = diff_map[i, :].copy(); row_diff[i] = 0
        col_diff = diff_map[:, i].copy(); col_diff[i] = 0
        row_attn = attention_map[i, :].copy(); row_attn[i] = 0
        col_attn = attention_map[:, i].copy(); col_attn[i] = 0

        f = [
            row_diff.mean(),
            col_diff.mean(),
            row_attn.max() if row_attn.max() > 0 else 0.0,
            col_attn.max() if col_attn.max() > 0 else 0.0,
            attention_map[i, i],
            np.linalg.norm(row_diff),
            row_attn.max(),
            baseline_map[i, :].max() if baseline_map is not None else 0.0,
        ]
        feats.append(f)
    return np.array(feats, dtype=np.float64)


# =============================================================================
# 2. PreferenceLearner  (mirrors CoExBO._gp_classifier)
# =============================================================================

class PreferenceLearner:
    """
    GP-based pairwise preference classifier.

    Adapted from CoExBO's DirichletGPModel / set_and_train_classifier.
    Uses sklearn's GaussianProcessClassifier (RBF kernel) as a drop-in
    that requires no extra dependencies.

    Training data is a list of (feature_A, feature_B, label) triples where
    label=1 means A is preferred (more anomalous root cause) and label=0
    means B is preferred.

    The concatenated [feature_A | feature_B] vector is the input to the GP
    classifier, identical to CoExBO's X_pairwise convention.
    """

    def __init__(self, n_restarts: int = 3):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self._gpc = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            random_state=42,
        )
        self._is_fitted = False

    def fit(self, X_pairwise: np.ndarray, y: np.ndarray) -> "PreferenceLearner":
        """
        Train the preference classifier.

        Args:
            X_pairwise: [M, 2*d] — each row is [feature_A | feature_B]
            y: [M] — binary label (1 = A preferred, 0 = B preferred)
        """
        if len(np.unique(y)) < 2:
            # sklearn GPC requires both classes; add one synthetic sample
            flip_idx = 0 if y[0] == 1 else -1
            X_pairwise = np.vstack([X_pairwise, X_pairwise[flip_idx]])
            y = np.concatenate([y, [1 - y[flip_idx]]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gpc.fit(X_pairwise, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X_pairwise: np.ndarray) -> np.ndarray:
        """
        Compute P(A preferred | feature_A, feature_B).

        Returns:
            probs: [M] float array in [0, 1]
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self._gpc.predict_proba(X_pairwise)
        # column index 1 = class "1" (A preferred)
        classes = list(self._gpc.classes_)
        idx = classes.index(1) if 1 in classes else 1
        return proba[:, idx]


# =============================================================================
# 3. SoftCopelandScorer  (mirrors CoExBO._monte_carlo_quadrature)
# =============================================================================

class SoftCopelandScorer:
    """
    Monte Carlo Quadrature for the soft-Copeland score.

    For each candidate sensor i, the soft-Copeland score is:

        C(i) = E_{j ~ Uniform(candidates)} [ P(i preferred over j) ]

    This is equivalent to the probability that sensor i would be chosen as
    the root cause when compared against a randomly drawn opponent — the same
    formulation used in CoExBO._monte_carlo_quadrature.MonteCarloQuadrature.

    The score is used to rank sensors from most to least likely to be the root
    cause, providing a global ranking from pairwise preferences (CoExBO insight).
    """

    def __init__(self, preference_learner: PreferenceLearner, n_mc: int = 256):
        self._pl = preference_learner
        self._n_mc = n_mc

    def score(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute soft-Copeland scores for all N sensors.

        Args:
            features: [N, d] per-sensor feature matrix

        Returns:
            scores_mean: [N] mean soft-Copeland score per sensor
            scores_std : [N] std of the MC estimate (uncertainty)
        """
        N, d = features.shape
        rng = np.random.default_rng(seed=42)
        mc_indices = rng.integers(0, N, size=self._n_mc)
        X_mc = features[mc_indices]   # [n_mc, d]

        scores_all = []
        for i in range(N):
            # For each MC opponent j, build pairwise vector [feat_i | feat_j]
            feat_i_rep = np.tile(features[i], (self._n_mc, 1))  # [n_mc, d]
            X_pairs = np.hstack([feat_i_rep, X_mc])             # [n_mc, 2d]
            probs = self._pl.predict_proba(X_pairs)             # [n_mc]
            scores_all.append(probs)

        scores_all = np.array(scores_all)           # [N, n_mc]
        scores_mean = scores_all.mean(axis=1)       # [N]
        scores_std = scores_all.std(axis=1)         # [N]
        return scores_mean, scores_std


# =============================================================================
# 4. PriorAugmentedRanker  (mirrors CoExBO._acquisition_function.CoExBO_UCB)
# =============================================================================

class PriorAugmentedRanker:
    """
    Combines the attention-drift likelihood with the soft-Copeland preference
    prior in a Gaussian conjugate update — the same idea as CoExBO_UCB.

    The posterior score for sensor i is:

        score_i = w_posterior * prior_mean_i + (1-w_posterior) * likelihood_i

    where
        w_posterior = σ²_likelihood / (σ²_likelihood + γ * σ²_prior)

    This mirrors Eq.(7) of CoExBO:
        posterior_gp_std² = σ²_prior_max * σ²_likelihood / (σ²_prior_max + σ²_likelihood)
    where σ²_prior_max = γ * σ²_likelihood + σ²_prior.

    gamma controls how strongly the preference prior is weighted relative to
    the data-driven likelihood (attention drift evidence).
    """

    @staticmethod
    def fuse(
        likelihood_scores: np.ndarray,
        prior_scores: np.ndarray,
        prior_std: np.ndarray,
        gamma: float = 0.01,
    ) -> np.ndarray:
        """
        Compute the prior-augmented score vector.

        Args:
            likelihood_scores: [N] drift-based scores (from AttentionDriftDiagnoser)
            prior_scores:      [N] soft-Copeland scores (from SoftCopelandScorer)
            prior_std:         [N] uncertainty on the soft-Copeland score
            gamma:             Prior inflation factor (CoExBO Eq. 7).
                               Larger gamma → more weight to drift evidence.

        Returns:
            fused: [N] posterior scores
        """
        # Normalise both to [0, 1] for comparability
        def _norm(x):
            r = x - x.min()
            return r / (r.max() + 1e-9)

        mu_l = _norm(likelihood_scores)
        mu_p = _norm(prior_scores)
        sigma_p = prior_std / (prior_std.max() + 1e-9)

        # Posterior update (Gaussian conjugate, mirroring CoExBO_UCB.posterior_gp)
        sigma_p_max = np.sqrt(gamma * (mu_l - mu_l.mean()) ** 2 + sigma_p ** 2)
        # weight of preference prior
        w = sigma_p_max ** 2 / (sigma_p_max ** 2 + (mu_l - mu_l.mean()) ** 2 + 1e-9)
        fused = w * mu_p + (1 - w) * mu_l
        return fused


# =============================================================================
# 5. PairwiseDataBuilder  (automatic "synthetic human" from CoExBOwithSimulation)
# =============================================================================

def _build_synthetic_pairwise_data(
    features: np.ndarray,
    drift_scores: np.ndarray,
    n_pairs: int = 200,
    noise_std: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate expert pairwise preferences using drift scores as ground truth.

    Mirrors CoExBO's CoExBOwithSimulation where the "human" responds based on
    the true objective function (here: per-sensor drift score).

    P(A preferred over B) = sigmoid( (drift_A - drift_B) / sigma )

    Args:
        features:    [N, d] per-sensor features
        drift_scores:[N]   per-sensor drift magnitude (from AttentionDrift)
        n_pairs:     number of pairwise comparisons to simulate
        noise_std:   Gaussian noise standard deviation on the preference signal
        rng:         optional random generator for reproducibility

    Returns:
        X_pairwise: [n_pairs, 2*d] pairwise feature matrix
        y:          [n_pairs] binary labels
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N, d = features.shape
    X_pairwise, y = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(N, size=2, replace=False)
        noisy_diff = (drift_scores[i] - drift_scores[j]) + rng.normal(0, noise_std)
        prob_i_wins = expit(noisy_diff / (drift_scores.std() + 1e-9))
        label = 1 if rng.random() < prob_i_wins else 0
        X_pairwise.append(np.concatenate([features[i], features[j]]))
        y.append(label)
    return np.array(X_pairwise), np.array(y)


# =============================================================================
# 6. CoExBODiagnoser — the full BaseDiagnoser integration
# =============================================================================

class CoExBODiagnoser(BaseDiagnoser):
    """
    Fault diagnosis via CoExBO's preference-learning pipeline.

    Workflow (mirrors CoExBO.__call__):
        1. fit()     : learn baseline from normal attention maps;
                       build a GP preference model using simulated or expert
                       pairwise comparisons.
        2. diagnose(): extract per-sensor features from the test attention map,
                       compute soft-Copeland scores, fuse with drift evidence,
                       and return a ranked DiagnosisResult.

    Expert-in-the-loop usage::

        diagnoser = CoExBODiagnoser(config)
        diagnoser.fit(normal_maps)
        # Optionally supply expert labels: list of (idx_A, idx_B, winner_idx)
        diagnoser.add_expert_preferences([(3, 7, 3), (12, 5, 12)])
        diagnoser.update_preference_model()
        result = diagnoser.diagnose(faulty_map)

    Automatic (synthetic) mode::

        diagnoser = CoExBODiagnoser(config)
        diagnoser.fit(normal_maps)          # uses simulated preferences only
        result = diagnoser.diagnose(faulty_map)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get("drift_threshold", 0.5)
        self.top_k = config.get("top_k", 5)
        self.n_mc = config.get("n_mc_quadrature", 256)
        self.n_pairs = config.get("n_init_pref", 300)
        self.noise_std = config.get("pref_noise_std", 0.05)
        self.gamma = config.get("gamma", 0.01)
        self.n_gp_restarts = config.get("n_gp_restarts", 3)

        # State (set by fit)
        self.baseline_map: Optional[np.ndarray] = None
        self._features_baseline: Optional[np.ndarray] = None
        self._drift_scores_baseline: Optional[np.ndarray] = None

        # Preference learning state
        self._X_pairwise: Optional[np.ndarray] = None
        self._y_pairwise: Optional[np.ndarray] = None
        self._preference_learner: Optional[PreferenceLearner] = None
        self._scorer: Optional[SoftCopelandScorer] = None

        # Expert label buffer (list of (idx_A, idx_B, winner_idx))
        self._expert_labels: List[Tuple[int, int, int]] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, normal_attention_maps: np.ndarray) -> None:
        """
        Learn the baseline and train an initial preference model from simulated
        pairwise comparisons over the healthy attention maps.

        Args:
            normal_attention_maps: [S, N, N] — S samples of normal attention.
        """
        print(
            f"🧠 [CoExBODiagnoser] Fitting on {len(normal_attention_maps)} normal samples..."
        )
        self.baseline_map = np.mean(normal_attention_maps, axis=0)

        # Build per-sensor drift features for the baseline (zero diff → healthy)
        zero_diff = np.zeros_like(self.baseline_map)
        self._features_baseline = _extract_sensor_features(
            self.baseline_map, zero_diff, self.baseline_map
        )
        # Per-sensor row-wise drift norm from sample variance (healthy baseline)
        sample_var = normal_attention_maps.var(axis=0)  # [N, N]
        self._drift_scores_baseline = np.linalg.norm(sample_var, axis=1)

        # Build initial pairwise dataset via synthetic human (CoExBOwithSimulation style)
        self._X_pairwise, self._y_pairwise = _build_synthetic_pairwise_data(
            self._features_baseline,
            self._drift_scores_baseline,
            n_pairs=self.n_pairs,
            noise_std=self.noise_std,
        )

        self._fit_preference_model()
        print(
            f"✅ [CoExBODiagnoser] Preference model trained on {len(self._y_pairwise)} pairs."
        )

    def _fit_preference_model(self) -> None:
        """Train (or re-train) the GP preference classifier."""
        pl = PreferenceLearner(n_restarts=self.n_gp_restarts)
        pl.fit(self._X_pairwise, self._y_pairwise)
        self._preference_learner = pl
        self._scorer = SoftCopelandScorer(pl, n_mc=self.n_mc)

    # ------------------------------------------------------------------
    # Expert label interface (human-in-the-loop)
    # ------------------------------------------------------------------

    def add_expert_preferences(
        self, labels: List[Tuple[int, int, int]]
    ) -> "CoExBODiagnoser":
        """
        Register expert pairwise preferences.

        Args:
            labels: list of (idx_A, idx_B, winner_idx) tuples.
                    winner_idx must be either idx_A or idx_B.
                    e.g. [(3, 7, 3)] means sensor 3 is more likely the root
                    cause than sensor 7.

        Returns:
            self (for chaining)
        """
        for idx_a, idx_b, winner in labels:
            if winner not in (idx_a, idx_b):
                raise ValueError(
                    f"winner_idx {winner} must be idx_A {idx_a} or idx_B {idx_b}"
                )
            self._expert_labels.append((idx_a, idx_b, winner))
        return self

    def update_preference_model(self) -> "CoExBODiagnoser":
        """
        Re-train the preference model incorporating any expert labels added via
        ``add_expert_preferences()``.

        Mirrors CoExBO's dataset update cycle (update_datasets + set_models).
        """
        if not self._expert_labels or self._features_baseline is None:
            return self

        feats = self._features_baseline
        new_X, new_y = [], []
        for idx_a, idx_b, winner in self._expert_labels:
            new_X.append(np.concatenate([feats[idx_a], feats[idx_b]]))
            new_y.append(1 if winner == idx_a else 0)

        self._X_pairwise = np.vstack(
            [self._X_pairwise, np.array(new_X)]
        )
        self._y_pairwise = np.concatenate(
            [self._y_pairwise, np.array(new_y)]
        )
        self._fit_preference_model()
        print(
            f"✅ [CoExBODiagnoser] Model updated with {len(new_X)} expert label(s). "
            f"Total pairs: {len(self._y_pairwise)}."
        )
        return self

    # ------------------------------------------------------------------
    # Diagnose
    # ------------------------------------------------------------------

    def diagnose(
        self,
        current_attention_map: np.ndarray,
        prediction_error: Optional[np.ndarray] = None,
    ) -> DiagnosisResult:
        """
        Rank sensors by their posterior probability of being the root cause.

        Pipeline (mirrors CoExBO.__call__):
            1. Compute per-sensor drift features from the test attention map.
            2. Build synthetic pairwise dataset from test drift scores.
            3. Compute soft-Copeland score (SoftCopelandScorer).
            4. Fuse with attention drift evidence (PriorAugmentedRanker).
            5. Return ranked DiagnosisResult.

        Args:
            current_attention_map: [N, N] test attention matrix.
            prediction_error: optional [N] per-sensor prediction errors.

        Returns:
            DiagnosisResult with ranked evidence list.
        """
        if self.baseline_map is None:
            raise RuntimeError("Call fit() before diagnose().")

        N = current_attention_map.shape[0]
        diff_map = np.abs(current_attention_map - self.baseline_map)

        # --- Step 1: Per-sensor drift scores (likelihood term) ---
        drift_scores = np.linalg.norm(diff_map, axis=1)  # [N]
        if prediction_error is not None and len(prediction_error) == N:
            drift_scores = 0.5 * drift_scores + 0.5 * np.array(prediction_error)

        # --- Step 2: Per-sensor feature extraction ---
        features = _extract_sensor_features(
            current_attention_map, diff_map, self.baseline_map
        )

        # --- Step 3: Soft-Copeland score (prior term) ---
        # Build a temporary preference learner on the test drift scores
        # (synthetic human updated from test observation — CoExBO loop Step 1)
        X_test_pairs, y_test = _build_synthetic_pairwise_data(
            features, drift_scores, n_pairs=self.n_pairs, noise_std=self.noise_std
        )
        # Merge with existing pairwise dataset (CoExBO: update_datasets)
        X_merged = np.vstack([self._X_pairwise, X_test_pairs])
        y_merged = np.concatenate([self._y_pairwise, y_test])
        temp_pl = PreferenceLearner(n_restarts=self.n_gp_restarts)
        temp_pl.fit(X_merged, y_merged)
        scorer = SoftCopelandScorer(temp_pl, n_mc=self.n_mc)

        copeland_mean, copeland_std = scorer.score(features)  # [N], [N]

        # --- Step 4: Prior-augmented ranking (CoExBO_UCB posterior fusion) ---
        fused_scores = PriorAugmentedRanker.fuse(
            drift_scores, copeland_mean, copeland_std, gamma=self.gamma
        )

        # --- Step 5: Overall anomaly detection ---
        overall_drift = float(np.linalg.norm(diff_map))
        is_anomaly = overall_drift > self.threshold
        severity = min(overall_drift / (self.threshold * 2 + 1e-9), 1.0)

        # --- Step 6: Build Top-K evidence list ---
        top_k = min(self.top_k, N)
        top_indices = np.argsort(fused_scores)[::-1][:top_k]

        evidence = []
        for rank, idx in enumerate(top_indices):
            name = self.feature_names[idx] if self.feature_names else f"Sensor_{idx}"
            evidence.append(
                {
                    "rank": rank + 1,
                    "sensor": name,
                    "sensor_idx": int(idx),
                    "copeland_score": float(copeland_mean[idx]),
                    "copeland_uncertainty": float(copeland_std[idx]),
                    "drift_score": float(drift_scores[idx]),
                    "fused_score": float(fused_scores[idx]),
                    "type": "root_cause_candidate",
                }
            )

        description = (
            f"CoExBO preference-learning ranked {N} sensors. "
            f"Overall drift: {overall_drift:.4f} (threshold: {self.threshold}). "
            f"Top root cause: "
            + (self.feature_names[top_indices[0]] if self.feature_names else f"Sensor_{top_indices[0]}")
        )

        return DiagnosisResult(
            is_anomaly=is_anomaly,
            severity=severity,
            diagnosis_type="CoExBOPreferenceLearning",
            description=description,
            evidence=evidence,
            details={
                "copeland_scores": copeland_mean.tolist(),
                "copeland_std": copeland_std.tolist(),
                "drift_scores": drift_scores.tolist(),
                "fused_scores": fused_scores.tolist(),
                "n_pairs_used": len(y_merged),
                "gamma": self.gamma,
            },
        )
