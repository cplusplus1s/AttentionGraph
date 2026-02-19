"""
Utilities for loading experiment results (attention weights, sensor names,
prediction arrays) produced by the iTransformer training pipeline.

Module-level functions
----------------------
- ``find_latest_result_folder`` : Locate the most recent result directory.
- ``load_sensor_names``         : Read feature names from a processed CSV header.
- ``load_attention_weights``    : Load & clean the saved attention weight array.

``ResultLoader`` class
---------------------
A convenience wrapper that combines the three functions above into a single
``load_data()`` call, and caches ``latest_folder`` for downstream use
(e.g. determining where to write output figures).
"""

import os
import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Module-level utilities (reusable by other modules without instantiation)
# ---------------------------------------------------------------------------

def find_latest_result_folder(results_root: str) -> str:
    """
    Return the path of the most recently modified result folder under
    *results_root*.

    Searches for folders matching known naming patterns first, then falls
    back to any subdirectory if none are found.

    :raises FileNotFoundError: If no result directories exist.
    """
    patterns = ["MSD_*", "sensor_analysis_short*", "sensor_analysis_merged*", "Exp*"]
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(
            p for p in glob.glob(os.path.join(results_root, pattern))
            if os.path.isdir(p)
        )

    if not candidates:
        # Fallback: any directory inside results_root
        candidates = [
            p for p in glob.glob(os.path.join(results_root, "*"))
            if os.path.isdir(p)
        ]

    if not candidates:
        raise FileNotFoundError(
            f"No result folders found in: {results_root}"
        )

    return max(candidates, key=os.path.getmtime)


def load_sensor_names(data_path: str) -> List[str]:
    """
    Read the column headers of a processed CSV and return the feature names,
    excluding the metadata columns ``'date'`` and ``'OT'``.

    :param data_path: Path to the processed CSV file.
    :returns: List of sensor/feature column names.
    """
    df = pd.read_csv(data_path, nrows=1)
    return [c for c in df.columns if c not in ('date', 'OT')]


def load_attention_weights(folder_path: str, n_sensors: int) -> np.ndarray:
    """
    Load and clean the ``attention_weights.npy`` file from a result folder.

    The raw file saved by the modified iTransformer has object-array dtype
    (list of per-batch tensors).  This function normalises it to a regular
    float array and averages over layers and attention heads, returning a
    per-sample attention matrix.

    Raw shape  : ``[Layers, Samples, Heads, N_query, N_key]`` (stored as
                 object array).
    Return shape: ``[Samples, n_sensors, n_sensors]``

    :param folder_path: Directory containing ``attention_weights.npy``.
    :param n_sensors: Number of sensors to extract (truncates larger models).
    :raises FileNotFoundError: If the .npy file is not present.
    :raises ValueError: If the model's feature dimension is smaller than
                        *n_sensors*.
    """
    npy_path = os.path.join(folder_path, 'attention_weights.npy')
    if not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"attention_weights.npy not found in: {folder_path}\n"
            f"Did you run training with '--output_attention'?"
        )

    raw_attn = np.load(npy_path, allow_pickle=True)

    # Object arrays are lists of per-batch tensors; concatenate them first.
    if raw_attn.dtype == object:
        raw_attn = np.concatenate(list(raw_attn), axis=0)

    # Average over layers (axis 0) and heads (axis 2) → [Samples, N, N]
    avg_attn = np.mean(raw_attn, axis=(0, 2))

    model_features = avg_attn.shape[-1]
    if model_features < n_sensors:
        raise ValueError(
            f"Dimension mismatch: model attention has {model_features} features "
            f"but the processed CSV has {n_sensors} sensors."
        )

    return avg_attn[..., :n_sensors, :n_sensors].astype(float)


# ---------------------------------------------------------------------------
# Convenience class for the visualisation pipeline
# ---------------------------------------------------------------------------

class ResultLoader:
    """
    Loads the most recent experiment's attention matrix and sensor names
    in a single call, caching ``latest_folder`` for downstream use.

    Example::

        loader = ResultLoader(
            results_root=config['paths']['results_dir'],
            data_path=config['paths']['processed_csv'],
        )
        matrix, sensor_names = loader.load_data()
        output_dir = os.path.join(loader.latest_folder, "figures")
    """

    def __init__(self, results_root: str, data_path: str) -> None:
        self.results_root = results_root
        self.data_path = data_path
        self.latest_folder: Optional[str] = None

    def load_data(self) -> Tuple[np.ndarray, List[str]]:
        """
        Locate the latest result folder, load sensor names and the
        aggregated attention matrix.

        :returns: ``(matrix, sensor_names)`` where *matrix* is a
                  ``[N, N]`` float array (mean over all samples).
        """
        self.latest_folder = find_latest_result_folder(self.results_root)
        print(f"📂 Analyzing result folder: {self.latest_folder}")

        sensor_names = load_sensor_names(self.data_path)
        n = len(sensor_names)

        # [Samples, N, N] — then collapse to a single [N, N] heatmap
        all_samples = load_attention_weights(self.latest_folder, n)
        matrix = np.mean(all_samples, axis=0)

        return matrix, sensor_names
