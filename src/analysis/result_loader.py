import os
import glob
import numpy as np
import pandas as pd

class ResultLoader:
    def __init__(self, results_root, data_path):
        self.results_root = results_root
        self.data_path = data_path
        self.latest_folder = None
        self.sensor_names = []

    def load_data(self):
        """Load latest Attention Matrix and signal channels names"""
        self.latest_folder = self._find_latest_folder()
        print(f"📂 Analyzing result folder: {self.latest_folder}")

        # 1. Read csv to get column names (drop 'date' and 'OT')
        df = pd.read_csv(self.data_path, nrows=1)
        self.sensor_names = [c for c in df.columns if c not in ['date', 'OT']]
        n_sensors = len(self.sensor_names)

        # 2. Load attention weights
        npy_path = os.path.join(self.latest_folder, 'attention_weights.npy')
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"attention_weights.npy not found in {self.latest_folder}")

        raw_attn = np.load(npy_path, allow_pickle=True)
        if raw_attn.dtype == 'O':
            raw_attn = np.concatenate(list(raw_attn), axis=0)

        # Aggregation: [Layers, N, N] -> fetch last layer -> [N, N]
        avg_attn = np.mean(raw_attn, axis=(0, 2))[-1]

        # 3. Dimensional checks and truncation
        if avg_attn.shape[0] < n_sensors:
            raise ValueError(f"Dimension Mismatch: Model({avg_attn.shape[0]}) < CSV({n_sensors})")

        # Extract the corresponding attention
        clean_attn = avg_attn[:n_sensors, :n_sensors].astype(float)

        return clean_attn, self.sensor_names

    def _find_latest_folder(self):
        # Search folders start with sensor_analysis_short*
        search_pattern = os.path.join(self.results_root, "sensor_analysis_short*")
        folders = glob.glob(search_pattern)
        if not folders:
            raise FileNotFoundError("No result folders found in results directory.")
        return max(folders, key=os.path.getmtime)