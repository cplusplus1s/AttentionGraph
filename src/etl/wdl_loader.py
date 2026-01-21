import pandas as pd
import io
import os
from .base import BaseLoader

class WDLReplayLoader(BaseLoader):
    def load(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Raw file not found: {file_path}")

        print(f"🔄 Loading WDL file: {file_path}...")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 1. Locate HistoricalData
        start_idx = -1
        for i, line in enumerate(lines):
            if "HistoricalData:" in line:
                start_idx = i
                break

        if start_idx == -1:
            raise ValueError("Invalid WDL format: 'HistoricalData' section not found.")

        # 2. Build table header
        raw_names = lines[start_idx + 1].strip().split('\t')
        headers = []
        for name in raw_names:
            clean_name = name.strip()
            headers.append(f"{clean_name}_Time_msec")
            headers.append(f"{clean_name}_Value")

        # 3. Get data block
        data_content = "".join(lines[start_idx + 3:])

        df = pd.read_csv(
            io.StringIO(data_content),
            sep='\t',
            names=headers,
            index_col=False,
            usecols=range(len(headers)),
            low_memory=False,
            na_values='---'
        )
        print(f"✅ Loaded raw data shape: {df.shape}")
        return df