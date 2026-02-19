import io
import os
import pandas as pd

from .base import BaseLoader


class WDLReplayLoader(BaseLoader):
    """
    Loads a single WDL Replay export file.

    WDL files contain an interleaved table where every sensor occupies *two*
    consecutive columns: ``<SensorName>_Time_msec`` and ``<SensorName>_Value``.
    The raw DataFrame returned by this loader preserves that interleaved
    structure; time-alignment and resampling are handled by ``WDLPreprocessor``.
    """

    def load(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"WDL file not found: {file_path}")

        print(f"🔄 Loading WDL file: {file_path} ...")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Locate the HistoricalData section
        start_idx = next(
            (i for i, line in enumerate(lines) if "HistoricalData:" in line),
            -1
        )
        if start_idx == -1:
            raise ValueError(
                f"Invalid WDL format: 'HistoricalData:' section not found in {file_path}"
            )

        # Build interleaved column headers from the sensor name row
        raw_names = lines[start_idx + 1].strip().split('\t')
        headers = []
        for name in raw_names:
            clean = name.strip()
            headers.append(f"{clean}_Time_msec")
            headers.append(f"{clean}_Value")

        # Parse the data block (starts two lines after the header row)
        data_content = "".join(lines[start_idx + 3:])
        df = pd.read_csv(
            io.StringIO(data_content),
            sep='\t',
            names=headers,
            index_col=False,
            usecols=range(len(headers)),
            low_memory=False,
            na_values='---',
        )

        print(f"✅ Loaded raw WDL data: {df.shape}")
        return df
