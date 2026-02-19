import pandas as pd

from .base import BasePreprocessor


class WDLPreprocessor(BasePreprocessor):
    """
    Preprocessor for WDL Replay data.

    Pipeline:
        1. Align interleaved (Time_ms, Value) column pairs onto a shared
           time grid and resample to a uniform frequency.
        2. Convert the TimedeltaIndex to absolute timestamps.
        3. Filter to selected sensor columns.
        4. Drop constant-valued (zero-variance) columns.
        5. Add the OT target column and reset the index to 'date'.
    """

    def process(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        print("🔄 WDL Preprocessing pipeline...")

        df_aligned = self._align_and_resample(df_raw)
        df_aligned.index = self.start_date + df_aligned.index   # Timedelta → Timestamp

        df_selected = self._select_columns(df_aligned)
        df_clean = self._drop_constant_columns(df_selected)
        df_final = self._finalize_format(df_clean)

        print(f"✅ WDL preprocessing done. Final shape: {df_final.shape}")
        return df_final

    def _align_and_resample(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Merge interleaved (Time_ms, Value) column pairs into a single
        DataFrame on a uniform time grid.

        Each sensor in a WDL file has its own independent timestamp axis,
        so we build individual pd.Series with TimedeltaIndex and then
        outer-join them before resampling.
        """
        sensor_series = []
        n_sensors = len(df_raw.columns) // 2

        for i in range(n_sensors):
            t_col = pd.to_numeric(df_raw.iloc[:, 2 * i], errors='coerce')
            v_col = pd.to_numeric(df_raw.iloc[:, 2 * i + 1], errors='coerce')
            col_name = df_raw.columns[2 * i + 1].strip()

            valid_mask = t_col.notna() & v_col.notna()
            if not valid_mask.any():
                continue

            s = pd.Series(
                v_col[valid_mask].values,
                index=pd.to_timedelta(t_col[valid_mask], unit='ms'),
                name=col_name,
            )
            # Deduplicate timestamps (keep mean for any duplicate entries)
            s = s.groupby(s.index).mean()
            sensor_series.append(s)

        if not sensor_series:
            raise ValueError("No valid sensor data found in WDL file.")

        df_merged = pd.concat(sensor_series, axis=1)
        df_resampled = df_merged.resample(self.resample_rate).mean()
        return df_resampled.ffill().bfill()


class MatlabPreprocessor(BasePreprocessor):
    """
    Preprocessor for MATLAB simulation data.

    Expects a DataFrame with a ``TimedeltaIndex`` as produced by
    ``MatlabLoader``.  The steps mirror those of ``WDLPreprocessor`` but
    skip WDL-specific time-alignment because the MATLAB loader already
    provides aligned, regular-interval data.

    Pipeline:
        1. Resample to the configured frequency.
        2. Convert the TimedeltaIndex to absolute timestamps.
        3. Filter to selected sensor columns.
        4. Drop constant-valued (zero-variance) columns.
        5. Add the OT target column and reset the index to 'date'.
    """

    def process(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        print("🔄 MATLAB Preprocessing pipeline...")

        df_resampled = df_raw.resample(self.resample_rate).mean().ffill().bfill()
        df_resampled.index = self.start_date + df_resampled.index  # Timedelta → Timestamp

        df_selected = self._select_columns(df_resampled)
        df_clean = self._drop_constant_columns(df_selected)
        df_final = self._finalize_format(df_clean)

        print(f"✅ MATLAB preprocessing done. Final shape: {df_final.shape}")
        return df_final
