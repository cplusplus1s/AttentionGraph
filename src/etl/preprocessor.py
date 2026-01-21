import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, config):
        """
        :param config: The full processing config dict.
        """
        self.resample_rate = config.get('resample_rate', '200ms')
        self.start_date = pd.Timestamp(config.get('start_date', "2024-01-01"))
        self.selection_config = config.get('selection', {}) # signal channels filter

    def process(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        print("🔄 Starting preprocessing pipeline (Keeping original column names)...")

        # 1. Time align and resample
        df_aligned = self._align_and_resample(df_raw)

        # 2. Convert milliseconds to timestamp
        df_aligned.index = self.start_date + df_aligned.index

        # 3. Columns filter
        df_selected = self._select_columns(df_aligned)

        # 4. Remove constant columns which result in gradient explosion during training
        df_clean = self._drop_constant_columns(df_selected)

        # 5. Add date and OT
        df_final = self._finalize_format(df_clean)

        print(f"✅ Preprocessing done. Final shape: {df_final.shape}")
        return df_final

    def _select_columns(self, df):
        """Filter columns based on configuration (Exact match)."""
        if not self.selection_config.get('enabled', False):
            return df

        targets = self.selection_config.get('selected_sensors', [])
        if not targets:
            print("⚠️ Selection enabled but no sensors specified. Keeping all.")
            return df

        available_cols = [col for col in targets if col in df.columns]
        missing_cols = [col for col in targets if col not in df.columns]

        if missing_cols:
            print(f"⚠️ Warning: The following sensors were not found in raw data: {missing_cols}")

        if not available_cols:
            print(f"ℹ️ Available columns in data: {df.columns[:10].tolist()}...")
            raise ValueError("❌ No selected sensors found! Please check settings.yaml match exact raw names.")

        print(f"✂️ Selected {len(available_cols)} features from {len(df.columns)} original features.")
        return df[available_cols].copy()

    def _align_and_resample(self, df_raw):
        """
        Timestapm align and resample
        """
        sensor_series_list = []
        n_sensors = len(df_raw.columns) // 2

        for i in range(n_sensors):
            time_col_idx = 2 * i
            value_col_idx = 2 * i + 1

            # Extrac time and vlue column
            t_col = df_raw.iloc[:, time_col_idx]
            v_col = df_raw.iloc[:, value_col_idx]

            # Get original column name
            raw_col_name = df_raw.columns[value_col_idx]
            clean_name = raw_col_name.strip()

            # Convert into numeric type
            t_col = pd.to_numeric(t_col, errors='coerce')
            v_col = pd.to_numeric(v_col, errors='coerce')

            # Remove invalid values
            valid_idx = t_col.notna() & v_col.notna()

            if not valid_idx.any():
                continue

            s = pd.Series(
                v_col[valid_idx].values,
                index=pd.to_timedelta(t_col[valid_idx], unit='ms')
            )
            s.name = clean_name

            # Deduplication
            s = s.groupby(s.index).mean()
            sensor_series_list.append(s)

        if not sensor_series_list:
            raise ValueError("Error: No valid sensor data found to align.")

        df_merged = pd.concat(sensor_series_list, axis=1)

        df_resampled = df_merged.resample(self.resample_rate).mean()

        df_resampled = df_resampled.ffill().bfill()

        return df_resampled

    def _drop_constant_columns(self, df):
        # Drop constant value columns
        std_vals = df.std()
        cols_to_drop = std_vals[std_vals == 0].index.tolist()
        if cols_to_drop:
            print(f"⚠️ Dropping {len(cols_to_drop)} constant columns.")
            return df.drop(columns=cols_to_drop)
        return df.copy()

    def _finalize_format(self, df):
        # 1. Get target column
        target_name = self.selection_config.get('target_col')

        # 2. Specify OT column
        if target_name and target_name in df.columns:
            print(f"🎯 Setting Target (OT) column from: {target_name}")
            df['OT'] = df[target_name]
        else:
            if target_name:
                print(f"⚠️ Target column '{target_name}' not found in processed data.")
            else:
                print("ℹ️ No target_col specified. Using last column as OT.")
            last_col = df.columns[-1]
            df['OT'] = df[last_col]

        # 3. Put date into the first column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        return df