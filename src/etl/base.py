from abc import ABC, abstractmethod
import pandas as pd


class BaseLoader(ABC):
    """Abstract base class for all raw data loaders."""

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """
        Load raw data from the given path and return a DataFrame.

        The returned DataFrame should have a meaningful index
        (e.g. TimedeltaIndex for time-series sources) and human-readable
        column names, but it is NOT yet preprocessed or resampled.
        All preprocessing is delegated to a BasePreprocessor.
        """
        pass


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessors.

    Shared helper methods (_select_columns, _drop_constant_columns,
    _finalize_format) are implemented here so that every concrete
    preprocessor can reuse them without duplication.
    """

    def __init__(self, config: dict):
        """
        :param config: The 'processing' section of settings.yaml.
        """
        self.resample_rate: str = config.get('resample_rate', '200ms')
        self.start_date: pd.Timestamp = pd.Timestamp(
            config.get('start_date', '2024-01-01')
        )
        self.selection_config: dict = config.get('selection', {})

    @abstractmethod
    def process(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw loader output into a clean, training-ready DataFrame.

        Every concrete subclass must implement the source-specific steps
        (e.g. time-alignment, resampling) and then call the shared helpers
        provided by this base class.
        """
        pass

    # ------------------------------------------------------------------
    # Shared helper methods
    # ------------------------------------------------------------------

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter columns to only those listed in selection_config.
        If selection is disabled or no sensors are listed, returns df unchanged.
        """
        if not self.selection_config.get('enabled', False):
            return df

        targets = self.selection_config.get('selected_sensors', [])
        if not targets:
            print("⚠️  Selection enabled but 'selected_sensors' is empty. Keeping all columns.")
            return df

        available = [c for c in targets if c in df.columns]
        missing = [c for c in targets if c not in df.columns]

        if missing:
            print(f"⚠️  The following sensors were not found in the data: {missing}")

        if not available:
            print(f"ℹ️  First 10 available columns: {df.columns[:10].tolist()}")
            raise ValueError(
                "❌ None of the selected sensors were found. "
                "Check that 'selected_sensors' in settings.yaml matches the exact column names."
            )

        print(f"✂️  Kept {len(available)} of {len(df.columns)} columns.")
        return df[available].copy()

    def _drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns whose standard deviation is zero.
        Constant features cause gradient explosion during training.
        """
        zero_std_cols = df.std()[lambda s: s == 0].index.tolist()
        if zero_std_cols:
            print(f"⚠️  Dropping {len(zero_std_cols)} constant column(s): {zero_std_cols}")
            return df.drop(columns=zero_std_cols)
        return df.copy()

    def _finalize_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the 'OT' (output target) column and reset the index so that
        the timestamp becomes a plain 'date' column — the format expected
        by iTransformer's Dataset_Custom loader.
        """
        target_col = self.selection_config.get('target_col')

        if target_col and target_col in df.columns:
            print(f"🎯 OT column copied from: '{target_col}'")
            df = df.copy()
            df['OT'] = df[target_col]
        else:
            if target_col:
                print(
                    f"⚠️  target_col '{target_col}' not found in processed data. "
                    f"Falling back to last column."
                )
            else:
                print("ℹ️  No 'target_col' specified. Using the last column as OT.")
            df = df.copy()
            df['OT'] = df[df.columns[-1]]

        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})
        return df
