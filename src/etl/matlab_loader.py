import os
import pandas as pd
from scipy.io import loadmat

from .base import BaseLoader


class MatlabLoader(BaseLoader):
    """
    Loads MATLAB simulation data from a structured directory tree.

    Expected directory layout::

        root_path/
        └── mass_N/
            ├── acc/acc_node.mat
            ├── vel/vel_node.mat
            └── pos/pos_node.mat

    Returns a DataFrame with a ``TimedeltaIndex`` (seconds from t=0) and
    columns named ``mass_N_<metric>``.  Resampling is intentionally left
    to ``MatlabPreprocessor`` so that the loader has a single responsibility.
    """

    _METRICS = ('acc', 'vel', 'pos')

    def load(self, root_path: str) -> pd.DataFrame:
        if not os.path.isdir(root_path):
            raise FileNotFoundError(f"MATLAB data directory not found: {root_path}")

        mass_dirs = sorted(
            [d for d in os.listdir(root_path) if d.startswith('mass_')],
            key=lambda name: int(''.join(filter(str.isdigit, name)))
        )
        if not mass_dirs:
            raise ValueError(f"No 'mass_*' subdirectories found in: {root_path}")

        all_series = []
        common_time = None

        for mass_dir in mass_dirs:
            for metric in self._METRICS:
                file_path = os.path.join(root_path, mass_dir, metric, f'{metric}_node.mat')
                if not os.path.exists(file_path):
                    continue

                mat_dict = loadmat(file_path)
                struct_s = mat_dict['S'][0, 0]
                data = struct_s['data'].flatten()

                if common_time is None:
                    common_time = struct_s['time'].flatten()

                all_series.append(
                    pd.Series(data, name=f"{mass_dir}_{metric}")
                )

        if not all_series or common_time is None:
            raise ValueError(f"No valid .mat files found under: {root_path}")

        df = pd.concat(all_series, axis=1)
        df.index = pd.to_timedelta(common_time, unit='s')

        print(f"✅ Loaded MATLAB data: {df.shape} from {len(mass_dirs)} mass block(s).")
        return df


class Matlab2DLoader(BaseLoader):
    """
    Loads 2D MATLAB simulation data.
    Extracts both 'data_x' and 'data_y' and creates separate feature columns (24-dim).
    """
    _METRICS = ('acc', 'vel', 'pos')

    def load(self, root_path: str) -> pd.DataFrame:
        if not os.path.isdir(root_path):
            raise FileNotFoundError(f"MATLAB data directory not found: {root_path}")

        mass_dirs = sorted(
            [d for d in os.listdir(root_path) if d.startswith('mass_')],
            key=lambda name: int(''.join(filter(str.isdigit, name)))
        )

        if not mass_dirs:
            raise ValueError(f"No 'mass_*' subdirectories found in: {root_path}")

        all_series = []
        common_time = None

        for mass_dir in mass_dirs:
            for metric in self._METRICS:
                file_path = os.path.join(root_path, mass_dir, metric, f'{metric}_node.mat')
                if not os.path.exists(file_path):
                    continue

                mat_dict = loadmat(file_path)
                struct_s = mat_dict['S'][0, 0]

                if 'data_x' in struct_s.dtype.names and 'data_y' in struct_s.dtype.names:
                    data_x = struct_s['data_x'].flatten()
                    data_y = struct_s['data_y'].flatten()
                    all_series.append(pd.Series(data_x, name=f"{mass_dir}_{metric}_x"))
                    all_series.append(pd.Series(data_y, name=f"{mass_dir}_{metric}_y"))
                else:
                    if 'data' in struct_s.dtype.names:
                        data = struct_s['data'].flatten()
                        all_series.append(pd.Series(data, name=f"{mass_dir}_{metric}"))

                if common_time is None and 'time' in struct_s.dtype.names:
                    common_time = struct_s['time'].flatten()

        if not all_series or common_time is None:
            raise ValueError(f"No valid .mat files found under: {root_path}")

        df = pd.concat(all_series, axis=1)

        td_index = pd.to_timedelta(common_time, unit='s')
        df.index = td_index
        df = df.groupby(df.index).mean()

        return df