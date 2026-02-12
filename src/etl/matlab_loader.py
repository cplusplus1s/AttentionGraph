import os
import pandas as pd
from scipy.io import loadmat
from src.etl.base import BaseLoader

class MatlabDataLoader(BaseLoader):
    def __init__(self, start_date="2024-01-01", resample_rate="10ms"):
        self.start_date = pd.Timestamp(start_date)
        self.resample_rate = resample_rate

    def load(self, root_path: str) -> pd.DataFrame:
        all_features = []
        common_time = None
        metrics = ['acc', 'vel', 'pos']

        mass_dirs = [d for d in os.listdir(root_path) if d.startswith('mass_')]
        mass_dirs.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

        for mass_dir in mass_dirs:
            mass_df = pd.DataFrame()
            for metric in metrics:
                file_path = os.path.join(root_path, mass_dir, metric, f'{metric}_node.mat')
                if os.path.exists(file_path):
                    mat_dict = loadmat(file_path)
                    struct_s = mat_dict['S'][0, 0]
                    data = struct_s['data'].flatten()
                    if common_time is None:
                        common_time = struct_s['time'].flatten()
                    mass_df[f"{mass_dir}_{metric}"] = data
            all_features.append(mass_df)

        df = pd.concat(all_features, axis=1)

        df.index = self.start_date + pd.to_timedelta(common_time, unit='s')
        df = df.resample(self.resample_rate).mean().ffill().bfill()

        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        return df