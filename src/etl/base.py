from abc import ABC, abstractmethod
import pandas as pd

class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> pd.DataFrame:
        """Load raw data and return DataFrame"""
        pass