"""
ETL (Extract-Transform-Load) package for the AttentionGraph project.
"""
from .base import BaseLoader, BasePreprocessor
from .matlab_loader import MatlabLoader, Matlab2DLoader
from .wdl_loader import WDLReplayLoader
from .brian2_loader import Brian2Loader
from .preprocessor import MatlabPreprocessor, WDLPreprocessor, Matlab2DPreprocessor, Brian2Preprocessor

__all__ = [
    "BaseLoader",
    "BasePreprocessor",
    "MatlabLoader",
    "Matlab2DLoader",
    "WDLReplayLoader",
    "Brian2Loader",
    "MatlabPreprocessor",
    "WDLPreprocessor",
    "Matlab2DPreprocessor",
    "Brian2Preprocessor",
    "create_etl_pipeline",
]

# Registry maps the config 'type' string to (LoaderClass, PreprocessorClass)
_REGISTRY: dict = {
    "matlab": (MatlabLoader, MatlabPreprocessor),
    "matlab2d": (Matlab2DLoader, Matlab2DPreprocessor),
    "wdl": (WDLReplayLoader, WDLPreprocessor),
    "brian2": (Brian2Loader, Brian2Preprocessor),
}

def create_etl_pipeline(config: dict) -> tuple[BaseLoader, BasePreprocessor]:
    loader_type: str = config.get('data_loader', {}).get('type', 'wdl')
    processing_cfg: dict = config.get('processing', {})

    if loader_type not in _REGISTRY:
        raise ValueError(
            f"Unknown data_loader type '{loader_type}'. "
            f"Supported values: {list(_REGISTRY)}"
        )

    LoaderCls, PreprocessorCls = _REGISTRY[loader_type]
    loader = LoaderCls()
    preprocessor = PreprocessorCls(processing_cfg)

    print(
        f"[ETL] Loader      : {LoaderCls.__name__}\n"
        f"[ETL] Preprocessor: {PreprocessorCls.__name__}"
    )

    return loader, preprocessor