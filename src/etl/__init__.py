"""
ETL (Extract-Transform-Load) package for the AttentionGraph project.
"""
from .base import BaseLoader, BasePreprocessor
from .matlab_loader import MatlabLoader, Matlab2DLoader
from .wdl_loader import WDLReplayLoader
from .preprocessor import MatlabPreprocessor, WDLPreprocessor, Matlab2DPreprocessor

__all__ = [
    "BaseLoader",
    "BasePreprocessor",
    "MatlabLoader",
    "Matlab2DLoader",
    "WDLReplayLoader",
    "MatlabPreprocessor",
    "WDLPreprocessor",
    "Matlab2DPreprocessor",
    "create_etl_pipeline",
]

# Registry maps the config 'type' string to (LoaderClass, PreprocessorClass)
_REGISTRY: dict = {
    "matlab": (MatlabLoader, MatlabPreprocessor),
    "matlab2d": (Matlab2DLoader, Matlab2DPreprocessor),
    "wdl": (WDLReplayLoader, WDLPreprocessor),
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