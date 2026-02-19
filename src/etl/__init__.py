"""
ETL (Extract-Transform-Load) package for the AttentionGraph project.

Public API
----------
- ``BaseLoader``           : Abstract base for all raw-data loaders.
- ``BasePreprocessor``     : Abstract base for all preprocessors (shared helpers included).
- ``MatlabLoader``         : Loads MATLAB mass-spring-damper simulation data.
- ``WDLReplayLoader``      : Loads WDL Replay export files.
- ``MatlabPreprocessor``   : Preprocesses MATLAB loader output.
- ``WDLPreprocessor``      : Preprocesses WDL loader output.
- ``create_etl_pipeline``  : Factory that returns the (loader, preprocessor) pair
                             appropriate for the configured data source type.
"""

from .base import BaseLoader, BasePreprocessor
from .matlab_loader import MatlabLoader
from .wdl_loader import WDLReplayLoader
from .preprocessor import MatlabPreprocessor, WDLPreprocessor

__all__ = [
    "BaseLoader",
    "BasePreprocessor",
    "MatlabLoader",
    "WDLReplayLoader",
    "MatlabPreprocessor",
    "WDLPreprocessor",
    "create_etl_pipeline",
]

# Registry maps the config 'type' string to (LoaderClass, PreprocessorClass)
_REGISTRY: dict = {
    "matlab": (MatlabLoader, MatlabPreprocessor),
    "wdl":    (WDLReplayLoader, WDLPreprocessor),
}


def create_etl_pipeline(config: dict) -> tuple[BaseLoader, BasePreprocessor]:
    """
    Factory function that reads ``config['data_loader']['type']`` and
    returns the matching ``(loader, preprocessor)`` pair.

    :param config: The full parsed settings.yaml dictionary.
    :returns: A ``(loader, preprocessor)`` tuple ready to use.
    :raises ValueError: If the configured type is not registered.

    Usage::

        loader, preprocessor = create_etl_pipeline(config)
        df_raw  = loader.load(raw_path)
        df_clean = preprocessor.process(df_raw)
    """
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
