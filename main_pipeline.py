"""
main_pipeline.py — Step 1 of the AttentionGraph pipeline.

Reads raw data files, applies the appropriate ETL pipeline (WDL or MATLAB),
and saves the processed CSV ready for iTransformer training.

Configuration is driven entirely by ``config/settings.yaml``; no code
changes are needed when switching data sources — only update
``data_loader.type`` in the YAML.
"""

import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.etl import create_etl_pipeline

# Number of zero-filled rows inserted between consecutive WDL segments
# to prevent the model from learning spurious cross-segment patterns.
_GAP_SIZE = 100


def _build_gap_row(columns) -> pd.DataFrame:
    """Return a zero-filled DataFrame with the given columns."""
    return pd.DataFrame(np.zeros((_GAP_SIZE, len(columns))), columns=columns)


def _run_matlab_pipeline(loader, preprocessor, raw_dir: str, out_path: str) -> None:
    """Single-directory MATLAB pipeline: load → preprocess → save."""
    print(f"🚀 MATLAB mode — processing directory: {raw_dir}")
    df_raw = loader.load(raw_dir)
    df_final = preprocessor.process(df_raw)
    df_final.to_csv(out_path, index=False)
    print(f"🎉 Done. Saved {df_final.shape} → {out_path}")


def _run_wdl_pipeline(
    loader, preprocessor, raw_dir: str, out_path: str, processing_cfg: dict
) -> None:
    """
    Multi-file WDL pipeline: iterate over every file in raw_dir,
    insert zero-padding gaps between segments, then concatenate and save.
    """
    file_list = sorted(
        f for f in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, f))
    )
    if not file_list:
        print(f"❌ No files found in {raw_dir}")
        return

    print(f"📦 Found {len(file_list)} file(s). Batch processing with zero-padding gaps...")
    segments = []

    for idx, filename in enumerate(file_list):
        print(f"   [{idx + 1}/{len(file_list)}] Processing: {filename}")
        try:
            df_raw = loader.load(os.path.join(raw_dir, filename))
            df_part = preprocessor.process(df_raw)

            if segments:
                # Insert a gap between segments so the model sees a clear boundary
                segments.append(_build_gap_row(df_part.columns))

            segments.append(df_part)

        except Exception as exc:
            print(f"   ❌ Skipped '{filename}': {exc}")

    if not segments:
        print("❌ No files were processed successfully.")
        return

    print("🔗 Concatenating all segments...")
    df_final = pd.concat(segments, axis=0, ignore_index=True)

    # Assign a synthetic, continuous date axis over the merged data
    freq = processing_cfg.get('resample_rate', '200ms')
    start = pd.Timestamp(processing_cfg.get('start_date', '2024-01-01'))
    df_final['date'] = pd.date_range(start=start, periods=len(df_final), freq=freq)

    df_final.to_csv(out_path, index=False)
    print(f"🎉 Done. Saved merged data {df_final.shape} → {out_path}")


def main() -> None:
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_dir = config['paths']['raw_data']
    out_path = config['paths']['processed_csv']
    loader_type = config.get('data_loader', {}).get('type', 'wdl')
    processing_cfg = config.get('processing', {})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    loader, preprocessor = create_etl_pipeline(config)

    if loader_type == 'matlab':
        _run_matlab_pipeline(loader, preprocessor, raw_dir, out_path)
    else:
        _run_wdl_pipeline(loader, preprocessor, raw_dir, out_path, processing_cfg)


if __name__ == "__main__":
    main()
