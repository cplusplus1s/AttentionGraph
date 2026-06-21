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
_GAP_SIZE = 50 #_GAP_SIZE > seq_len + pred_len


def _build_gap_row(columns) -> pd.DataFrame:
    """Return a zero-filled DataFrame with the given columns."""
    return pd.DataFrame(np.zeros((_GAP_SIZE, len(columns))), columns=columns)


def _run_matlab_pipeline(loader, preprocessor, raw_dir: str, out_path: str, processing_cfg: dict) -> None:
    """Multi-directory MATLAB pipeline: concatenate with zero-padding for training, and save individuals for inference."""
    print(f"🚀 MATLAB mode — batch processing & merging directory: {raw_dir}")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    subdirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    if not subdirs:
        print(f"❌ No subdirectories found in {raw_dir}")
        return

    # Sort from xxx_1 to xxx_30
    subdirs_sorted = sorted(subdirs, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
    segments = []

    for idx, subdir in enumerate(subdirs_sorted):
        print(f"   [{idx + 1}/{len(subdirs_sorted)}] Loading: {subdir}")
        current_raw_dir = os.path.join(raw_dir, subdir)

        try:
            df_raw = loader.load(current_raw_dir)
            df_part = preprocessor.process(df_raw)

            is_healthy = "unhealthy" not in subdir.lower()
            prefix = "healthy" if is_healthy else "unhealthy"
            n_suffix = subdir.split('_')[-1]

            # Save independent CSV file with suffix for inference
            indiv_name = f"matlab_{prefix}_{n_suffix}.csv"
            df_part.to_csv(os.path.join(out_dir, indiv_name), index=False)

            # Healthy data merge into a large CSV for training
            if is_healthy:
                if segments:
                    segments.append(_build_gap_row(df_part.columns))
                segments.append(df_part)

        except Exception as exc:
            print(f"   ❌ Skipped '{subdir}': {exc}")

    if segments:
        print("\n🔗 Concatenating all healthy segments with 0-padding gaps for MASTER TRAINING...")
        df_final = pd.concat(segments, axis=0, ignore_index=True)

        freq = processing_cfg.get('resample_rate', '200ms')
        start_str = processing_cfg.get('start_date', '2024-01-01')
        start = pd.Timestamp(start_str)

        df_final['date'] = pd.date_range(start=start, periods=len(df_final), freq=freq)

        combined_path = os.path.join(out_dir, "combined_healthy_train.csv")
        df_final.to_csv(combined_path, index=False)
        print(f"🎉 Done! Master training dataset saved -> {combined_path}")


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


def _run_brian2_pipeline(loader, preprocessor, raw_dir: str, out_path: str, processing_cfg: dict) -> None:
    """Multi-directory Brian2 pipeline.

    Pass 1 (healthy):   processed exactly as before; saves individual + combined CSVs.
    Pass 2 (unhealthy): aligned to the healthy column schema; any columns the preprocessor
                        dropped (dead neurons after a fault) are restored as white-noise
                        columns so iTransformer inference sees the same feature dim as
                        during training.
    """
    print(f"🚀 Brian2 mode — batch processing & merging directory: {raw_dir}")

    out_dir = os.path.abspath("./data/processed")
    os.makedirs(out_dir, exist_ok=True)

    freq = processing_cfg.get('resample_rate', '1ms')
    start_str = processing_cfg.get('start_date', '2024-01-01')
    start = pd.Timestamp(start_str)

    subdirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    if not subdirs:
        print(f"❌ No subdirectories found in {raw_dir}")
        return

    def _suffix_num(name: str) -> int:
        tail = name.split('_')[-1]
        return int(tail) if tail.isdigit() else 0

    healthy_subdirs   = sorted([d for d in subdirs if "unhealthy" not in d.lower()], key=_suffix_num)
    unhealthy_subdirs = sorted([d for d in subdirs if "unhealthy"     in d.lower()], key=_suffix_num)

    # ---- Shared per-subdir load+preprocess (date column added, time_sec dropped) -----
    def _process_subdir(subdir: str):
        current_raw_dir = os.path.join(raw_dir, subdir)
        csv_files = [f for f in os.listdir(current_raw_dir) if f.endswith('.csv')]
        if not csv_files:
            return None
        df_raw = loader.load(os.path.join(current_raw_dir, csv_files[0]))
        df_part = preprocessor.process(df_raw)
        if 'time_sec' in df_part.columns:
            df_part = df_part.drop(columns=['time_sec'])
        df_part['date'] = pd.date_range(start=start, periods=len(df_part), freq=freq)
        cols = ['date'] + [c for c in df_part.columns if c != 'date']
        return df_part[cols]

    # ==========================================================================
    # Pass 1 — HEALTHY (unchanged behavior + capture schema & noise scale)
    # ==========================================================================
    healthy_schema = None        # full column list, incl. 'date'
    healthy_noise_scale = None   # σ for the white noise used in Pass 2
    segments = []

    for idx, subdir in enumerate(healthy_subdirs):
        print(f"   [healthy {idx + 1}/{len(healthy_subdirs)}] Loading: {subdir}")
        try:
            df_part = _process_subdir(subdir)
            if df_part is None:
                continue

            if healthy_schema is None:
                healthy_schema = list(df_part.columns)
                sensor_cols = [c for c in healthy_schema if c != 'date']
                # Use the smallest non-zero per-column std as the white-noise σ.
                # This makes restored dead-neuron columns look like the quietest
                # real sensor — small enough not to drown real signal, large
                # enough that the column isn't flat.
                stds = df_part[sensor_cols].std().replace(0, np.nan).dropna()
                healthy_noise_scale = float(stds.min()) if len(stds) else 1e-3
                print(f"   📐 Healthy schema captured: {len(sensor_cols)} sensor columns")
                print(f"   📐 White-noise σ for dead columns: {healthy_noise_scale:.4g}")

            n_suffix = subdir.split('_')[-1]
            df_part.to_csv(os.path.join(out_dir, f"brian2_healthy_{n_suffix}.csv"), index=False)

            if segments:
                segments.append(_build_gap_row(df_part.columns))
            segments.append(df_part)

        except Exception as exc:
            print(f"   ❌ Skipped '{subdir}': {exc}")

    if segments:
        print("\n🔗 Concatenating all healthy segments with 0-padding gaps for MASTER TRAINING...")
        df_final = pd.concat(segments, axis=0, ignore_index=True)
        df_final['date'] = pd.date_range(start=start, periods=len(df_final), freq=freq)
        cols = ['date'] + [c for c in df_final.columns if c != 'date']
        df_final = df_final[cols]

        combined_path = os.path.join(out_dir, "combined_healthy_train.csv")
        df_final.to_csv(combined_path, index=False)
        print(f"🎉 Done! Master training dataset saved -> {combined_path}")

    # ==========================================================================
    # Pass 2 — UNHEALTHY (align to healthy schema, fill missing with white noise)
    # ==========================================================================
    if not unhealthy_subdirs:
        return
    if healthy_schema is None:
        print("⚠️  No healthy data was processed — cannot align unhealthy schema. Skipping unhealthy.")
        return

    sensor_schema = [c for c in healthy_schema if c != 'date']
    rng = np.random.default_rng()

    for idx, subdir in enumerate(unhealthy_subdirs):
        print(f"   [unhealthy {idx + 1}/{len(unhealthy_subdirs)}] Loading: {subdir}")
        try:
            df_part = _process_subdir(subdir)
            if df_part is None:
                continue

            # Restore any columns the preprocessor dropped (dead neurons) with white noise.
            missing = [c for c in sensor_schema if c not in df_part.columns]
            if missing:
                n = len(df_part)
                for c in missing:
                    df_part[c] = rng.normal(loc=0.0, scale=healthy_noise_scale, size=n)
                preview = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
                print(f"      ⚠️  Filled {len(missing)} dead-neuron column(s) "
                      f"with white noise (σ={healthy_noise_scale:.4g}): {preview}")

            # Drop any unexpected extra columns (defensive — shouldn't happen, but keeps schema exact)
            extras = [c for c in df_part.columns if c not in healthy_schema]
            if extras:
                df_part = df_part.drop(columns=extras)
                print(f"      ℹ️  Dropped {len(extras)} unexpected extra column(s): {extras[:5]}")

            # Reorder to match healthy column order exactly
            df_part = df_part[healthy_schema]

            n_suffix = subdir.split('_')[-1]
            df_part.to_csv(os.path.join(out_dir, f"brian2_unhealthy_{n_suffix}.csv"), index=False)

        except Exception as exc:
            print(f"   ❌ Skipped '{subdir}': {exc}")


def main() -> None:
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_dir = config['paths']['raw_data']
    out_path = config['paths']['processed_csv']
    loader_type = config.get('data_loader', {}).get('type', 'wdl')
    processing_cfg = config.get('processing', {})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    loader, preprocessor = create_etl_pipeline(config)

    if loader_type in ['matlab', 'matlab2d']:
        _run_matlab_pipeline(loader, preprocessor, raw_dir, out_path, processing_cfg)
    elif loader_type == 'brian2':
        _run_brian2_pipeline(loader, preprocessor, raw_dir, out_path, processing_cfg)
    else:
        _run_wdl_pipeline(loader, preprocessor, raw_dir, out_path, processing_cfg)


if __name__ == "__main__":
    main()
