import yaml
import os
import sys
import pandas as pd
import numpy as np

# Make sure the src can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.etl.wdl_loader import WDLReplayLoader, BaseLoader
from src.etl.matlab_loader import MatlabDataLoader
from src.etl.preprocessor import DataPreprocessor

def get_loader(config) -> BaseLoader:
    """
    The simple factory pattern: the Loader type depends on data_loader type
    """
    loader_cfg = config.get('data_loader', {})
    loader_type = loader_cfg.get('type', 'wdl') # default type is wdl

    start_date = config.get('processing', {}).get('start_date', '2024-01-01')
    resample_rate = config.get('processing', {}).get('resample_rate', '10ms')

    if loader_type == "matlab":
        print(f"Using MatlabDataLoader with start_date: {start_date}, rate: {resample_rate}")
        return MatlabDataLoader(start_date=start_date, resample_rate=resample_rate)

    elif loader_type == "wdl":
        print("Using WDLLoader")
        return WDLReplayLoader()

    else:
        raise ValueError(f"Unsupported data_loader type: {loader_type}")

def main():
    # Load settings
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Path setup
    raw_dir = config['paths']['raw_data']
    out_path = config['paths']['processed_csv']
    # Check loader type
    loader_type = config.get('data_loader', {}).get('type', 'wdl')
    GAP_SIZE = 100

    # Check output dir
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Instantiate processor and loader
    processor = DataPreprocessor(config['processing'])
    loader = get_loader(config)
    processed_dfs = []

    # Matlab
    if loader_type == "matlab":
        print(f"🚀 Detected MATLAB data structure. Processing directory: {raw_dir}")
        try:
            # Load and resample
            df_raw = loader.load(raw_dir)

            df_selected = processor._select_columns(df_raw.set_index('date'))
            df_clean = processor._drop_constant_columns(df_selected)
            df_final = processor._finalize_format(df_clean)

            df_final.to_csv(out_path, index=False)
            print(f"🎉 MATLAB Pipeline Finished! Shape: {df_final.shape}")
            return
        except Exception as e:
            print(f"❌ Error during MATLAB batch processing: {str(e)}")
            return
    # --------------------------------

    # WDL
    file_list = sorted([f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))])

    if not file_list:
        print(f"❌ No files found in {raw_dir}")
        return

    print(f"📦 Found {len(file_list)} files. Starting Batch Processing with Zero-Padding...")

    # Loop & Process
    for i, filename in enumerate(file_list):
        raw_path_file = os.path.join(raw_dir, filename)
        print(f"   [{i+1}/{len(file_list)}] 🔄 Processing: {filename}")

        try:
            df_raw = loader.load(raw_path_file)
            df_part = processor.process(df_raw)

            # Insert 0-Padding
            if len(processed_dfs) > 0:
                print(f"       ➕ Inserting Gap ({GAP_SIZE} rows)...")
                gap_df = pd.DataFrame(
                    np.zeros((GAP_SIZE, len(df_part.columns))),
                    columns=df_part.columns
                )
                processed_dfs.append(gap_df)

            processed_dfs.append(df_part)

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {str(e)}")
            continue

    if not processed_dfs:
        print("❌ No data processed.")
        return

    # Concatenation
    print("🔗 Concatenating all segments...")
    df_final = pd.concat(processed_dfs, axis=0, ignore_index=True)

    # Set synthetic timeline
    freq = config['processing'].get('resample_rate', '200ms')
    start_date_str = config['processing'].get('start_date', "2024-01-01")
    start_date = pd.Timestamp(start_date_str)

    print(f"⏳ Re-indexing synthetic time axis (Start: {start_date}, Freq: {freq})...")
    new_dates = pd.date_range(start=start_date, periods=len(df_final), freq=freq)
    df_final['date'] = new_dates

    # Save
    df_final.to_csv(out_path, index=False)
    print(f"🎉 Pipeline Finished! Merged Data Shape: {df_final.shape}")

if __name__ == "__main__":
    main()