import yaml
import os
import sys
# Make sure the src can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.etl.wdl_loader import WDLReplayLoader
from src.etl.preprocessor import DataPreprocessor

def main():
    # 1. Load settings
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_path = config['paths']['raw_data']
    out_path = config['paths']['processed_csv']

    # 2. Check output dir
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 3. Run ETL
    loader = WDLReplayLoader()
    df_raw = loader.load(raw_path)

    processor = DataPreprocessor(config['processing'])
    df_final = processor.process(df_raw)

    # 4. Save
    df_final.to_csv(out_path, index=False)
    print(f"🎉 Pipeline Finished! Data saved to {out_path}")

if __name__ == "__main__":
    main()