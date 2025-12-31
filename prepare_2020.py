import pandas as pd
import os
from utils.download_data import download_data
from utils.data_loader import DataLoader

def prepare_2020():
    # 1. Download 2020 Data
    print("--- Downloading 2020 Data ---")
    # Download from Jan 1st 2020
    df_2020 = download_data(since_str='2020-01-01 00:00:00')
    
    # Filter for 2020 only
    df_2020 = df_2020[df_2020['timestamp'].dt.year == 2020]
    
    raw_path = 'data/btc_2020_5m.csv'
    df_2020.to_csv(raw_path, index=False)
    print(f"Saved Raw 2020 Data: {len(df_2020)} rows to {raw_path}")

    # 2. Preprocess 2020 Data
    print("\n--- Preprocessing 2020 Data ---")
    loader = DataLoader(raw_path)
    df_proc = loader.preprocess()
    
    # Reset index to ensure timestamp is a column
    df_proc.reset_index(inplace=True)
    
    proc_path = 'data/preprocessed_2020.csv'
    df_proc.to_csv(proc_path, index=False)
    print(f"Saved Preprocessed 2020 Data to {proc_path}")

if __name__ == "__main__":
    prepare_2020()
