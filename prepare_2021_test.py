import pandas as pd
import os
from utils.download_data import download_data
from utils.data_loader import DataLoader

def prepare_2021_dataset():
    # 1. Download 2021 Data
    print("--- Downloading 2021 Data ---")
    # Download starting from 2021
    df_2021 = download_data(since_str='2021-01-01 00:00:00')
    
    # Filter for 2021 only
    df_2021 = df_2021[df_2021['timestamp'].dt.year == 2021]
    
    if df_2021.empty:
        print("Error: No data found for 2021.")
        return

    output_path = 'data/btc_2021_5m.csv'
    df_2021.to_csv(output_path, index=False)
    print(f"Saved 2021 data: {len(df_2021)} rows to {output_path}")

    # 2. Preprocess 2021 Data
    print("\n--- Preprocessing 2021 Data ---")
    loader = DataLoader(output_path)
    df_proc = loader.preprocess()
    
    proc_output_path = 'data/preprocessed_2021.csv'
    df_proc.to_csv(proc_output_path, index=False)
    print(f"Saved preprocessed data to {proc_output_path}")

if __name__ == "__main__":
    prepare_2021_dataset()
