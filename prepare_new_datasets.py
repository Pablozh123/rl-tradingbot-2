import pandas as pd
import os
from utils.download_data import download_data
from utils.data_loader import DataLoader

def prepare_datasets():
    # 1. Download 2024 Data
    print("--- Downloading 2024 Data ---")
    df_2024 = download_data(since_str='2024-01-01 00:00:00')
    # Filter for 2024 only (just in case)
    df_2024 = df_2024[df_2024['timestamp'].dt.year == 2024]
    df_2024.to_csv('data/btc_2024_5m.csv', index=False)
    print(f"Saved 2024 data: {len(df_2024)} rows")

    # 2. Merge 2022 and 2023 for Training
    print("\n--- Merging 2022 & 2023 for Training ---")
    if not os.path.exists('data/btc_2022_5m.csv') or not os.path.exists('data/btc_2023_5m.csv'):
        print("Error: 2022 or 2023 data missing. Please ensure they exist.")
        return

    df_22 = pd.read_csv('data/btc_2022_5m.csv')
    df_23 = pd.read_csv('data/btc_2023_5m.csv')
    
    df_train = pd.concat([df_22, df_23], ignore_index=True)
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    df_train.sort_values('timestamp', inplace=True)
    df_train.to_csv('data/btc_train_22_23.csv', index=False)
    print(f"Saved Training Data (2022+2023): {len(df_train)} rows")

    # 3. Preprocess Training Data
    print("\n--- Preprocessing Training Data ---")
    loader_train = DataLoader('data/btc_train_22_23.csv')
    df_train_proc = loader_train.preprocess()
    df_train_proc.to_csv('data/preprocessed_train_22_23.csv', index=False)
    print("Saved 'data/preprocessed_train_22_23.csv'")

    # 4. Preprocess Validation Data (2024)
    print("\n--- Preprocessing Validation Data (2024) ---")
    loader_test = DataLoader('data/btc_2024_5m.csv')
    df_test_proc = loader_test.preprocess()
    df_test_proc.to_csv('data/preprocessed_test_2024.csv', index=False)
    print("Saved 'data/preprocessed_test_2024.csv'")

if __name__ == "__main__":
    prepare_datasets()
