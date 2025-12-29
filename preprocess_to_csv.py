import pandas as pd
from utils.data_loader import DataLoader

# Passe den Pfad ggf. an, z.B. auf eine vorhandene CSV-Datei mit OHLCV-Daten
RAW_PATH = 'data/btc_2023_5m.csv'  # Input jetzt CSV
OUT_PATH = 'data/preprocessed_2023.csv'

if __name__ == '__main__':
    loader = DataLoader(RAW_PATH, window_size=50)
    df = loader.preprocess()
    # Reset index to save the timestamp as a column
    df.reset_index(inplace=True)
    df.to_csv(OUT_PATH, index=False)
    print(f'Preprocessing abgeschlossen. Datei gespeichert unter: {OUT_PATH}')
