import ccxt
import pandas as pd
import time
import os

SYMBOL = 'BTC/USDT:USDT'  # Binance Futures Symbol
TIMEFRAME = '5m'
LIMIT = 1500  # Max Kerzen pro Anfrage
TOTAL_BARS = 50000
DATA_DIR = 'data'
OUT_PATH = os.path.join(DATA_DIR, 'btc_5m.csv')

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    exchange = ccxt.binanceusdm()
    all_ohlcv = []
    since = None
    print('Starte Download...')
    while len(all_ohlcv) < TOTAL_BARS:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=LIMIT)
        if not ohlcv:
            print('Keine weiteren Daten erhalten. Abbruch.')
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # Nächster Startzeitpunkt
        print(f"Geladen: {len(all_ohlcv)} / {TOTAL_BARS}")
        if len(ohlcv) < LIMIT:
            break  # Keine weiteren Daten verfügbar
        time.sleep(0.5)
    # Nur die letzten TOTAL_BARS behalten
    all_ohlcv = all_ohlcv[-TOTAL_BARS:]
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_csv(OUT_PATH, index=False)
    print(f'Download abgeschlossen. Datei gespeichert unter: {OUT_PATH}')
