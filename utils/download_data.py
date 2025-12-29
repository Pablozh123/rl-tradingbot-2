import ccxt
import pandas as pd
import time
from datetime import datetime

def download_data(symbol='BTC/USDT', timeframe='5m', since_str='2024-01-01 00:00:00', limit=1000):
    print(f"Downloading {symbol} ({timeframe}) from {since_str}...")
    exchange = ccxt.binance({'options': {'defaultType': 'future'}})
    since = exchange.parse8601(since_str)
    
    all_candles = []
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not candles:
                break
            
            since = candles[-1][0] + 1 # Next timestamp
            all_candles += candles
            
            print(f"Downloaded {len(all_candles)} candles total. Last date: {datetime.fromtimestamp(candles[-1][0]/1000)}")
            
            if len(candles) < limit:
                break
                
            time.sleep(0.1) # Rate limit
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            
    df = pd.DataFrame(all_candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

if __name__ == "__main__":
    # Example: Download data from Jan 2022
    # You can change the date here
    START_DATE = '2022-01-01 00:00:00'
    FILENAME = 'data/btc_2022_5m.csv'
    
    df = download_data(since_str=START_DATE)
    # Filter to only include 2022 data (optional, but good for cleanliness)
    df = df[df['timestamp'].dt.year == 2022]
    
    df.to_csv(FILENAME, index=False)
    print(f"Saved {len(df)} rows to {FILENAME}")
