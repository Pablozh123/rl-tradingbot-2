import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'data/preprocessed_data.csv'

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    if 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'])
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

def run_backtest(df, sl_strategy='tight', sl_param=0.0, filters=None):
    """
    Simulates SFP trades with a specific SL strategy and filters.
    sl_strategy: 'tight' (Low), 'pct' (Low * (1-param)), 'atr' (Low - ATR*param)
    filters: dict of filters e.g. {'volume': 1.5, 'wick': 0.001, 'close_pos': 0.6}
    """
    balance = 10000
    trades = []
    
    # State
    in_position = False
    entry_price = 0
    sl_price = 0
    side = None
    last_pivot_level = -1
    
    # Pre-calculate indicators for filters
    df['tr'] = df['High'] - df['Low']
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['vol_ma'] = df['Volume'].rolling(window=20).mean()
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # Check Exit first
        if in_position:
            # Check SL
            if side == 'long':
                if row['Low'] <= sl_price:
                    pnl = (sl_price - entry_price) / entry_price
                    balance *= (1 + pnl)
                    trades.append({'type': 'sl', 'pnl': pnl})
                    in_position = False
                elif row['Close'] > entry_price * 1.03: # Reduced TP to 3% for more hits
                     pnl = 0.03
                     balance *= (1 + pnl)
                     trades.append({'type': 'tp', 'pnl': pnl})
                     in_position = False
                
            elif side == 'short':
                if row['High'] >= sl_price:
                    pnl = (entry_price - sl_price) / entry_price
                    balance *= (1 + pnl)
                    trades.append({'type': 'sl', 'pnl': pnl})
                    in_position = False
                elif row['Close'] < entry_price * 0.97: # Reduced TP to 3%
                     pnl = 0.03
                     balance *= (1 + pnl)
                     trades.append({'type': 'tp', 'pnl': pnl})
                     in_position = False

        # Check Entry
        if not in_position:
            # Apply Filters
            valid_filter = True
            if filters:
                if 'volume' in filters and row['Volume'] < (row['vol_ma'] * filters['volume']):
                    valid_filter = False
                
                # Wick Size (Sweep depth)
                if 'wick' in filters:
                    if row['is_sfp_long']:
                        sweep_depth = (row['active_pivot_low'] - row['Low']) / row['Close']
                        if sweep_depth < filters['wick']: valid_filter = False
                    elif row['is_sfp_short']:
                        sweep_depth = (row['High'] - row['active_pivot_high']) / row['Close']
                        if sweep_depth < filters['wick']: valid_filter = False
                
                # Close Position (Rejection strength)
                if 'close_pos' in filters:
                    candle_range = row['High'] - row['Low']
                    if candle_range > 0:
                        if row['is_sfp_long']:
                            pos = (row['Close'] - row['Low']) / candle_range
                            if pos < filters['close_pos']: valid_filter = False
                        elif row['is_sfp_short']:
                            pos = (row['High'] - row['Close']) / candle_range
                            if pos < filters['close_pos']: valid_filter = False

            if valid_filter:
                # SFP Long
                if row['is_sfp_long'] and row['active_pivot_low'] != last_pivot_level:
                    entry_price = row['Close']
                    side = 'long'
                    last_pivot_level = row['active_pivot_low']
                    
                    # SL Logic
                    if sl_strategy == 'tight': sl_price = row['Low']
                    elif sl_strategy == 'pct': sl_price = row['Low'] * (1 - sl_param)
                    elif sl_strategy == 'atr': sl_price = row['Low'] - (row['atr'] * sl_param)
                    
                    in_position = True
                    
                # SFP Short
                elif row['is_sfp_short'] and row['active_pivot_high'] != last_pivot_level:
                    entry_price = row['Close']
                    side = 'short'
                    last_pivot_level = row['active_pivot_high']
                    
                    # SL Logic
                    if sl_strategy == 'tight': sl_price = row['High']
                    elif sl_strategy == 'pct': sl_price = row['High'] * (1 + sl_param)
                    elif sl_strategy == 'atr': sl_price = row['High'] + (row['atr'] * sl_param)
                    
                    in_position = True

    # Calculate Stats
    if not trades:
        return 0, 0, 0
        
    df_trades = pd.DataFrame(trades)
    win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades)
    total_return = (balance - 10000) / 10000 * 100
    
    return win_rate, total_return, len(trades)

if __name__ == "__main__":
    df = load_data()
    
    print("\n--- Testing SFP Filters (TP=3%, SL=Buffer 0.5%) ---")
    
    base_sl_strat = 'pct'
    base_sl_param = 0.005
    
    # 1. Baseline (No Filters)
    wr, ret, count = run_backtest(df, base_sl_strat, base_sl_param, filters=None)
    print(f"Baseline (No Filters) | Win Rate: {wr:.2%} | Return: {ret:.2f}% | Trades: {count}")
    
    # 2. Volume Filter (> 1.5x Avg)
    wr, ret, count = run_backtest(df, base_sl_strat, base_sl_param, filters={'volume': 1.5})
    print(f"Volume Filter (1.5x)  | Win Rate: {wr:.2%} | Return: {ret:.2f}% | Trades: {count}")
    
    # 3. Wick Filter (> 0.1% Sweep)
    wr, ret, count = run_backtest(df, base_sl_strat, base_sl_param, filters={'wick': 0.001})
    print(f"Wick Filter (0.1%)    | Win Rate: {wr:.2%} | Return: {ret:.2f}% | Trades: {count}")
    
    # 4. Close Position (> 60%)
    wr, ret, count = run_backtest(df, base_sl_strat, base_sl_param, filters={'close_pos': 0.6})
    print(f"Close Pos (> 60%)     | Win Rate: {wr:.2%} | Return: {ret:.2f}% | Trades: {count}")
    
    # 5. Combined
    wr, ret, count = run_backtest(df, base_sl_strat, base_sl_param, filters={'volume': 1.2, 'wick': 0.0005, 'close_pos': 0.5})
    print(f"Combined (Moderate)   | Win Rate: {wr:.2%} | Return: {ret:.2f}% | Trades: {count}")
