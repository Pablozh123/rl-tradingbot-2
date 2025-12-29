import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import linregress

# --- 1. Data Fetching ---
def fetch_ohlcv(symbol, timeframe, limit):
    exchange = ccxt.binanceusdm()
    # We load markets only once implicitly or if needed, avoiding print loops here
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df

# Main Execution
SYMBOL = 'BTC/USDT:USDT' 
print(f"Loading data for {SYMBOL}...") 
df_4h = fetch_ohlcv(SYMBOL, '4h', 500)
df_1h = fetch_ohlcv(SYMBOL, '1h', 2000)

if df_4h.empty or df_1h.empty:
    raise ValueError("Data could not be loaded.")

# --- 2. 4H Pivot Definition ---
ORDER = 3
# Find Pivots
high_idx = argrelextrema(df_4h['high'].values, np.greater_equal, order=ORDER)[0]
low_idx = argrelextrema(df_4h['low'].values, np.less_equal, order=ORDER)[0]

df_4h['is_pivot_high'] = False
df_4h['is_pivot_low'] = False
df_4h.iloc[high_idx, df_4h.columns.get_loc('is_pivot_high')] = True
df_4h.iloc[low_idx, df_4h.columns.get_loc('is_pivot_low')] = True

df_4h['pivot_high'] = np.where(df_4h['is_pivot_high'], df_4h['high'], np.nan)
df_4h['pivot_low'] = np.where(df_4h['is_pivot_low'], df_4h['low'], np.nan)

# Active Pivots (ffill) - These are the lines for the SFP
df_4h['active_pivot_high'] = df_4h['pivot_high'].ffill()
df_4h['active_pivot_low'] = df_4h['pivot_low'].ffill()


# --- 3. Golden Pocket Calculation (on 4h Basis) ---
df_4h['bullish_range_low'] = np.nan
df_4h['bullish_range_high'] = np.nan
df_4h['bearish_range_low'] = np.nan
df_4h['bearish_range_high'] = np.nan

for i in range(1, len(df_4h)):
    # Bullish Range Logic
    if not pd.isna(df_4h['active_pivot_low'].iloc[i]):
        low = df_4h['active_pivot_low'].iloc[i]
        high = df_4h['high'].iloc[i] 
        if high > low:
            range_size = high - low
            df_4h.at[df_4h.index[i], 'bullish_range_low'] = high - (range_size * 0.67)
            df_4h.at[df_4h.index[i], 'bullish_range_high'] = high - (range_size * 0.618)

    # Bearish Range Logic
    if not pd.isna(df_4h['active_pivot_high'].iloc[i]):
        high = df_4h['active_pivot_high'].iloc[i]
        low = df_4h['low'].iloc[i]
        if low < high:
            range_size = high - low
            df_4h.at[df_4h.index[i], 'bearish_range_high'] = low + (range_size * 0.67)
            df_4h.at[df_4h.index[i], 'bearish_range_low'] = low + (range_size * 0.618)


# --- 4. Merge 4h Levels into 1h Data ---
# ERROR FIX: We must include active_pivot_high/low in this list!
cols_to_merge = [
    'active_pivot_high', 'active_pivot_low', 
    'bullish_range_low', 'bullish_range_high', 
    'bearish_range_low', 'bearish_range_high'
]

df_1h_reset = df_1h.reset_index().sort_values('timestamp')
df_4h_reset = df_4h[cols_to_merge].dropna(how='all').reset_index().sort_values('timestamp')

merged = pd.merge_asof(
    df_1h_reset,
    df_4h_reset,
    on='timestamp',
    direction='backward'
)
df_1h = merged.set_index('timestamp')
print("Spalten nach Merge:", df_1h.columns)


# --- 5. Signal Logic on 1h Basis ---

# A) SFP Detection (Now df_1h has the active_pivot_high column)
df_1h['is_bearish_sfp'] = (df_1h['high'] > df_1h['active_pivot_high']) & (df_1h['close'] < df_1h['active_pivot_high'])
df_1h['is_bullish_sfp'] = (df_1h['low'] < df_1h['active_pivot_low']) & (df_1h['close'] > df_1h['active_pivot_low'])

# B) Grind Detection
def rolling_r2(series, window=20):
    def get_r2(y):
        if len(y) < 2: return 0
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return r_value**2
    return series.rolling(window=window).apply(get_r2, raw=True)

df_1h['trend_r2'] = rolling_r2(df_1h['close'], window=20)
df_1h['is_grind'] = df_1h['trend_r2'] > 0.85

# C) Golden Pocket Entry Check
df_1h['gp_entry_signal'] = 'none'
df_1h['gp_short_signal'] = False

# Bullish Entry Check
bull_cond = (df_1h['low'] >= df_1h['bullish_range_low']) & (df_1h['low'] <= df_1h['bullish_range_high'])
df_1h.loc[bull_cond, 'gp_entry_signal'] = 'bullish'

# Bearish Entry Check
bear_cond = (df_1h['high'] >= df_1h['bearish_range_low']) & (df_1h['high'] <= df_1h['bearish_range_high'])
df_1h.loc[bear_cond, 'gp_entry_signal'] = 'bearish'


bear_gp_touched = False
for i in range(len(df_1h)):
    row = df_1h.iloc[i]
    # Short Signal: High taucht in Bearish GP Zone ein
    if not bear_gp_touched and not pd.isna(row['bearish_range_low']) and not pd.isna(row['bearish_range_high']):
        if row['high'] >= row['bearish_range_low'] and row['high'] <= row['bearish_range_high']:
            df_1h.iloc[i, df_1h.columns.get_loc('gp_short_signal')] = True
            bear_gp_touched = True
    # Reset Signal wenn neue Range beginnt
    if i > 0 and (row['bearish_range_low'] != df_1h.iloc[i-1]['bearish_range_low'] or row['bearish_range_high'] != df_1h.iloc[i-1]['bearish_range_high']):
        bear_gp_touched = False


# SFP-Logik mit Gewichtungskategorien (hoch/mittel/niedrig)
if 'sfp_weight' not in df_1h.columns:
    df_1h['sfp_weight'] = 0

df_1h['sfp_weight_cat'] = 'niedrig'
for i in range(len(df_1h)):
    weight = df_1h.iloc[i]['sfp_weight']
    # 1h-Kerzen zu Tagen umrechnen
    days = weight / 24 if weight > 0 else 0
    if days >= 30:
        df_1h.iloc[i, df_1h.columns.get_loc('sfp_weight_cat')] = 'hoch'
    elif days >= 7:
        df_1h.iloc[i, df_1h.columns.get_loc('sfp_weight_cat')] = 'mittel'
    elif days > 0:
        df_1h.iloc[i, df_1h.columns.get_loc('sfp_weight_cat')] = 'niedrig'


# --- 6. Plotting ---
print("Creating Plot...")
plot_data = df_1h.iloc[-300:] 

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Chart 1: Price & Pattern
ax1.plot(plot_data.index, plot_data['close'], label='1h Close', color='black', alpha=0.6)

# 4h Pivots
ax1.plot(plot_data.index, plot_data['active_pivot_high'], color='red', linestyle='--', alpha=0.3, label='4h Pivot High')
ax1.plot(plot_data.index, plot_data['active_pivot_low'], color='green', linestyle='--', alpha=0.3, label='4h Pivot Low')

# Golden Pockets Areas
ax1.fill_between(plot_data.index, plot_data['bullish_range_low'], plot_data['bullish_range_high'], color='green', alpha=0.15, label='Bull GP')
ax1.fill_between(plot_data.index, plot_data['bearish_range_low'], plot_data['bearish_range_high'], color='red', alpha=0.15, label='Bear GP')

# Signal Markers
# SFP
sfp_bear = plot_data[plot_data['is_bearish_sfp']]
ax1.scatter(sfp_bear.index, sfp_bear['high'], color='red', marker='v', s=50, label='Bearish SFP', zorder=5)

sfp_bull = plot_data[plot_data['is_bullish_sfp']]
ax1.scatter(sfp_bull.index, sfp_bull['low'], color='green', marker='^', s=50, label='Bullish SFP', zorder=5)

# GP Entries
gp_long = plot_data[plot_data['gp_entry_signal'] == 'bullish']
ax1.scatter(gp_long.index, gp_long['low'], color='blue', marker='D', s=30, label='GP Long Touch')

# Golden Pocket Entry (Short)
gp_short = plot_data[plot_data['gp_short_signal']]
ax1.scatter(gp_short.index, gp_short['close'], color='purple', label='GP Short Entry', marker='X', s=80)

# Grinds
grinds = plot_data[plot_data['is_grind']]
ax1.scatter(grinds.index, grinds['close'], color='lime', s=10, label='Grind')

ax1.set_title(f'Validation: {SYMBOL} 1H (with 4H Structure)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.1)

# Chart 2: R2 Indicator
ax2.plot(plot_data.index, plot_data['trend_r2'], color='orange', label='R² (Grind Strength)')
ax2.axhline(0.85, color='red', linestyle='--', label='Threshold (0.85)')
ax2.set_ylabel('R² Value')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('validation_fixed.png')
print("Done! Plot saved as validation_fixed.png")

# Visualisierung für Juli 2025 - Oktober 2025
start_jul = pd.Timestamp('2025-07-01')
end_oct = pd.Timestamp('2025-10-31 23:59:59')
df_jul_oct = df_1h[(df_1h.index >= start_jul) & (df_1h.index <= end_oct)]

plt.figure(figsize=(18, 8))
plt.plot(df_jul_oct.index, df_jul_oct['close'], label='Close', color='black', linewidth=1)
farben = {'hoch': 'red', 'mittel': 'orange', 'niedrig': 'yellow'}
legende_gesetzt = set()
for cat, color in farben.items():
    mask = (df_jul_oct['sfp_weight_cat'] == cat) & (df_jul_oct['sfp_weight'] > 0)
    if mask.any():
        label = f'SFP ({cat})' if cat not in legende_gesetzt else None
        plt.scatter(
            df_jul_oct.index[mask],
            df_jul_oct['close'][mask],
            s=df_jul_oct['sfp_weight'][mask] * 10,
            color=color, label=label, marker='v', alpha=0.7
        )
        legende_gesetzt.add(cat)
plt.title('BTCUSDT SFPs Juli 2025 - Oktober 2025')
plt.xlabel('Zeit')
plt.ylabel('Preis (USDT)')
plt.legend()
plt.tight_layout()
plt.show()