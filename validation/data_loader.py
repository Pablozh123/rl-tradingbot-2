import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.signal import argrelextrema

class DataLoader:
    def __init__(self, file_path, window_size=50):
        """
        Initialize the DataLoader.

        :param file_path: Path to the historical data file (CSV).
        :param window_size: Rolling window size for trend calculations.
        """
        self.file_path = file_path
        self.window_size = window_size

    def load_data(self):
        """
        Load historical data from a CSV file.

        :return: DataFrame with historical data.
        """
        if self.file_path.endswith('.csv'):
            data = pd.read_csv(self.file_path)
            # Ensure timestamp is datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            elif 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
        else:
            raise ValueError("Unsupported file format. Use CSV.")

        # Ensure numeric columns
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data.dropna()

    def calculate_log_returns(self, data):
        """
        Calculate log-returns for OHLCV columns.

        :param data: DataFrame with historical data.
        :return: DataFrame with log-returns normalized to [-1, 1].
        """
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[f'{col}_log_return'] = np.log(data[col] / data[col].shift(1))
                # Normalize to [-1, 1] (simple scaling, can be improved)
                max_val = data[f'{col}_log_return'].abs().max()
                if max_val > 0:
                    data[f'{col}_log_return'] = data[f'{col}_log_return'] / max_val
                else:
                    data[f'{col}_log_return'] = 0

        return data

    def _calculate_4h_structure(self, data):
        """
        Resample to 4H, calculate Pivots and Golden Pockets.
        """
        # Resample to 4H
        df_4h = data.resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # --- 4H Pivot Definition ---
        ORDER = 3
        # Find Pivots
        high_idx = argrelextrema(df_4h['High'].values, np.greater_equal, order=ORDER)[0]
        low_idx = argrelextrema(df_4h['Low'].values, np.less_equal, order=ORDER)[0]

        df_4h['is_pivot_high'] = False
        df_4h['is_pivot_low'] = False
        df_4h.iloc[high_idx, df_4h.columns.get_loc('is_pivot_high')] = True
        df_4h.iloc[low_idx, df_4h.columns.get_loc('is_pivot_low')] = True

        df_4h['pivot_high'] = np.where(df_4h['is_pivot_high'], df_4h['High'], np.nan)
        df_4h['pivot_low'] = np.where(df_4h['is_pivot_low'], df_4h['Low'], np.nan)

        # Active Pivots (ffill)
        df_4h['active_pivot_high'] = df_4h['pivot_high'].ffill()
        df_4h['active_pivot_low'] = df_4h['pivot_low'].ffill()

        # --- Golden Pocket Calculation ---
        df_4h['bullish_range_low'] = np.nan
        df_4h['bullish_range_high'] = np.nan
        df_4h['bearish_range_low'] = np.nan
        df_4h['bearish_range_high'] = np.nan

        for i in range(1, len(df_4h)):
            # Bullish Range Logic (Low -> High)
            if not pd.isna(df_4h['active_pivot_low'].iloc[i]):
                low = df_4h['active_pivot_low'].iloc[i]
                high = df_4h['High'].iloc[i] 
                if high > low:
                    range_size = high - low
                    # Retracement from High down to 0.618-0.67
                    df_4h.at[df_4h.index[i], 'bullish_range_low'] = high - (range_size * 0.67)
                    df_4h.at[df_4h.index[i], 'bullish_range_high'] = high - (range_size * 0.618)

            # Bearish Range Logic (High -> Low)
            if not pd.isna(df_4h['active_pivot_high'].iloc[i]):
                high = df_4h['active_pivot_high'].iloc[i]
                low = df_4h['Low'].iloc[i]
                if low < high:
                    range_size = high - low
                    # Retracement from Low up to 0.618-0.67
                    df_4h.at[df_4h.index[i], 'bearish_range_high'] = low + (range_size * 0.67)
                    df_4h.at[df_4h.index[i], 'bearish_range_low'] = low + (range_size * 0.618)
        
        return df_4h[['active_pivot_high', 'active_pivot_low', 
                      'bullish_range_low', 'bullish_range_high', 
                      'bearish_range_low', 'bearish_range_high']]

    def calculate_trend_indicators(self, data):
        """
        Calculate trend indicators (Grind) and merge 4H structure (SFP, GP).
        """
        # 1. Grind / Trend Slope & R2
        # Using a simple loop or rolling apply for R2 (can be slow, but accurate)
        def get_slope_r2(y):
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            return slope, r_value**2

        # Rolling apply is slow for complex functions, we can optimize or use it as is for preprocessing
        # For speed, we might just calculate slope simply, but R2 needs linregress
        # Let's use a simplified approach or the one from signal_validation if available.
        # Here we stick to the requested logic:
        
        # Pre-calculate rolling windows to avoid re-creating them
        closes = data['Close'].values
        window = 20
        
        slopes = np.full(len(data), np.nan)
        r2s = np.full(len(data), np.nan)
        
        # Optimization: Vectorized approach is hard for R2, using iteration for clarity/correctness
        # Since this is preprocessing (run once), speed is acceptable.
        for i in range(window, len(data)):
            y = closes[i-window:i]
            x = np.arange(window)
            slope, intercept, r_value, _, _ = linregress(x, y)
            slopes[i] = slope
            r2s[i] = r_value**2
            
        data['trend_slope'] = slopes
        data['trend_r2'] = r2s

        # 2. Multi-Timeframe Merge
        df_4h_structure = self._calculate_4h_structure(data.copy())
        
        # Merge 4H levels to original data (backward merge to avoid lookahead bias)
        data = data.sort_index()
        df_4h_structure = df_4h_structure.sort_index()
        
        data = pd.merge_asof(
            data,
            df_4h_structure,
            left_index=True,
            right_index=True,
            direction='backward'
        )

        # 3. Calculate Signals based on merged data
        
        # SFP Signals
        # Bearish SFP: Price breaks 4H-High but closes below it
        data['is_sfp_short'] = (data['High'] > data['active_pivot_high']) & (data['Close'] < data['active_pivot_high'])
        # Bullish SFP: Price breaks 4H-Low but closes above it
        data['is_sfp_long'] = (data['Low'] < data['active_pivot_low']) & (data['Close'] > data['active_pivot_low'])

        # Golden Pocket Distances
        # Distance to GP Long Zone
        data['dist_to_gp_long'] = np.where(
            (data['Low'] <= data['bullish_range_high']) & (data['Low'] >= data['bullish_range_low']),
            0, # Inside zone
            np.minimum(
                np.abs(data['Low'] - data['bullish_range_high']),
                np.abs(data['Low'] - data['bullish_range_low'])
            )
        )
        # Normalize distance roughly (e.g. by price)
        data['dist_to_gp_long'] = data['dist_to_gp_long'] / data['Close']

        # Distance to GP Short Zone
        data['dist_to_gp_short'] = np.where(
            (data['High'] >= data['bearish_range_low']) & (data['High'] <= data['bearish_range_high']),
            0, # Inside zone
            np.minimum(
                np.abs(data['High'] - data['bearish_range_low']),
                np.abs(data['High'] - data['bearish_range_high'])
            )
        )
        data['dist_to_gp_short'] = data['dist_to_gp_short'] / data['Close']

        # Fill NaNs
        data.fillna(0, inplace=True)
        
        return data

    def preprocess(self):
        """
        Full preprocessing pipeline: load data, calculate features, and indicators.

        :return: Preprocessed DataFrame.
        """
        data = self.load_data()
        data = self.calculate_log_returns(data)
        data = self.calculate_trend_indicators(data)
        return data