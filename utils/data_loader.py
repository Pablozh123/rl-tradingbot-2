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
        Resample to 4H for SFP and 1D for GP.
        """
        # Resample to 4H
        df_4h = data.resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # --- 4H Pivot Definition (Minor - for SFP) ---
        ORDER_MINOR = 3
        df_4h['is_pivot_high_minor'] = False
        df_4h['is_pivot_low_minor'] = False
        
        high_idx_minor = argrelextrema(df_4h['High'].values, np.greater_equal, order=ORDER_MINOR)[0]
        low_idx_minor = argrelextrema(df_4h['Low'].values, np.less_equal, order=ORDER_MINOR)[0]
        
        df_4h.iloc[high_idx_minor, df_4h.columns.get_loc('is_pivot_high_minor')] = True
        df_4h.iloc[low_idx_minor, df_4h.columns.get_loc('is_pivot_low_minor')] = True
        
        df_4h['pivot_high_minor'] = np.where(df_4h['is_pivot_high_minor'], df_4h['High'], np.nan)
        df_4h['pivot_low_minor'] = np.where(df_4h['is_pivot_low_minor'], df_4h['Low'], np.nan)

        # Active Minor Pivots
        df_4h['active_pivot_high'] = df_4h['pivot_high_minor'].shift(ORDER_MINOR).ffill()
        df_4h['active_pivot_low'] = df_4h['pivot_low_minor'].shift(ORDER_MINOR).ffill()
        
        # --- 4H Major Structure (Restored for SFP) ---
        ORDER_MAJOR_4H = 6
        df_4h['is_pivot_high_major_4h'] = False
        df_4h['is_pivot_low_major_4h'] = False
        
        high_idx_major = argrelextrema(df_4h['High'].values, np.greater_equal, order=ORDER_MAJOR_4H)[0]
        low_idx_major = argrelextrema(df_4h['Low'].values, np.less_equal, order=ORDER_MAJOR_4H)[0]
        
        df_4h.iloc[high_idx_major, df_4h.columns.get_loc('is_pivot_high_major_4h')] = True
        df_4h.iloc[low_idx_major, df_4h.columns.get_loc('is_pivot_low_major_4h')] = True
        
        df_4h['pivot_high_major_4h'] = np.where(df_4h['is_pivot_high_major_4h'], df_4h['High'], np.nan)
        df_4h['pivot_low_major_4h'] = np.where(df_4h['is_pivot_low_major_4h'], df_4h['Low'], np.nan)
        
        df_4h['active_pivot_high_major_4h'] = df_4h['pivot_high_major_4h'].shift(ORDER_MAJOR_4H).ffill()
        df_4h['active_pivot_low_major_4h'] = df_4h['pivot_low_major_4h'].shift(ORDER_MAJOR_4H).ffill()

        # Track indices for 4H Major
        df_4h['pivot_high_major_4h_idx'] = np.where(df_4h['is_pivot_high_major_4h'], df_4h.index, pd.NaT)
        df_4h['pivot_low_major_4h_idx'] = np.where(df_4h['is_pivot_low_major_4h'], df_4h.index, pd.NaT)
        
        df_4h['active_pivot_high_major_4h_idx'] = df_4h['pivot_high_major_4h_idx'].shift(ORDER_MAJOR_4H).ffill()
        df_4h['active_pivot_low_major_4h_idx'] = df_4h['pivot_low_major_4h_idx'].shift(ORDER_MAJOR_4H).ffill()

        # --- GP Calculation on 4H Major with 3% Rule ---
        df_4h['bullish_range_low'] = np.nan
        df_4h['bullish_range_high'] = np.nan
        df_4h['bearish_range_low'] = np.nan
        df_4h['bearish_range_high'] = np.nan

        # Calculate Ranges
        range_bullish = df_4h['active_pivot_high_major_4h'] - df_4h['active_pivot_low_major_4h']
        range_bearish = df_4h['active_pivot_high_major_4h'] - df_4h['active_pivot_low_major_4h']

        # 3% Rule
        # Bullish GP: Pump (Low->High) > 3%
        is_valid_bullish_move = (range_bullish / df_4h['active_pivot_low_major_4h']) > 0.03
        
        # Bearish GP: Dump (High->Low) > 3%
        is_valid_bearish_move = (range_bearish / df_4h['active_pivot_high_major_4h']) > 0.03

        # Apply Mask
        mask_bullish = (df_4h['active_pivot_high_major_4h_idx'] > df_4h['active_pivot_low_major_4h_idx']) & is_valid_bullish_move
        mask_bearish = (df_4h['active_pivot_low_major_4h_idx'] > df_4h['active_pivot_high_major_4h_idx']) & is_valid_bearish_move

        # Bullish GP
        df_4h.loc[mask_bullish, 'bullish_range_high'] = df_4h.loc[mask_bullish, 'active_pivot_high_major_4h'] - (range_bullish[mask_bullish] * 0.618)
        df_4h.loc[mask_bullish, 'bullish_range_low'] = df_4h.loc[mask_bullish, 'active_pivot_high_major_4h'] - (range_bullish[mask_bullish] * 0.67)

        # Bearish GP
        df_4h.loc[mask_bearish, 'bearish_range_low'] = df_4h.loc[mask_bearish, 'active_pivot_low_major_4h'] + (range_bearish[mask_bearish] * 0.618)
        df_4h.loc[mask_bearish, 'bearish_range_high'] = df_4h.loc[mask_bearish, 'active_pivot_low_major_4h'] + (range_bearish[mask_bearish] * 0.67)
        
        # --- Daily Candles (PDH/PDL) ---
        df_daily = data.resample('1D').agg({'High': 'max', 'Low': 'min'}).dropna()
        df_daily['prev_day_high'] = df_daily['High'].shift(1)
        df_daily['prev_day_low'] = df_daily['Low'].shift(1)
        
        # Merge Daily back to 4H
        df_4h = pd.merge_asof(df_4h, df_daily[['prev_day_high', 'prev_day_low']], left_index=True, right_index=True, direction='backward')

        return df_4h[['active_pivot_high', 'active_pivot_low', 
                      'active_pivot_high_major_4h', 'active_pivot_low_major_4h',
                      'bullish_range_low', 'bullish_range_high', 
                      'bearish_range_low', 'bearish_range_high',
                      'prev_day_high', 'prev_day_low']]

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
        # Reduced window from 10 to 6 for earlier detection
        window = 6
        
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

        # Add EMAs for Grind Strategy
        data['ema_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['ema_200'] = data['Close'].ewm(span=200, adjust=False).mean()

        # Volatility Metrics (for Low Vola Grind)
        data['candle_range'] = data['High'] - data['Low']
        data['avg_range_10'] = data['candle_range'].rolling(window=10).mean()
        data['avg_range_100'] = data['candle_range'].rolling(window=100).mean()
        # Ratio < 1.0 means current volatility is lower than long-term average
        data['volatility_ratio'] = data['avg_range_10'] / data['avg_range_100']

        # 2. Multi-Timeframe Merge
        df_4h_structure = self._calculate_4h_structure(data.copy())
        
        # Merge 4H levels to original data (backward merge to avoid lookahead bias)
        data = data.sort_index()
        df_4h_structure = df_4h_structure.sort_index()
        
        # Drop columns that will be merged to avoid _x/_y suffixes
        cols_to_merge = df_4h_structure.columns
        data.drop(columns=[c for c in cols_to_merge if c in data.columns], inplace=True)
        
        data = pd.merge_asof(
            data,
            df_4h_structure,
            left_index=True,
            right_index=True,
            direction='backward'
        )

        # 3. Calculate Signals based on merged data
        
        # SFP Signals with Wick Filter (0.05% Sweep) AND Close Position Filter (Rejection)
        SWEEP_THRESHOLD = 0.0005
        CLOSE_POS_THRESHOLD = 0.5 # Close must be in the top/bottom 50% of the candle

        # Calculate Candle Range for Close Position
        candle_range = data['High'] - data['Low']
        # Avoid division by zero
        candle_range = candle_range.replace(0, 1e-6)
        
        close_pos = (data['Close'] - data['Low']) / candle_range

        # Bearish SFP: Price breaks 4H-High (Minor OR Major) but closes below it
        # We check if High > Pivot AND Close < Pivot AND Sweep Depth > Threshold AND Close in lower 50%
        sweep_short_minor = (data['High'] - data['active_pivot_high']) / data['Close']
        data['is_sfp_short_minor'] = (data['High'] > data['active_pivot_high']) & \
                                     (data['Close'] < data['active_pivot_high']) & \
                                     (sweep_short_minor > SWEEP_THRESHOLD) & \
                                     (close_pos < (1 - CLOSE_POS_THRESHOLD))
                                     
        sweep_short_major = (data['High'] - data['active_pivot_high_major_4h']) / data['Close']
        data['is_sfp_short_major'] = (data['High'] > data['active_pivot_high_major_4h']) & \
                                     (data['Close'] < data['active_pivot_high_major_4h']) & \
                                     (sweep_short_major > SWEEP_THRESHOLD) & \
                                     (close_pos < (1 - CLOSE_POS_THRESHOLD))
                                     
        data['is_sfp_short'] = data['is_sfp_short_minor'] | data['is_sfp_short_major']

        # Bullish SFP: Price breaks 4H-Low (Minor OR Major) but closes above it
        # Close in upper 50%
        sweep_long_minor = (data['active_pivot_low'] - data['Low']) / data['Close']
        data['is_sfp_long_minor'] = (data['Low'] < data['active_pivot_low']) & \
                                    (data['Close'] > data['active_pivot_low']) & \
                                    (sweep_long_minor > SWEEP_THRESHOLD) & \
                                    (close_pos > CLOSE_POS_THRESHOLD)
                                    
        sweep_long_major = (data['active_pivot_low_major_4h'] - data['Low']) / data['Close']
        data['is_sfp_long_major'] = (data['Low'] < data['active_pivot_low_major_4h']) & \
                                    (data['Close'] > data['active_pivot_low_major_4h']) & \
                                    (sweep_long_major > SWEEP_THRESHOLD) & \
                                    (close_pos > CLOSE_POS_THRESHOLD)
                                    
        data['is_sfp_long'] = data['is_sfp_long_minor'] | data['is_sfp_long_major']

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