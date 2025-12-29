import time
import ccxt
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from utils.data_loader import DataLoader
import logging
from datetime import datetime
import os
import csv

# --- CONFIGURATION ---
API_KEY = 'YOUR_BINANCE_API_KEY'
API_SECRET = 'YOUR_BINANCE_SECRET'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '5m'
MODEL_PATH = "models/ppo_lstm_parallel_1765916372" # Updated Model
WINDOW_SIZE = 50
LEVERAGE = 1
RISK_PER_TRADE = 0.01 # 1%
DRY_RUN = True # Paper Trading Mode

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class LiveTrader:
    def __init__(self):
        # 1. Connect to Exchange
        try:
            config = {
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            }
            # Only use keys if they are not placeholders
            if API_KEY != 'YOUR_BINANCE_API_KEY':
                config['apiKey'] = API_KEY
                config['secret'] = API_SECRET
                
            self.exchange = ccxt.binanceusdm(config)
            # self.exchange.load_markets() # Optional for Dry Run, good for real
            logger.info("Connected to Binance Futures (Dry Run: " + str(DRY_RUN) + ")")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            exit()

        # 2. Load Model
        model_file = MODEL_PATH + ".zip"
        if not os.path.exists(model_file):
            logger.error(f"Model not found at {model_file}")
            exit()
            
        self.model = RecurrentPPO.load(MODEL_PATH)
        self.lstm_states = None # LSTM Memory
        self.last_dones = np.ones(1) 
        logger.info(f"Model loaded successfully: {MODEL_PATH}")

        # 3. Initialize Data Loader
        self.loader = DataLoader(None) 

        # 4. State Management (Mirroring Environment)
        self.balance = 10000 # Simulated Balance
        self.scalp_trade = {'size': 0, 'entry': 0, 'sl': 0, 'side': None, 'reason': None, 'tp_taken': False}
        self.swing_trade = {'size': 0, 'entry': 0, 'sl': 0, 'side': None, 'reason': None, 'tp_taken': False}
        
        # Initialize Trade Log
        self.log_file = "live_trades.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Type', 'Side', 'Entry', 'Exit', 'Size', 'PnL', 'Reason', 'Balance'])

    def fetch_live_data(self):
        try:
            # Fetch enough candles for indicators (200 is safe for EMA200)
            ohlcv = self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=300)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None

    def prepare_features(self, df):
        try:
            # 1. Log Returns
            df = self.loader.calculate_log_returns(df)
            
            # 2. Indicators (Grind, SFP, GP, Trend Slope/R2)
            df = self.loader.calculate_trend_indicators(df)
            
            # 3. Select Features for Model
            # [Close_log_return, Volume_log_return, dist_gp_long, dist_gp_short, is_sfp_long, is_sfp_short]
            last_window = df.iloc[-WINDOW_SIZE:].copy()
            
            obs_data = np.column_stack([
                last_window['Close_log_return'].values,
                last_window['Volume_log_return'].values,
                last_window['dist_to_gp_long'].values,
                last_window['dist_to_gp_short'].values,
                last_window['is_sfp_long'].values.astype(float),
                last_window['is_sfp_short'].values.astype(float)
            ])
            
            return obs_data.astype(np.float32), df
            
        except Exception as e:
            logger.error(f"Feature calculation error: {e}")
            return None, None

    def check_trend_stop(self, df):
        """
        Checks if the current trend is weak (Trend Stop Logic).
        Returns True if position should be closed.
        """
        if self.scalp_trade['size'] == 0:
            return False
            
        current_step = df.iloc[-1]
        slope = current_step['trend_slope']
        r2 = current_step['trend_r2']
        reason = self.scalp_trade['reason']
        
        should_close = False
        
        # Logic from futures_lstm_env.py
        if reason == 'grind_long' and (slope < -0.1 or r2 < 0.50):
            should_close = True
            logger.info(f"Trend Stop Triggered (Long): Slope={slope:.4f}, R2={r2:.4f}")
            
        if reason == 'grind_short' and (slope > 0.1 or r2 < 0.50):
            should_close = True
            logger.info(f"Trend Stop Triggered (Short): Slope={slope:.4f}, R2={r2:.4f}")
            
        return should_close

    def log_trade(self, trade_type, side, entry, exit_price, size, pnl, reason):
        self.balance += pnl
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), trade_type, side, entry, exit_price, size, pnl, reason, self.balance])
        logger.info(f"Trade Closed: {side} {trade_type} | PnL: {pnl:.2f} | New Balance: {self.balance:.2f}")

    def execute_trade(self, action, current_price, df):
        """
        Maps model action to execution logic.
        """
        # --- 1. Handle Exits (Action 1 or Trend Stop handled in run loop) ---
        if action == 1:
            if self.scalp_trade['size'] > 0:
                self._close_position('scalp', current_price, 'Model Signal')
            # Note: Model action 1 usually closes Scalp in the Env logic. 
            # Swing trades are managed by SL/Reversal usually, but let's stick to Env logic.
            
        # --- 2. Handle Entries ---
        # Logic from Env:
        # Action 2: Long (Full)
        # Action 3: Long (Half)
        # Action 4: Short (Full)
        # Action 5: Short (Half)
        
        # We need to determine if it's a Scalp (Grind) or Swing (SFP/GP) entry.
        # The Env uses internal state flags (is_grind_long, is_sfp_long, etc.) to decide WHICH trade to open.
        # We need to replicate that check here.
        
        current_row = df.iloc[-1]
        
        # Re-derive signals (simplified from Env logic)
        # Note: In Env, these are pre-calculated. Here we check the latest row.
        
        # Grind Signals
        is_grind_long = (current_row['trend_slope'] > 0.1) and (current_row['trend_r2'] > 0.85)
        is_grind_short = (current_row['trend_slope'] < -0.1) and (current_row['trend_r2'] > 0.85)
        
        # SFP Signals
        is_sfp_long = current_row['is_sfp_long']
        is_sfp_short = current_row['is_sfp_short']
        
        # GP Signals (Distance < Threshold)
        # Assuming threshold is small, e.g., 0.001 or just check if dist == 0 (inside zone)
        # In Env: is_valid_gp_long = (dist_gp_long == 0)
        is_gp_long = (current_row['dist_to_gp_long'] == 0)
        is_gp_short = (current_row['dist_to_gp_short'] == 0)
        
        # Priority: Swing (SFP/GP) > Scalp (Grind)
        
        # --- Swing Entry ---
        if self.swing_trade['size'] == 0:
            swing_reason = None
            side = None
            
            if action in [2, 3]: # Long
                if is_sfp_long: swing_reason = 'sfp_long'; side = 'long'
                elif is_gp_long: swing_reason = 'gp_long'; side = 'long'
            elif action in [4, 5]: # Short
                if is_sfp_short: swing_reason = 'sfp_short'; side = 'short'
                elif is_gp_short: swing_reason = 'gp_short'; side = 'short'
                
            if swing_reason:
                self._open_position('swing', side, current_price, action, swing_reason)
                return # Prioritize one trade per step

        # --- Scalp Entry ---
        if self.scalp_trade['size'] == 0:
            scalp_reason = None
            side = None
            
            if action in [2, 3]: # Long
                if is_grind_long: scalp_reason = 'grind_long'; side = 'long'
            elif action in [4, 5]: # Short
                if is_grind_short: scalp_reason = 'grind_short'; side = 'short'
                
            if scalp_reason:
                self._open_position('scalp', side, current_price, action, scalp_reason)

    def _open_position(self, trade_type, side, price, action, reason):
        # Size Calculation
        risk_percent = 0.01
        if trade_type == 'scalp':
            risk_percent = 0.005 if action in [2, 4] else 0.01
        else:
            risk_percent = 0.01 if action in [2, 4] else 0.02
            
        # Simulated SL (Simplified)
        # In real env, SL is calculated based on ATR or Structure.
        # For Dry Run, let's assume a fixed % SL for simplicity or try to calculate ATR.
        # Let's use 1% SL for now to keep it simple, or fetch ATR if possible.
        sl_dist = price * 0.01 
        sl = price - sl_dist if side == 'long' else price + sl_dist
        
        size = (self.balance * risk_percent) / sl_dist
        
        trade_dict = self.scalp_trade if trade_type == 'scalp' else self.swing_trade
        trade_dict.update({
            'size': size,
            'entry': price,
            'sl': sl,
            'side': side,
            'reason': reason,
            'tp_taken': False
        })
        
        logger.info(f"OPEN {trade_type.upper()} {side.upper()} | Price: {price} | Size: {size:.4f} | Reason: {reason}")

    def _close_position(self, trade_type, price, reason):
        trade = self.scalp_trade if trade_type == 'scalp' else self.swing_trade
        if trade['size'] == 0: return
        
        pnl = (price - trade['entry']) * trade['size'] if trade['side'] == 'long' else (trade['entry'] - price) * trade['size']
        
        self.log_trade(trade_type, trade['side'], trade['entry'], price, trade['size'], pnl, reason)
        
        # Reset
        trade.update({'size': 0, 'entry': 0, 'sl': 0, 'side': None, 'reason': None})

    def run(self):
        logger.info(f"Starting Live Trader for {SYMBOL} on {TIMEFRAME}")
        logger.info(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}")
        
        while True:
            # 1. Wait for Candle Close
            now = time.time()
            next_candle_time = (int(now) // 300 + 1) * 300
            sleep_seconds = next_candle_time - now + 5 # +5s buffer
            
            logger.info(f"Waiting {sleep_seconds:.0f}s for candle close...")
            time.sleep(sleep_seconds)
            
            # 2. Fetch & Process Data
            df = self.fetch_live_data()
            if df is None: continue
            
            obs, df_processed = self.prepare_features(df)
            if obs is None: continue
            
            current_price = df['Close'].iloc[-1]
            
            # 3. Check Trend Stop (Crucial Fix)
            if self.check_trend_stop(df_processed):
                self._close_position('scalp', current_price, 'Trend Stop')
                # We can still predict, but we just closed the scalp.
            
            # Reshape for model
            obs = obs.reshape(1, WINDOW_SIZE, -1)
            
            # 4. Predict
            action, self.lstm_states = self.model.predict(
                obs,
                state=self.lstm_states,
                episode_start=self.last_dones,
                deterministic=True
            )
            self.last_dones = np.zeros(1)
            
            # 5. Execute
            self.execute_trade(action[0], current_price, df_processed)

if __name__ == "__main__":
    trader = LiveTrader()
    trader.run()
