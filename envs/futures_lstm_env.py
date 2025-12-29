import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from utils.data_loader import DataLoader

class CryptoLstmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data=None, data_path=None, window_size=50, initial_balance=10_000):
        super().__init__()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.trading_fee = 0.0  # No Fees
        self.max_leverage = 10.0   # Safety Cap
        # Daten laden und Indikatoren berechnen
        if data is not None:
            self.df = data.copy()
        elif data_path is not None:
            loader = DataLoader(data_path, window_size=window_size)
            # Check if data is already preprocessed
            raw_data = loader.load_data()
            if 'trend_slope' in raw_data.columns and 'ema_50' in raw_data.columns:
                # Already preprocessed
                self.df = raw_data
            else:
                # Need preprocessing
                self.df = loader.preprocess()
        else:
            raise ValueError("Bitte entweder ein DataFrame (data) oder einen Dateipfad (data_path) angeben!")
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # Convert to NumPy arrays for speed
        self.prices = self.df['Close'].values.astype(np.float32)
        self.highs = self.df['High'].values.astype(np.float32)
        self.lows = self.df['Low'].values.astype(np.float32)
        self.close_log_returns = self.df['Close_log_return'].values.astype(np.float32)
        self.volume_log_returns = self.df['Volume_log_return'].values.astype(np.float32)
        
        # Handle optional columns safely
        n = len(self.df)
        self.dist_to_gp_long = self.df.get('dist_to_gp_long', pd.Series(np.zeros(n))).values.astype(np.float32)
        self.dist_to_gp_short = self.df.get('dist_to_gp_short', pd.Series(np.zeros(n))).values.astype(np.float32)
        self.is_sfp_long = self.df.get('is_sfp_long', pd.Series(np.zeros(n))).values.astype(bool)
        self.is_sfp_short = self.df.get('is_sfp_short', pd.Series(np.zeros(n))).values.astype(bool)
        
        self.bullish_range_low = self.df.get('bullish_range_low', pd.Series(np.zeros(n))).values.astype(np.float32)
        self.active_pivot_low = self.df.get('active_pivot_low', pd.Series(np.zeros(n))).values.astype(np.float32)
        self.bearish_range_high = self.df.get('bearish_range_high', pd.Series(np.zeros(n))).values.astype(np.float32)
        self.active_pivot_high = self.df.get('active_pivot_high', pd.Series(np.zeros(n))).values.astype(np.float32)
        
        # Trend Features for Grinding Logic
        self.trend_slope = self.df.get('trend_slope', pd.Series(np.zeros(n))).values.astype(np.float32)
        self.trend_r2 = self.df.get('trend_r2', pd.Series(np.zeros(n))).values.astype(np.float32)
        self.volatility_ratio = self.df.get('volatility_ratio', pd.Series(np.ones(n))).values.astype(np.float32)
        self.avg_range_10 = self.df.get('avg_range_10', pd.Series(np.zeros(n))).values.astype(np.float32)
        self.candle_range = self.df.get('candle_range', pd.Series(np.zeros(n))).values.astype(np.float32)
        
        # EMAs for Grind
        self.ema_50 = self.df.get('ema_50', pd.Series(np.zeros(n))).values.astype(np.float32)
        self.ema_200 = self.df.get('ema_200', pd.Series(np.zeros(n))).values.astype(np.float32)

        self._setup_spaces()
        self._reset_account()

    def _setup_spaces(self):
        # Observation Space: log_return, volume, dist_to_gp_long, dist_to_gp_short, is_sfp_long, is_sfp_short
        obs_shape = (self.window_size, 6)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    def _reset_account(self):
        self.balance = self.initial_balance
        
        # Separate Trade Slots
        self.swing_trade = {'size': 0, 'entry': 0, 'sl': 0, 'side': None, 'reason': None, 'tp_taken': False}
        self.scalp_trade = {'size': 0, 'entry': 0, 'sl': 0, 'side': None, 'reason': None, 'tp_taken': False}
        
        self.done = False
        self.current_step = self.window_size
        
        # GP Trade State Tracking
        self.last_traded_gp_long_level = -1
        self.last_traded_gp_short_level = -1
        self.last_bullish_range_high = -1
        self.last_bearish_range_low = -1
        
        # Grind Trade State Tracking (One trade per sequence)
        self.has_traded_current_grind_long = False
        self.has_traded_current_grind_short = False
        
        # SFP Trade State Tracking (One trade per pivot level)
        self.last_traded_sfp_low_level = -1
        self.last_traded_sfp_high_level = -1
        
        # Statistics
        self.long_trades_count = 0
        self.short_trades_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_account()
        
        # Random Start Logic for better Parallelization
        # Ensure we have enough data left for a meaningful episode
        min_steps_remaining = 1000 
        max_start_index = len(self.df) - min_steps_remaining
        
        if max_start_index > self.window_size:
            # Pick a random start index
            self.current_step = np.random.randint(self.window_size, max_start_index)
        else:
            self.current_step = self.window_size
            
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        
        obs = np.stack([
            self.close_log_returns[start:end],
            self.volume_log_returns[start:end],
            self.dist_to_gp_long[start:end],
            self.dist_to_gp_short[start:end],
            self.is_sfp_long[start:end].astype(np.float32),
            self.is_sfp_short[start:end].astype(np.float32)
        ], axis=1)
        return obs

    def step(self, action):
        info = {}
        reward = 0
        done = False
        
        # Access current step data from arrays
        price = self.prices[self.current_step]
        high = self.highs[self.current_step]
        low = self.lows[self.current_step]
        
        dist_gp_long = self.dist_to_gp_long[self.current_step]
        dist_gp_short = self.dist_to_gp_short[self.current_step]
        is_sfp_long_raw = self.is_sfp_long[self.current_step]
        is_sfp_short_raw = self.is_sfp_short[self.current_step]
        
        current_pivot_low = self.active_pivot_low[self.current_step]
        current_pivot_high = self.active_pivot_high[self.current_step]

        # Effective SFP Signal (Only if not traded this pivot level)
        # ENABLE SFP STRATEGY
        is_sfp_long = is_sfp_long_raw and (current_pivot_low != self.last_traded_sfp_low_level)
        is_sfp_short = is_sfp_short_raw and (current_pivot_high != self.last_traded_sfp_high_level)
        
        # --- EMA 200 FILTER (SFP Only) ---
        # Filter SFP trades against the major trend (EMA 200)
        ema_200 = self.ema_200[self.current_step]
        
        if is_sfp_short and (price > ema_200):
            is_sfp_short = False # Don't Short SFP in Bull Trend
            
        # if is_sfp_long and (price < ema_200):
        #     is_sfp_long = False # REMOVED: Allow buying deep dips below EMA 200
            
        # Grinding Trend Logic
        slope = self.trend_slope[self.current_step]
        r2 = self.trend_r2[self.current_step]
        vola_ratio = self.volatility_ratio[self.current_step]
        
        # --- Grind Logic with 10-Candle Average Confirmation ---
        is_grind_long_confirmed = True
        is_grind_short_confirmed = True
        
        slopes = []
        r2s = []
        volas = []
        
        # Check last 10 candles (including current)
        for i in range(10):
            idx = self.current_step - i
            if idx < 0:
                is_grind_long_confirmed = False
                is_grind_short_confirmed = False
                break
            
            s_slope = self.trend_slope[idx]
            s_r2 = self.trend_r2[idx]
            s_vola = self.volatility_ratio[idx]
            s_close = self.prices[idx]
            s_ema50 = self.ema_50[idx]
            
            slopes.append(s_slope)
            r2s.append(s_r2)
            volas.append(s_vola)
            
            # Hard Constraint: Price vs EMA50 must hold for ALL candles
            if not (s_close > s_ema50): is_grind_long_confirmed = False
            if not (s_close < s_ema50): is_grind_short_confirmed = False
        
        if is_grind_long_confirmed or is_grind_short_confirmed:
            avg_slope = np.mean(slopes)
            avg_r2 = np.mean(r2s)
            avg_vola = np.mean(volas)
            
            # Average Conditions
            # Slope > 0.5, R2 > 0.80, Vola < 1.5
            if is_grind_long_confirmed:
                if not ((avg_slope > 0.5) and (avg_r2 > 0.80) and (avg_vola < 1.5)):
                    is_grind_long_confirmed = False
            
            if is_grind_short_confirmed:
                if not ((avg_slope < -0.5) and (avg_r2 > 0.80) and (avg_vola < 1.5)):
                    is_grind_short_confirmed = False

        is_grind_long_raw = is_grind_long_confirmed
        is_grind_short_raw = is_grind_short_confirmed

        # Manage Grind State (Reset if Grind breaks)
        if not is_grind_long_raw:
            self.has_traded_current_grind_long = False
        if not is_grind_short_raw:
            self.has_traded_current_grind_short = False

        # Effective Signal (Only if not traded this sequence)
        is_grind_long = is_grind_long_raw and (not self.has_traded_current_grind_long)
        is_grind_short = is_grind_short_raw and (not self.has_traded_current_grind_short)

        # --- 1. Manage GP State (Reset if Zone Changes) ---
        curr_bull_high_val = self.df['bullish_range_high'].iloc[self.current_step]
        curr_bear_low_val = self.df['bearish_range_low'].iloc[self.current_step]
        
        # --- 2. Define Signal Validity for this step ---
        # DISABLE GP STRATEGY
        # is_valid_gp_long = (dist_gp_long < 0.01) and (abs(curr_bull_high_val - self.last_traded_gp_long_level) > 1e-6) and (curr_bull_high_val > 0)
        # is_valid_gp_short = (dist_gp_short < 0.01) and (abs(curr_bear_low_val - self.last_traded_gp_short_level) > 1e-6) and (curr_bear_low_val > 0)
        is_valid_gp_long = False
        is_valid_gp_short = False

        # --- 3. Entry Logic (Separated Slots) ---
        if action in [2, 3, 4, 5]:
            # Determine Side
            side = 'long' if action in [2, 3] else 'short'
            
            # --- SWING SLOT (SFP / GP) ---
            if self.swing_trade['size'] == 0:
                swing_entry_reason = None
                sl = 0
                
                if side == 'long':
                    if is_valid_gp_long:
                        sl = self.bullish_range_low[self.current_step]
                        if sl == 0: sl = price * 0.98
                        swing_entry_reason = 'gp_long'
                    elif is_sfp_long:
                        sl = low * 0.995 # SL at Signal Low - 0.5% Buffer
                        if sl == 0: sl = price * 0.98
                        swing_entry_reason = 'sfp_long'
                else: # Short
                    if is_valid_gp_short:
                        sl = self.bearish_range_high[self.current_step]
                        if sl == 0: sl = price * 1.02
                        swing_entry_reason = 'gp_short'
                    elif is_sfp_short:
                        sl = high * 1.005 # SL at Signal High + 0.5% Buffer
                        if sl == 0: sl = price * 1.02
                        swing_entry_reason = 'sfp_short'
                
                if swing_entry_reason:
                    # Execute Swing Entry
                    # DYNAMIC POSITION SIZING based on Action
                    # Action 2/4 = 1% Risk, Action 3/5 = 2% Risk
                    if action in [2, 4]:
                        risk_percent = 0.01
                    else:
                        risk_percent = 0.02
                        
                    amount = (self.balance * risk_percent) / max(abs(price - sl), 1e-6)
                    
                    # Cap Leverage
                    max_pos = self.balance * self.max_leverage
                    if amount * price > max_pos: amount = max_pos / price
                    
                    self.swing_trade = {'size': amount, 'entry': price, 'sl': sl, 'side': side, 'reason': swing_entry_reason, 'tp_taken': False}
                    
                    # Update State Trackers
                    if swing_entry_reason == 'gp_long': self.last_traded_gp_long_level = curr_bull_high_val
                    if swing_entry_reason == 'gp_short': self.last_traded_gp_short_level = curr_bear_low_val
                    if swing_entry_reason == 'sfp_long': self.last_traded_sfp_low_level = current_pivot_low
                    if swing_entry_reason == 'sfp_short': self.last_traded_sfp_high_level = current_pivot_high
                    
                    # Fee
                    fee = (amount * price) * self.trading_fee
                    self.balance -= fee
                    reward -= fee
                    
                    if side == 'long': self.long_trades_count += 1
                    else: self.short_trades_count += 1
                    
                    info['swing_entry'] = swing_entry_reason

            # --- SCALP SLOT (Grind) ---
            if self.scalp_trade['size'] == 0:
                scalp_entry_reason = None
                sl = 0
                
                if side == 'long' and is_grind_long:
                    atr = self.avg_range_10[self.current_step]
                    sl = price - (2 * atr)
                    scalp_entry_reason = 'grind_long'
                elif side == 'short' and is_grind_short:
                    atr = self.avg_range_10[self.current_step]
                    sl = price + (2 * atr)
                    scalp_entry_reason = 'grind_short'
                    
                if scalp_entry_reason:
                    # Execute Scalp Entry
                    # DYNAMIC POSITION SIZING based on Action
                    # Action 2/4 = 0.5% Risk, Action 3/5 = 1.0% Risk (Scalps are smaller)
                    if action in [2, 4]:
                        risk_percent = 0.005
                    else:
                        risk_percent = 0.01

                    amount = (self.balance * risk_percent) / max(abs(price - sl), 1e-6)
                    
                    max_pos = self.balance * self.max_leverage
                    if amount * price > max_pos: amount = max_pos / price
                    
                    self.scalp_trade = {'size': amount, 'entry': price, 'sl': sl, 'initial_sl': sl, 'side': side, 'reason': scalp_entry_reason, 'tp_taken': False, 'trailing_active': False, 'banked_pnl': 0}
                    
                    if scalp_entry_reason == 'grind_long': self.has_traded_current_grind_long = True
                    if scalp_entry_reason == 'grind_short': self.has_traded_current_grind_short = True
                    
                    fee = (amount * price) * self.trading_fee
                    self.balance -= fee
                    reward -= fee
                    
                    if side == 'long': self.long_trades_count += 1
                    else: self.short_trades_count += 1
                    
                    info['scalp_entry'] = scalp_entry_reason

        # --- Exit Logic (Separated) ---
        
        # 1. Manual Close (Action 1) -> Closes SCALP ONLY
        if action == 1 and self.scalp_trade['size'] > 0:
            pnl = self._close_position('scalp', price)
            reward += pnl
            info['scalp_manual_close'] = True
            
        # 2. Check Swing Exits (SL or Reversal)
        if self.swing_trade['size'] > 0:
            st = self.swing_trade
            
            # SL Hit
            if (st['side'] == 'long' and low < st['sl']) or (st['side'] == 'short' and high > st['sl']):
                pnl = self._close_position('swing', st['sl'])
                reward += pnl
                info['swing_sl_hit'] = True
            # Reversal Signal (Opposite Swing Signal)
            elif (st['side'] == 'long' and (is_sfp_short or is_valid_gp_short)) or \
                 (st['side'] == 'short' and (is_sfp_long or is_valid_gp_long)):
                pnl = self._close_position('swing', price)
                reward += pnl
                info['swing_reversal_exit'] = True
                
        # 3. Check Scalp Exits (SL or Trend Break)
        if self.scalp_trade['size'] > 0:
            st = self.scalp_trade
            
            # Hybrid: Partial TP (50% at 2R) + Activate Trailing SL
            if not st['tp_taken']:
                initial_sl = st.get('initial_sl', st['sl'])
                risk = abs(st['entry'] - initial_sl)
                
                if (st['side'] == 'long' and high >= st['entry'] + (2 * risk)) or \
                   (st['side'] == 'short' and low <= st['entry'] - (2 * risk)):
                    tp_price = st['entry'] + (2 * risk) if st['side'] == 'long' else st['entry'] - (2 * risk)
                    
                    # 1. Take Partial Profit (50%)
                    pnl = self._take_partial_profit('scalp', tp_price)
                    reward += pnl
                    info['scalp_partial_tp'] = True
                    
                    # 2. Activate Trailing SL (Tight: 1.0 ATR)
                    st['trailing_active'] = True
                    atr = self.avg_range_10[self.current_step]
                    if st['side'] == 'long':
                        st['sl'] = price - (1.0 * atr)
                    else:
                        st['sl'] = price + (1.0 * atr)

            # Update Trailing SL if Active
            if st.get('trailing_active', False):
                atr = self.avg_range_10[self.current_step]
                if st['side'] == 'long':
                    new_sl = price - (1.0 * atr)
                    if new_sl > st['sl']: st['sl'] = new_sl
                else:
                    new_sl = price + (1.0 * atr)
                    if new_sl < st['sl']: st['sl'] = new_sl

            # SL Hit
            if (st['side'] == 'long' and low < st['sl']) or (st['side'] == 'short' and high > st['sl']):
                pnl = self._close_position('scalp', st['sl'])
                reward += pnl
                info['scalp_sl_hit'] = True
            # Grind Specific Exits
            else:
                should_close = False
                # A. Trend Stop - ENABLED (Match Validation Logic)
                slope = self.trend_slope[self.current_step]
                r2 = self.trend_r2[self.current_step]
                
                if st['reason'] == 'grind_long' and (slope < -0.1 or r2 < 0.50): should_close = True
                if st['reason'] == 'grind_short' and (slope > 0.1 or r2 < 0.50): should_close = True
                
                # B. Impulse (High Volatility Spike) - DISABLED (Let AI learn)
                # current_range = self.candle_range[self.current_step]
                # avg_range = self.avg_range_10[self.current_step] # Fixed to avg_range_10
                # if current_range > 3 * avg_range: should_close = True
                
                if should_close:
                    pnl = self._close_position('scalp', price)
                    reward += pnl
                    info['scalp_trend_exit'] = True

        # --- Reward Shaping (Simplified) ---
        # Reward for holding profitable positions? Or just PnL?
        # Let's keep it simple: PnL is the main driver.
        # Maybe small reward for entering valid signals
        if action in [2, 3, 4, 5]:
            if 'swing_entry' in info: reward += 5
            if 'scalp_entry' in info: reward += 2

        # --- Step Forward ---
        self.current_step += 1
        if self.current_step >= len(self.prices) - 1 or self.balance <= 0:
            done = True
            self.done = True
            info['long_trades'] = self.long_trades_count
            info['short_trades'] = self.short_trades_count
            
        obs = self._get_observation()
        return obs, reward, done, False, info

    def _take_partial_profit(self, trade_type, current_price):
        if trade_type == 'swing':
            trade = self.swing_trade
        else:
            trade = self.scalp_trade
            
        if trade['size'] == 0 or trade.get('tp_taken', False): return 0

        # Close 50%
        close_size = trade['size'] * 0.5
        
        if trade['side'] == 'long':
            pnl = (current_price - trade['entry']) * close_size
        else:
            pnl = (trade['entry'] - current_price) * close_size
            
        # Fee
        exit_fee = (close_size * current_price) * self.trading_fee
        self.balance -= exit_fee
        
        realized_pnl = pnl - exit_fee
        self.balance += realized_pnl
        
        # Update Trade
        trade['size'] -= close_size
        trade['tp_taken'] = True
        
        return realized_pnl

    def _close_position(self, trade_type, exit_price):
        if trade_type == 'swing':
            trade = self.swing_trade
        else:
            trade = self.scalp_trade
            
        if trade['size'] == 0: return 0
        
        if trade['side'] == 'long':
            pnl = (exit_price - trade['entry']) * trade['size']
        else:
            pnl = (trade['entry'] - exit_price) * trade['size']
            
        # Apply Exit Fee
        exit_fee = (trade['size'] * exit_price) * self.trading_fee
        self.balance -= exit_fee
        
        realized_pnl = pnl - exit_fee
        self.balance += pnl
        
        # Reset Trade Slot
        if trade_type == 'swing':
            self.swing_trade = {'size': 0, 'entry': 0, 'sl': 0, 'side': None, 'reason': None, 'tp_taken': False}
        else:
            self.scalp_trade = {'size': 0, 'entry': 0, 'sl': 0, 'initial_sl': 0, 'side': None, 'reason': None, 'tp_taken': False, 'trailing_active': False, 'banked_pnl': 0}
            
        return realized_pnl

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Swing: {self.swing_trade['side']} | Scalp: {self.scalp_trade['side']}")

    def close(self):
        pass