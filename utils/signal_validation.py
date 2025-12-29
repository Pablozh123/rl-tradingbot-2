import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Load Data
DATA_PATH = 'data/preprocessed_2022.csv'

def validate_signals():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Ensure columns exist
    required_cols = ['trend_slope', 'trend_r2', 'volatility_ratio', 'Close', 'Date', 'is_sfp_long', 'is_sfp_short', 'dist_to_gp_long', 'dist_to_gp_short', 'avg_range_10', 'bullish_range_low', 'bearish_range_high', 'active_pivot_low', 'active_pivot_high', 'active_pivot_high_major_4h', 'active_pivot_low_major_4h', 'ema_50', 'prev_day_high', 'prev_day_low']
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing column: {col}")
            if col == 'Date' and 'timestamp' in df.columns:
                df['Date'] = df['timestamp']
            else:
                return

    # --- FILTER FOR FULL YEAR 2022 ---
    df['Date'] = pd.to_datetime(df['Date'])
    start_date = '2022-01-01'
    end_date = '2022-12-31'
    print(f"Filtering data from {start_date} to {end_date}...")
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)
    print(f"Data filtered. Rows: {len(df)}")
    # -------------------------------

    print("Data loaded. Simulating Trades (Swing vs Scalp)...")
    
    # Signal Lists (for plotting markers)
    grind_longs = []
    grind_shorts = []
    sfp_longs = []
    sfp_shorts = []
    gp_longs = []
    gp_shorts = []
    
    sfp_longs_major = []
    sfp_shorts_major = []
    sfp_longs_minor = []
    sfp_shorts_minor = []

    # State variables for Signal Detection
    last_bullish_range_high = -1
    last_bearish_range_low = -1
    has_traded_current_grind_long = False
    has_traded_current_grind_short = False
    last_traded_sfp_low_level = -1
    last_traded_sfp_high_level = -1
    
    # Pending GP State
    pending_gp_long = None # { 'active': True, 'start_idx': i, 'zone_top': ..., 'zone_bottom': ..., 'sl': ... }
    pending_gp_short = None
    
    # --- Simulation State ---
    swing_trade = {'size': 0, 'entry': 0, 'sl': 0, 'side': None, 'reason': None, 'start_idx': 0, 'tp_taken': False, 'breakeven_active': False}
    scalp_trade = {'size': 0, 'entry': 0, 'sl': 0, 'initial_sl': 0, 'side': None, 'reason': None, 'start_idx': 0, 'trailing_active': False, 'tp_taken': False}
    
    executed_trades = [] # List of dicts

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        open_price = df['Open'].iloc[i] if 'Open' in df.columns else price # Fallback
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        
        slope = df['trend_slope'].iloc[i]
        r2 = df['trend_r2'].iloc[i]
        vola = df['volatility_ratio'].iloc[i]
        avg_range = df['avg_range_10'].iloc[i]
        candle_range = high - low
        
        # Grind Definition with 10-Candle Average Confirmation
        is_grind_long_confirmed = True
        is_grind_short_confirmed = True
        
        slopes = []
        r2s = []
        volas = []
        
        for j in range(10):
            idx = i - j
            if idx < 0:
                is_grind_long_confirmed = False
                is_grind_short_confirmed = False
                break
            
            s_slope = df['trend_slope'].iloc[idx]
            s_r2 = df['trend_r2'].iloc[idx]
            s_vola = df['volatility_ratio'].iloc[idx]
            s_close = df['Close'].iloc[idx]
            s_ema50 = df['ema_50'].iloc[idx]
            
            slopes.append(s_slope)
            r2s.append(s_r2)
            volas.append(s_vola)
            
            # Hard Constraint: Price vs EMA50
            if not (s_close > s_ema50): is_grind_long_confirmed = False
            if not (s_close < s_ema50): is_grind_short_confirmed = False
            
        if is_grind_long_confirmed or is_grind_short_confirmed:
            avg_slope = np.mean(slopes)
            avg_r2 = np.mean(r2s)
            avg_vola = np.mean(volas)
            
            if is_grind_long_confirmed:
                if not ((avg_slope > 0.5) and (avg_r2 > 0.80) and (avg_vola < 1.5)):
                    is_grind_long_confirmed = False
            
            if is_grind_short_confirmed:
                if not ((avg_slope < -0.5) and (avg_r2 > 0.80) and (avg_vola < 1.5)):
                    is_grind_short_confirmed = False
            
        is_grind_long_raw = is_grind_long_confirmed
        is_grind_short_raw = is_grind_short_confirmed
        
        # Reset Grind State if condition breaks
        if not is_grind_long_raw: has_traded_current_grind_long = False
        if not is_grind_short_raw: has_traded_current_grind_short = False

        # Record Signal (Allow re-entry after 40 candles in same trend)
        if is_grind_long_raw:
            last_trade_idx = grind_longs[-1] if grind_longs else -999
            if (not has_traded_current_grind_long) or (i - last_trade_idx > 40):
                grind_longs.append(i)
                has_traded_current_grind_long = True
            
        if is_grind_short_raw:
            last_trade_idx = grind_shorts[-1] if grind_shorts else -999
            if (not has_traded_current_grind_short) or (i - last_trade_idx > 40):
                grind_shorts.append(i)
                has_traded_current_grind_short = True
        
        # SFP Definition
        is_sfp_long_raw = df['is_sfp_long'].iloc[i]
        is_sfp_short_raw = df['is_sfp_short'].iloc[i]
        
        is_sfp_long_major = df['is_sfp_long_major'].iloc[i]
        is_sfp_short_major = df['is_sfp_short_major'].iloc[i]
        
        current_pivot_low = df['active_pivot_low'].iloc[i]
        current_pivot_high = df['active_pivot_high'].iloc[i]

        if is_sfp_long_raw and (current_pivot_low != last_traded_sfp_low_level):
            sfp_longs.append(i)
            if is_sfp_long_major: sfp_longs_major.append(i)
            else: sfp_longs_minor.append(i)
            last_traded_sfp_low_level = current_pivot_low
        
        if is_sfp_short_raw and (current_pivot_high != last_traded_sfp_high_level):
            sfp_shorts.append(i)
            if is_sfp_short_major: sfp_shorts_major.append(i)
            else: sfp_shorts_minor.append(i)
            last_traded_sfp_high_level = current_pivot_high
        
        # GP Logic
        current_bull_entry = df['bullish_range_high'].iloc[i] # 0.618
        current_bear_entry = df['bearish_range_low'].iloc[i]  # 0.618
        
        # --- GP Long Logic ---
        # 1. Detection & Activation
        if pending_gp_long is None:
            # Check if valid zone exists (not NaN)
            if not np.isnan(current_bull_entry):
                # Check if we are entering the zone for the first time or re-entering
                # Condition: Low touches 0.618
                if low <= current_bull_entry:
                     # Calculate SL at 0.7 Retracement
                    range_val = df['active_pivot_high_major_4h'].iloc[i] - df['active_pivot_low_major_4h'].iloc[i]
                    sl_price = df['active_pivot_high_major_4h'].iloc[i] - (range_val * 0.7)
                    
                    # Only activate if we haven't already traded this specific zone
                    if abs(current_bull_entry - last_bullish_range_high) > 1e-6:
                        pending_gp_long = {
                            'active': True,
                            'start_idx': i,
                            'sl': sl_price,
                            'entry_level': current_bull_entry
                        }
        
        # 2. Monitoring Pending Long
        if pending_gp_long is not None:
            # A. Invalidation (Price breaks SL 0.7)
            if low < pending_gp_long['sl']:
                pending_gp_long = None # Cancelled/Failed
            
            # B. Timeout (e.g., 24 candles = 2 hours)
            elif (i - pending_gp_long['start_idx']) > 24:
                pending_gp_long = None
                
            # C. Reaction Trigger
            else:
                # Volatile Reaction:
                # 1. Candle is Green (Close > Open)
                # 2. Volatility is high (High - Low > Avg_Range * 0.8)
                
                is_green = price > open_price
                candle_volatility = high - low
                is_volatile = candle_volatility > (avg_range * 0.8) # Strong move
                
                if is_green and is_volatile:
                    # EXECUTE LONG
                    gp_longs.append(i)
                    last_bullish_range_high = pending_gp_long['entry_level'] # Mark as traded
                    
                    if swing_trade['size'] == 0:
                        pass
                        # GP TRADES DISABLED
                        # swing_trade = {
                        #     'size': 1.0,
                        #     'entry': price,
                        #     'sl': pending_gp_long['sl'],
                        #     'side': 'long',
                        #     'reason': 'GP Long (Volatile)',
                        #     'start_idx': i,
                        #     'tp_taken': False,
                        #     'breakeven_active': False
                        # }
                    pending_gp_long = None

        # --- GP Short Logic ---
        # 1. Detection & Activation
        if pending_gp_short is None:
            # Check if valid zone exists (not NaN)
            if not np.isnan(current_bear_entry):
                # Condition: High touches 0.618
                if high >= current_bear_entry:
                    # Calculate SL at 0.7 Retracement
                    range_val = df['active_pivot_high_major_4h'].iloc[i] - df['active_pivot_low_major_4h'].iloc[i]
                    sl_price = df['active_pivot_low_major_4h'].iloc[i] + (range_val * 0.7)
                    
                    if abs(current_bear_entry - last_bearish_range_low) > 1e-6:
                        pending_gp_short = {
                            'active': True,
                            'start_idx': i,
                            'sl': sl_price,
                            'entry_level': current_bear_entry
                        }
                    
        # 2. Monitoring Pending Short
        if pending_gp_short is not None:
            # A. Invalidation
            if high > pending_gp_short['sl']:
                pending_gp_short = None
            
            # B. Timeout
            elif (i - pending_gp_short['start_idx']) > 24:
                pending_gp_short = None
                
            # C. Reaction Trigger
            else:
                # Volatile Reaction: Red + High Volatility
                is_red = price < open_price
                candle_volatility = high - low
                is_volatile = candle_volatility > (avg_range * 0.8)
                
                if is_red and is_volatile:
                    # EXECUTE SHORT
                    gp_shorts.append(i)
                    last_bearish_range_low = pending_gp_short['entry_level']
                    
                    if swing_trade['size'] == 0:
                        pass
                        # GP TRADES DISABLED
                        # swing_trade = {
                        #     'size': 1.0,
                        #     'entry': price,
                        #     'sl': pending_gp_short['sl'],
                        #     'side': 'short',
                        #     'reason': 'GP Short (Volatile)',
                        #     'start_idx': i,
                        #     'tp_taken': False,
                        #     'breakeven_active': False
                        # }
                    pending_gp_short = None

        # --- SIMULATION LOGIC ---
        
        # 1. Check Exits
        
        # Swing Exits
        if swing_trade['size'] > 0:
            st = swing_trade
            exit_price = 0
            exit_reason = ''
            pnl_realized = 0
            
            # Breakeven Logic for GP Trades
            if 'gp' in st['reason'].lower() and not st.get('breakeven_active', False):
                risk = abs(st['entry'] - st['sl'])
                current_pnl = (high - st['entry']) if st['side'] == 'long' else (st['entry'] - low)
                
                if current_pnl >= (1.0 * risk):
                    st['sl'] = st['entry'] # Move SL to Entry
                    st['breakeven_active'] = True
            
            # SL Hit
            if (st['side'] == 'long' and low < st['sl']):
                exit_price = st['sl']
                exit_reason = 'SL Hit'
            elif (st['side'] == 'short' and high > st['sl']):
                exit_price = st['sl']
                exit_reason = 'SL Hit'
            # Reversal
            elif (st['side'] == 'long' and (is_sfp_short_raw or (current_bear_entry > 0 and df['dist_to_gp_short'].iloc[i] == 0))):
                exit_price = price
                exit_reason = 'Reversal'
            elif (st['side'] == 'short' and (is_sfp_long_raw or (current_bull_entry > 0 and df['dist_to_gp_long'].iloc[i] == 0))):
                exit_price = price
                exit_reason = 'Reversal'
                
            if exit_price > 0:
                # Calculate remaining PnL based on current size (1.0 or 0.5)
                size = st.get('size', 1.0)
                if size == 0: size = 1.0 # Default if not set properly
                
                final_pnl = ((exit_price - st['entry']) if st['side'] == 'long' else (st['entry'] - exit_price)) * size
                total_pnl = st.get('banked_pnl', 0) + final_pnl
                
                executed_trades.append({
                    'type': 'swing', 'side': st['side'], 'entry_idx': st['start_idx'], 'exit_idx': i,
                    'entry_price': st['entry'], 'exit_price': exit_price, 'sl': st['sl'], 'pnl': total_pnl, 'reason': st['reason'], 'exit_reason': exit_reason
                })
                swing_trade = {'size': 0, 'entry': 0, 'sl': 0, 'side': None, 'reason': None, 'start_idx': 0, 'tp_taken': False}

        # Scalp Exits
        if scalp_trade['size'] > 0:
            st = scalp_trade
            exit_price = 0
            exit_reason = ''
            pnl_realized = 0
            
            current_atr = df['avg_range_10'].iloc[i]
            initial_risk = abs(st['entry'] - st['initial_sl'])
            
            # Hybrid: Partial TP (50% at 2R) + Activate Trailing SL
            if not st['tp_taken']:
                # Check if we hit 2R Profit
                current_profit = (high - st['entry']) if st['side'] == 'long' else (st['entry'] - low)
                
                if current_profit >= 2 * initial_risk:
                    # 1. Take Partial Profit (50%)
                    tp_price = st['entry'] + (2 * initial_risk) if st['side'] == 'long' else st['entry'] - (2 * initial_risk)
                    pnl_realized += (tp_price - st['entry']) * 0.5 if st['side'] == 'long' else (st['entry'] - tp_price) * 0.5
                    st['size'] = 0.5
                    st['tp_taken'] = True
                    st['banked_pnl'] = pnl_realized
                    
                    # 2. Activate Trailing SL (Tight: 1.0 ATR)
                    st['trailing_active'] = True
                    if st['side'] == 'long':
                        st['sl'] = price - (1.0 * current_atr)
                    else:
                        st['sl'] = price + (1.0 * current_atr)

            # Update Trailing SL if Active
            if st['trailing_active']:
                if st['side'] == 'long':
                    new_sl = price - (1.0 * current_atr)
                    if new_sl > st['sl']: st['sl'] = new_sl
                else:
                    new_sl = price + (1.0 * current_atr)
                    if new_sl < st['sl']: st['sl'] = new_sl

            # SL Hit
            if (st['side'] == 'long' and low < st['sl']):
                exit_price = st['sl']
                exit_reason = 'SL Hit'
            elif (st['side'] == 'short' and high > st['sl']):
                exit_price = st['sl']
                exit_reason = 'SL Hit'
            else:
                # Trend Stop - Desensitized (Stay in trade longer)
                should_close = False
                # Old: (slope <= 0 or r2 < 0.80)
                # New: Allow flat slope (0) and lower R2 (0.5) before exiting
                if st['reason'] == 'grind_long' and (slope < -0.1 or r2 < 0.50): should_close = True
                if st['reason'] == 'grind_short' and (slope > 0.1 or r2 < 0.50): should_close = True
                
                if should_close:
                    exit_price = price
                    exit_reason = 'Trend Stop' # Renamed from Trend/Impulse since Impulse is off
            
            if exit_price > 0:
                # Calculate remaining PnL based on current size (1.0 or 0.5)
                size = st.get('size', 1.0)
                if size == 0: size = 1.0
                
                final_pnl = ((exit_price - st['entry']) if st['side'] == 'long' else (st['entry'] - exit_price)) * size
                total_pnl = st.get('banked_pnl', 0) + final_pnl
                
                executed_trades.append({
                    'type': 'scalp', 'side': st['side'], 'entry_idx': st['start_idx'], 'exit_idx': i,
                    'entry_price': st['entry'], 'exit_price': exit_price, 'sl': st['sl'], 'pnl': total_pnl, 'reason': st['reason'], 'exit_reason': exit_reason
                })
                scalp_trade = {'size': 0, 'entry': 0, 'sl': 0, 'initial_sl': 0, 'side': None, 'reason': None, 'start_idx': 0, 'trailing_active': False, 'tp_taken': False}

        # 2. Check Entries
        
        # Swing Entry (SFP / GP)
        if swing_trade['size'] == 0:
            # Check SFP (GP is handled explicitly above)
            if i in sfp_longs:
                sl = low # SL at Signal Low
                swing_trade = {'size': 1, 'entry': price, 'sl': sl, 'side': 'long', 'reason': 'sfp_long', 'start_idx': i, 'tp_taken': False}
            elif i in sfp_shorts:
                sl = high # SL at Signal High
                swing_trade = {'size': 1, 'entry': price, 'sl': sl, 'side': 'short', 'reason': 'sfp_short', 'start_idx': i, 'tp_taken': False}
        
        # Scalp Entry (Grind)
        if scalp_trade['size'] == 0:
            if i in grind_longs:
                atr = df['avg_range_10'].iloc[i]
                sl = price - (2 * atr)
                scalp_trade = {'size': 1, 'entry': price, 'sl': sl, 'initial_sl': sl, 'side': 'long', 'reason': 'grind_long', 'start_idx': i, 'trailing_active': False, 'tp_taken': False}
            elif i in grind_shorts:
                atr = df['avg_range_10'].iloc[i]
                sl = price + (2 * atr)
                scalp_trade = {'size': 1, 'entry': price, 'sl': sl, 'initial_sl': sl, 'side': 'short', 'reason': 'grind_short', 'start_idx': i, 'trailing_active': False, 'tp_taken': False}

    # Close open trades at end of simulation
    if swing_trade['size'] > 0:
        st = swing_trade
        exit_price = df['Close'].iloc[-1]
        final_pnl = ((exit_price - st['entry']) if st['side'] == 'long' else (st['entry'] - exit_price)) * st['size']
        executed_trades.append({
            'type': 'swing', 'side': st['side'], 'entry_idx': st['start_idx'], 'exit_idx': len(df)-1,
            'entry_price': st['entry'], 'exit_price': exit_price, 'sl': st['sl'], 'pnl': final_pnl, 'reason': st['reason'], 'exit_reason': 'End of Data'
        })

    print(f"Total Executed Trades: {len(executed_trades)} (Swing: {len([t for t in executed_trades if t['type']=='swing'])}, Scalp: {len([t for t in executed_trades if t['type']=='scalp'])})")
    
    # --- Performance Summary ---
    print("\n--- VALIDATION PERFORMANCE SUMMARY ---")
    strategies = ['grind_long', 'grind_short', 'sfp_long', 'sfp_short', 'gp_long', 'gp_short']
    print(f"{'Strategy':<15} | {'Trades':<8} | {'Win Rate':<10} | {'Total PnL':<12} | {'Avg PnL':<10}")
    print("-" * 65)
    
    for strat in strategies:
        # Fuzzy match for reason
        strat_key = strat.replace('_', ' ') # 'gp long'
        strat_trades = [t for t in executed_trades if strat_key in t['reason'].lower().replace('_', ' ')]
        
        count = len(strat_trades)
        if count > 0:
            wins = len([t for t in strat_trades if t['pnl'] > 0])
            win_rate = (wins / count) * 100
            total_pnl = sum([t['pnl'] for t in strat_trades])
            avg_pnl = total_pnl / count
            print(f"{strat:<15} | {count:<8} | {win_rate:<9.1f}% | {total_pnl:<12.2f} | {avg_pnl:<10.2f}")
        else:
            print(f"{strat:<15} | 0        | 0.0%       | 0.00         | 0.00")
    print("-" * 65)

    # --- Advanced Metrics Calculation ---
    if executed_trades:
        df_trades = pd.DataFrame(executed_trades)
        
        # 1. Profit Factor
        gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        
        # 2. Avg Risk:Reward
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean()
        avg_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].mean())
        avg_rr = avg_win / avg_loss if avg_loss != 0 else 0
        
        # 3. Sortino Ratio (Reconstructed Equity Curve)
        # Assume $10,000 starting balance
        balance = 10000
        equity_curve = [balance]
        
        # Sort trades by entry index to simulate time
        df_trades_sorted = df_trades.sort_values('entry_idx')
        
        for pnl in df_trades_sorted['pnl']:
            balance += pnl
            equity_curve.append(balance)
            
        returns = pd.Series(equity_curve).pct_change().dropna()
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) > 0:
            # Annualize (assuming this is 1 year of data)
            # We have N trades over 1 year.
            # Standard Sortino uses daily/hourly returns.
            # Here we use trade-based returns, which is an approximation.
            # Better: Map trades to a daily equity curve.
            
            # Create Daily Equity Curve
            df['Date'] = pd.to_datetime(df['Date'])
            daily_pnl = pd.Series(0.0, index=df['Date'])
            
            for _, t in df_trades.iterrows():
                exit_date = df['Date'].iloc[t['exit_idx']]
                daily_pnl[exit_date] += t['pnl']
                
            daily_equity = daily_pnl.resample('D').sum().cumsum() + 10000
            daily_returns = daily_equity.pct_change().dropna()
            
            downside_daily = daily_returns[daily_returns < 0]
            downside_dev = downside_daily.std() * np.sqrt(365)
            annual_return = daily_returns.mean() * 365
            
            sortino = annual_return / downside_dev if downside_dev != 0 else 0
        else:
            sortino = 0 # No losing days? Unlikely.

        print("\nADVANCED METRICS (2022 Simulation)")
        print("-" * 30)
        print(f"Profit Factor:      {profit_factor:.2f}")
        print(f"Avg Risk:Reward:    1:{avg_rr:.2f}")
        print(f"Sortino Ratio:      {sortino:.2f}")
        print("-" * 30)

    signals_dict = {
        'grind_long': grind_longs,
        'grind_short': grind_shorts,
        'sfp_long_major': sfp_longs_major,
        'sfp_short_major': sfp_shorts_major,
        'sfp_long_minor': sfp_longs_minor,
        'sfp_short_minor': sfp_shorts_minor,
        'gp_long': gp_longs,
        'gp_short': gp_shorts,
        'executed_trades': executed_trades
    }
    
    # Plotting
    print("Generating 2022 Equity Curve...")
    plot_equity_curve(df, signals_dict, "2022-01-01", "2022-12-31", "2022_NoGP_Equity")
    
    print("Generating 2022 Overview (Bear Market Start)...")
    plot_higher_timeframe(df, signals_dict, "2022-01-01", "2022-02-28", "2022_Bear_Start")

    # Generate Performance Report
    periods = {
        "Bear Start": ("2022-01-01", "2022-02-28"),
        "Full Year": ("2022-01-01", "2022-12-31")
    }
    generate_performance_report(df, signals_dict, periods, filename="Performance_Report_2022_Specifics.txt")

    # Save Trades to CSV for Monte Carlo
    trades_df = pd.DataFrame(executed_trades)
    if not trades_df.empty:
        # Add Date column
        trades_df['entry_date'] = trades_df['entry_idx'].apply(lambda x: df['Date'].iloc[x])
        trades_df['exit_date'] = trades_df['exit_idx'].apply(lambda x: df['Date'].iloc[x])
        trades_df.to_csv('validation_trades_2023.csv', index=False)
        print("Saved executed trades to validation_trades_2023.csv")

    # --- Generate Results WITHOUT GP Trades ---
    non_gp_trades = [t for t in executed_trades if 'gp' not in t['reason'].lower()]
    
    if non_gp_trades:
        df_trades = pd.DataFrame(non_gp_trades)
        
        # Metrics
        gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean()
        avg_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].mean())
        avg_rr = avg_win / avg_loss if avg_loss != 0 else 0
        
        # Sortino (Approx)
        df['Date'] = pd.to_datetime(df['Date'])
        daily_pnl = pd.Series(0.0, index=df['Date'])
        
        for _, t in df_trades.iterrows():
            exit_date = df['Date'].iloc[t['exit_idx']]
            daily_pnl[exit_date] += t['pnl']
            
        daily_equity = daily_pnl.resample('D').sum().cumsum() + 10000
        daily_returns = daily_equity.pct_change().dropna()
        
        downside_daily = daily_returns[daily_returns < 0]
        downside_dev = downside_daily.std() * np.sqrt(365)
        annual_return = daily_returns.mean() * 365
        sortino = annual_return / downside_dev if downside_dev != 0 else 0

        # Write to File
        with open('Results_No_GP_2023.txt', 'w') as f:
            f.write("RESULTS WITHOUT GP TRADES (2023)\n")
            f.write("================================\n\n")
            f.write(f"Total Trades:       {len(non_gp_trades)}\n")
            f.write(f"Total PnL:          ${sum(t['pnl'] for t in non_gp_trades):.2f}\n")
            f.write(f"Win Rate:           {len([t for t in non_gp_trades if t['pnl'] > 0]) / len(non_gp_trades) * 100:.2f}%\n\n")
            
            f.write("ADVANCED METRICS\n")
            f.write("----------------\n")
            f.write(f"Profit Factor:      {profit_factor:.2f}\n")
            f.write(f"Avg Risk:Reward:    1:{avg_rr:.2f}\n")
            f.write(f"Sortino Ratio:      {sortino:.2f}\n\n")
            
            f.write("STRATEGY BREAKDOWN\n")
            f.write("------------------\n")
            strategies = set(t['reason'] for t in non_gp_trades)
            f.write(f"{'Strategy':<20} | {'Count':<5} | {'Win Rate':<8} | {'PnL':<10}\n")
            f.write("-" * 55 + "\n")
            for strat in strategies:
                strat_trades = [t for t in non_gp_trades if t['reason'] == strat]
                s_count = len(strat_trades)
                s_wins = len([t for t in strat_trades if t['pnl'] > 0])
                s_wr = (s_wins / s_count) * 100
                s_pnl = sum(t['pnl'] for t in strat_trades)
                f.write(f"{strat:<20} | {s_count:<5} | {s_wr:<8.1f} | {s_pnl:<10.2f}\n")
        
        print("Saved results without GP trades to Results_No_GP_2023.txt")

def plot_grind_trades_only(df, signals_dict, start_date, end_date, filename_suffix):
    # Filter by date
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    subset = df.loc[mask].copy()
    
    if subset.empty:
        print(f"No data found for range {start_date} to {end_date}")
        return

    # Use raw 5m data for Intraday View
    subset.set_index('Date', inplace=True)
    
    plt.figure(figsize=(24, 12))
    # Plot 5m Close Price
    plt.plot(subset.index, subset['Close'], label='5m Close Price', color='black', alpha=0.5, linewidth=1)
    
    # Plot Executed Trades
    trades = signals_dict.get('executed_trades', [])
    
    # Filter trades in range AND Grind only
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    grind_trades = [t for t in trades if 'grind' in t['reason']]
    
    for trade in grind_trades:
        entry_date = pd.to_datetime(df['Date'].iloc[trade['entry_idx']])
        exit_date = pd.to_datetime(df['Date'].iloc[trade['exit_idx']])
        
        # Check if trade overlaps with the view window
        if (entry_date >= start_dt and entry_date <= end_dt) or (exit_date >= start_dt and exit_date <= end_dt):
            # Color based on PnL
            color = 'green' if trade['pnl'] > 0 else 'red'
            linestyle = '-'
            linewidth = 2
            
            # Plot Line (Entry to Exit)
            plt.plot([entry_date, exit_date], [trade['entry_price'], trade['exit_price']], 
                     color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.9, zorder=5)
            
            # Plot Markers
            # Entry = Circle (Same color as trade)
            plt.scatter(entry_date, trade['entry_price'], marker='o', color=color, s=100, zorder=6, edgecolors='black', label='Entry' if 'Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
            
            # Exit = Black X (Distinct)
            plt.scatter(exit_date, trade['exit_price'], marker='X', color='black', s=100, zorder=7, label='Exit' if 'Exit' not in plt.gca().get_legend_handles_labels()[1] else "")
            
            # Annotation for Exit Reason
            plt.annotate(trade.get('exit_reason', 'Exit'), 
                         (exit_date, trade['exit_price']),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', fontsize=9, fontweight='bold', color='black',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))
            
            # Plot SL (Horizontal Line)
            # Make it very visible: Thicker, distinct color, zorder high
            if 'sl' in trade:
                # Ensure line has length even if entry/exit are same candle
                sl_end_date = exit_date
                if exit_date == entry_date:
                    sl_end_date = entry_date + pd.Timedelta(minutes=5)
                
                plt.hlines(trade['sl'], xmin=entry_date, xmax=sl_end_date, colors='orange', linestyles='-', linewidth=2.5, alpha=1.0, zorder=4, label='SL' if 'SL' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"GRIND TRADES INTRADAY (5m): {start_date} - {end_date} (Green=Win, Red=Loss, Orange=SL)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    filename = f"validation_{filename_suffix}.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def plot_higher_timeframe(df, signals_dict, start_date, end_date, filename_suffix):
    # Filter by date
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    subset = df.loc[mask].copy()
    
    if subset.empty:
        print(f"No data found for range {start_date} to {end_date}")
        return

    # Resample to 1H for cleaner plot
    subset.set_index('Date', inplace=True)
    df_1h = subset['Close'].resample('1H').ohlc()
    df_1h['Close'] = df_1h['close']
    
    plt.figure(figsize=(24, 12))
    plt.plot(df_1h.index, df_1h['Close'], label='1H Close Price', color='gray', alpha=0.5, linewidth=1)
    
    # Plot Executed Trades
    trades = signals_dict.get('executed_trades', [])
    
    # Filter trades in range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    for trade in trades:
        entry_date = pd.to_datetime(df['Date'].iloc[trade['entry_idx']])
        exit_date = pd.to_datetime(df['Date'].iloc[trade['exit_idx']])
        
        if entry_date >= start_dt and exit_date <= end_dt:
            # Color based on Side (Green=Long, Red=Short)
            color = 'green' if trade['side'] == 'long' else 'red'
            
            # Marker based on Reason
            # SFP = Star (*), GP = Cross (x), Grind = Arrow (^/v)
            reason = trade['reason']
            if 'sfp' in reason:
                marker = '*'
                marker_size = 200
                label_prefix = "SFP"
            elif 'gp' in reason:
                marker = 'X' # Big X
                marker_size = 150
                label_prefix = "GP"
            elif 'grind' in reason:
                marker = '^' if trade['side'] == 'long' else 'v'
                marker_size = 100
                label_prefix = "Grind"
            else:
                marker = 'o'
                marker_size = 50
                label_prefix = "Other"

            # Plot Line (Entry to Exit)
            linestyle = '-' if trade['pnl'] > 0 else '--' # Solid for win, dashed for loss
            plt.plot([entry_date, exit_date], [trade['entry_price'], trade['exit_price']], 
                     color=color, linestyle=linestyle, linewidth=1.5, alpha=0.7)
            
            # Plot Entry Marker
            # Only add label once per type to avoid legend clutter
            label = f"{label_prefix} {trade['side'].title()}"
            if label in plt.gca().get_legend_handles_labels()[1]:
                label = ""
                
            plt.scatter(entry_date, trade['entry_price'], marker=marker, color=color, s=marker_size, zorder=5, label=label, edgecolors='black')
            
            # Plot Exit Marker (Small dot)
            plt.scatter(exit_date, trade['exit_price'], marker='o', color='black', s=20, zorder=4)

    plt.title(f"Trades View: {start_date} - {end_date} | Green=Long, Red=Short | Star=SFP, X=GP, Arrow=Grind | Solid=Win, Dashed=Loss")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    filename = f"validation_{filename_suffix}.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def plot_gp_detailed_view(df, signals_dict, start_date, end_date, filename_suffix):
    # Filter by date
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    subset = df.loc[mask].copy()
    
    if subset.empty:
        print(f"No data found for range {start_date} to {end_date}")
        return

    # Use raw 5m data for detailed view
    subset.set_index('Date', inplace=True)
    
    plt.figure(figsize=(24, 12))
    plt.plot(subset.index, subset['Close'], label='5m Close Price', color='black', alpha=0.5, linewidth=1)
    
    # Plot GP Zones
    # Bullish GP Zone (Green)
    if 'bullish_range_high' in subset.columns and 'bullish_range_low' in subset.columns:
        plt.fill_between(subset.index, 
                         subset['bullish_range_low'], 
                         subset['bullish_range_high'], 
                         color='green', alpha=0.15, label='Bullish GP Zone (0.618-0.67)')
                         
    # Bearish GP Zone (Red)
    if 'bearish_range_low' in subset.columns and 'bearish_range_high' in subset.columns:
        plt.fill_between(subset.index, 
                         subset['bearish_range_low'], 
                         subset['bearish_range_high'], 
                         color='red', alpha=0.15, label='Bearish GP Zone (0.618-0.67)')

    # Plot Executed Trades
    trades = signals_dict.get('executed_trades', [])
    
    # Filter trades in range AND GP only
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    gp_trades = [t for t in trades if 'gp' in t['reason']]
    
    for trade in gp_trades:
        entry_date = pd.to_datetime(df['Date'].iloc[trade['entry_idx']])
        exit_date = pd.to_datetime(df['Date'].iloc[trade['exit_idx']])
        
        if (entry_date >= start_dt and entry_date <= end_dt) or (exit_date >= start_dt and exit_date <= end_dt):
            # Color based on PnL
            color = 'green' if trade['pnl'] > 0 else 'red'
            
            # Plot Line (Entry to Exit)
            plt.plot([entry_date, exit_date], [trade['entry_price'], trade['exit_price']], 
                     color=color, linestyle='-', linewidth=2, alpha=0.9, zorder=5)
            
            # Plot Markers
            plt.scatter(entry_date, trade['entry_price'], marker='X', color=color, s=150, zorder=6, edgecolors='black', label='GP Entry' if 'GP Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.scatter(exit_date, trade['exit_price'], marker='o', color='black', s=50, zorder=7)
            
            # Annotate Reasoning
            idx = trade['entry_idx']
            if trade['side'] == 'long':
                sl_price = trade['sl']
                plt.annotate(f"GP Long\nEntry: {trade['entry_price']:.2f}\nSL: {sl_price:.2f}", 
                             (entry_date, trade['entry_price']),
                             xytext=(0, -40), textcoords='offset points',
                             ha='center', fontsize=9, fontweight='bold', color='green',
                             arrowprops=dict(arrowstyle="->", color='green'))
                             
            else:
                sl_price = trade['sl']
                plt.annotate(f"GP Short\nEntry: {trade['entry_price']:.2f}\nSL: {sl_price:.2f}", 
                             (entry_date, trade['entry_price']),
                             xytext=(0, 40), textcoords='offset points',
                             ha='center', fontsize=9, fontweight='bold', color='red',
                             arrowprops=dict(arrowstyle="->", color='red'))

            # Plot SL Line
            sl_end_date = exit_date if exit_date > entry_date else entry_date + pd.Timedelta(minutes=30)
            plt.hlines(trade['sl'], xmin=entry_date, xmax=sl_end_date, colors='orange', linestyles='--', linewidth=2, label='SL' if 'SL' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"GOLDEN POCKET (GP) TRADES DETAILED: {start_date} - {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    
    filename = f"validation_{filename_suffix}.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def generate_performance_report(df, signals_dict, periods, filename="Performance_Report_2022_Specifics.txt"):
    trades = signals_dict.get('executed_trades', [])
    df['Date'] = pd.to_datetime(df['Date'])
    
    with open(filename, 'w') as f:
        f.write("PERFORMANCE REPORT - 2022 SPECIFIC PERIODS\n")
        f.write("============================================\n\n")
        
        for title, (start_date, end_date) in periods.items():
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            period_trades = []
            for t in trades:
                entry_date = df['Date'].iloc[t['entry_idx']]
                if entry_date >= start_dt and entry_date <= end_dt:
                    period_trades.append(t)
            
            f.write(f"--- {title} ({start_date} to {end_date}) ---\n")
            f.write(f"Total Trades: {len(period_trades)}\n")
            
            if not period_trades:
                f.write("No trades found.\n\n")
                continue
                
            total_pnl = sum(t['pnl'] for t in period_trades)
            wins = len([t for t in period_trades if t['pnl'] > 0])
            win_rate = (wins / len(period_trades)) * 100
            
            f.write(f"Total PnL: ${total_pnl:.2f}\n")
            f.write(f"Win Rate: {win_rate:.1f}%\n")
            
            # By Strategy
            strategies = set(t['reason'] for t in period_trades)
            f.write("\nStrategy Breakdown:\n")
            f.write(f"{'Strategy':<15} | {'Count':<5} | {'Win Rate':<8} | {'PnL':<10}\n")
            f.write("-" * 45 + "\n")
            
            for strat in strategies:
                strat_trades = [t for t in period_trades if t['reason'] == strat]
                s_count = len(strat_trades)
                s_wins = len([t for t in strat_trades if t['pnl'] > 0])
                s_wr = (s_wins / s_count) * 100
                s_pnl = sum(t['pnl'] for t in strat_trades)
                f.write(f"{strat:<15} | {s_count:<5} | {s_wr:<8.1f} | {s_pnl:<10.2f}\n")
            
            f.write("\n\n")
            
    print(f"Performance report saved to {filename}")

def plot_sfp_only_overview(df, signals_dict, start_date, end_date, filename_suffix):
    # Filter by date
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    subset = df.loc[mask].copy()
    
    if subset.empty:
        print(f"No data found for range {start_date} to {end_date}")
        return

    # Resample to 4H for cleaner full-year plot
    subset.set_index('Date', inplace=True)
    df_4h = subset['Close'].resample('4h').ohlc()
    df_4h['Close'] = df_4h['close']
    
    plt.figure(figsize=(24, 12))
    plt.plot(df_4h.index, df_4h['Close'], label='4H Close Price', color='gray', alpha=0.5, linewidth=1)
    
    # Plot Executed Trades
    trades = signals_dict.get('executed_trades', [])
    
    # Filter trades in range AND SFP only
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    sfp_trades = [t for t in trades if 'sfp' in t['reason']]
    
    for trade in sfp_trades:
        entry_date = pd.to_datetime(df['Date'].iloc[trade['entry_idx']])
        exit_date = pd.to_datetime(df['Date'].iloc[trade['exit_idx']])
        
        if (entry_date >= start_dt and entry_date <= end_dt) or (exit_date >= start_dt and exit_date <= end_dt):
            # Color based on PnL (Green=Win, Red=Loss)
            color = 'green' if trade['pnl'] > 0 else 'red'
            
            # Plot Line (Entry to Exit)
            plt.plot([entry_date, exit_date], [trade['entry_price'], trade['exit_price']], 
                     color=color, linestyle='-', linewidth=2, alpha=0.8)
            
            # Plot Entry Marker
            marker = '^' if trade['side'] == 'long' else 'v'
            plt.scatter(entry_date, trade['entry_price'], marker=marker, color=color, s=100, zorder=5, edgecolors='black')

    plt.title(f"SFP TRADES OVERVIEW: {start_date} - {end_date} (Green=Win, Red=Loss)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    
    filename = f"validation_{filename_suffix}.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def plot_equity_curve(df, signals_dict, start_date, end_date, filename_suffix, initial_balance=10000):
    # Filter by date
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    subset = df.loc[mask].copy()
    
    if subset.empty:
        print(f"No data found for range {start_date} to {end_date}")
        return

    # Get Trades
    trades = signals_dict.get('executed_trades', [])
    # Filter out GP trades if we want to visualize the "No GP" performance
    # But let's visualize ALL executed trades passed in signals_dict
    # If we want "No GP", we should filter them before passing or here.
    # Let's filter GP out here to match the user's request for the "No GP" performance visualization
    trades = [t for t in trades if 'gp' not in t['reason'].lower()]

    # Create Equity Curve
    # We need a time series of balance.
    # Start with initial balance
    balance = initial_balance
    equity_data = [] # (date, balance)
    equity_data.append((pd.to_datetime(start_date), balance))
    
    # Sort trades by exit time
    trades_sorted = sorted(trades, key=lambda x: x['exit_idx'])
    
    current_trade_idx = 0
    
    # Iterate through subset to build daily/hourly equity curve?
    # Simpler: Just plot points at trade exits.
    for trade in trades_sorted:
        exit_date = df['Date'].iloc[trade['exit_idx']]
        if exit_date < pd.to_datetime(start_date): continue
        if exit_date > pd.to_datetime(end_date): break
        
        balance += trade['pnl']
        equity_data.append((exit_date, balance))
        
    # Add final point
    equity_data.append((pd.to_datetime(end_date), balance))
    
    equity_df = pd.DataFrame(equity_data, columns=['Date', 'Balance'])
    equity_df.set_index('Date', inplace=True)
    
    # Resample to have a continuous line
    # equity_df = equity_df.resample('1H').ffill()

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(24, 12))
    
    # Plot Equity Curve (Green/Red fill)
    ax1.plot(equity_df.index, equity_df['Balance'], label='Portfolio Balance', color='blue', linewidth=2)
    ax1.fill_between(equity_df.index, equity_df['Balance'], initial_balance, where=(equity_df['Balance'] >= initial_balance), color='green', alpha=0.1, interpolate=True)
    ax1.fill_between(equity_df.index, equity_df['Balance'], initial_balance, where=(equity_df['Balance'] < initial_balance), color='red', alpha=0.1, interpolate=True)
    
    ax1.set_ylabel('Balance ($)', color='blue', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axhline(y=initial_balance, color='black', linestyle='--', alpha=0.5, label='Initial Balance')
    
    # Plot Buy & Hold Comparison (BTC Price) on Secondary Axis
    ax2 = ax1.twinx()
    subset.set_index('Date', inplace=True)
    # Resample for cleaner plot
    price_series = subset['Close'].resample('4h').last()
    
    ax2.plot(price_series.index, price_series, label='BTC Price (Buy & Hold)', color='gray', alpha=0.4, linewidth=1.5, linestyle='--')
    ax2.set_ylabel('BTC Price ($)', color='gray', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Title & Legend
    plt.title(f"PORTFOLIO PERFORMANCE vs MARKET: {start_date} - {end_date}\nStrategy: SFP + Grind (No GP)", fontsize=16, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    
    filename = f"equity_curve_{filename_suffix}.png"
    plt.savefig(filename)
    print(f"Saved equity curve to {filename}")
    plt.close()

def plot_segment(df, start_idx, length, signals_dict, filename_suffix):
    # Legacy function, kept empty to avoid errors if called
    pass

if __name__ == "__main__":
    validate_signals()
