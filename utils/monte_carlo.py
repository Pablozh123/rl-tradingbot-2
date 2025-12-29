import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

TRADES_FILE = 'trade_log_detailed.csv'
INITIAL_CAPITAL = 10000
SIMULATIONS = 2000
TRADES_PER_SIMULATION = 2000

def run_monte_carlo():
    if not os.path.exists(TRADES_FILE):
        print(f"File not found: {TRADES_FILE}. Run evaluate.py first.")
        return

    print(f"Loading trades from {TRADES_FILE}...")
    df = pd.read_csv(TRADES_FILE)
    
    if df.empty:
        print("No trades found.")
        return

    # Extract PnL series
    pnls = df['pnl'].values
    
    print(f"Loaded {len(pnls)} historical trades.")
    print(f"Original Total PnL: ${np.sum(pnls):.2f}")
    
    # Original Equity Curve (Historical)
    original_equity = np.cumsum(np.insert(pnls, 0, 0)) + INITIAL_CAPITAL
    
    # Monte Carlo Simulation
    print(f"Running {SIMULATIONS} simulations of {TRADES_PER_SIMULATION} trades each (Resampling with Replacement)...")
    
    final_equities = []
    max_drawdowns = []
    
    plt.figure(figsize=(12, 8))
    
    # Plot a subset of random curves (e.g., 100) to avoid clutter
    for i in range(SIMULATIONS):
        # Resample with replacement to simulate future performance
        simulated_pnls = np.random.choice(pnls, size=TRADES_PER_SIMULATION, replace=True)
        equity_curve = np.cumsum(np.insert(simulated_pnls, 0, 0)) + INITIAL_CAPITAL
        
        final_equities.append(equity_curve[-1])
        
        # Calculate Max Drawdown for this curve
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdowns.append(np.min(drawdown))
        
        if i < 100:
            plt.plot(equity_curve, color='gray', alpha=0.1)

    # --- Statistics Calculation (Moved before plotting to include in chart) ---
    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)
    
    mean_equity = np.mean(final_equities)
    worst_case_equity = np.percentile(final_equities, 1)
    avg_mdd = np.mean(max_drawdowns) * 100
    prob_ruin = np.mean(max_drawdowns < -0.30) * 100
    prob_loss = np.mean(final_equities < INITIAL_CAPITAL) * 100

    # Plot Original Curve (will be shorter)
    plt.plot(original_equity, color='blue', linewidth=2, label='Historical Sample (2022)')
    
    # Add Metrics Text Box
    stats_text = (
        f"Mean Final Equity: ${mean_equity:,.0f}\n"
        f"Worst Case (1%): ${worst_case_equity:,.0f}\n"
        f"Avg Max Drawdown: {avg_mdd:.1f}%\n"
        f"Risk of Ruin (>30% DD): {prob_ruin:.1f}%\n"
        f"Prob. of Loss: {prob_loss:.1f}%"
    )
    
    # Position text box in top left
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f"Monte Carlo Simulation: {TRADES_PER_SIMULATION} Trades Projection")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity ($)")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('monte_carlo_equity_cone.png')
    print("Saved equity cone plot to monte_carlo_equity_cone.png")
    plt.close()
    
    print("\n--- MONTE CARLO RESULTS (Projected 2000 Trades) ---")
    print(f"Mean Final Equity:     ${mean_equity:.2f}")
    print(f"Median Final Equity:   ${np.median(final_equities):.2f}")
    print(f"Std Dev Final Equity:  ${np.std(final_equities):.2f}")
    print("-" * 30)
    print(f"Best Case (99%):       ${np.percentile(final_equities, 99):.2f}")
    print(f"Worst Case (1%):       ${worst_case_equity:.2f}")
    print("-" * 30)
    print("Max Drawdown Estimates:")
    print(f"Average MDD:           {avg_mdd:.2f}%")
    print(f"Worst Case MDD (1%):   {np.percentile(max_drawdowns, 1)*100:.2f}%") 
    
    # Probability of Loss
    print(f"Probability of Loss:   {prob_loss:.2f}%")
    
    # Ruin Probability (Drawdown > 30%)
    print(f"Risk of Ruin (>30% DD): {prob_ruin:.2f}%")
    
    # Ruin Probability (Bankruptcy)
    prob_bankrupt = np.mean(max_drawdowns <= -1.0) * 100
    print(f"Risk of Bankruptcy:    {prob_bankrupt:.2f}%")

if __name__ == "__main__":
    run_monte_carlo()
