import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_trades(price_data, trades_df, title="Trade Visualization"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1.plot(price_data.index, price_data['Close'], label='Price', alpha=0.7, linewidth=1)
    
    buy_signals = trades_df[trades_df['action'] == 'OPEN'][trades_df['side'] == 'LONG']
    sell_signals = trades_df[trades_df['action'] == 'OPEN'][trades_df['side'] == 'SHORT']
    exit_points = trades_df[trades_df['action'] == 'CLOSE']
    
    if not buy_signals.empty:
        ax1.scatter(buy_signals['time'], buy_signals['price'], 
                   color='green', marker='^', s=100, label='Buy', alpha=0.8)
    
    if not sell_signals.empty:
        ax1.scatter(sell_signals['time'], sell_signals['price'], 
                   color='red', marker='v', s=100, label='Sell', alpha=0.8)
    
    if not exit_points.empty:
        ax1.scatter(exit_points['time'], exit_points['price'], 
                   color='blue', marker='x', s=50, label='Exit', alpha=0.6)
    
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if 'equity' in trades_df.columns:
        equity_df = trades_df.dropna(subset=['equity'])
        if not equity_df.empty:
            ax2.plot(equity_df['time'], equity_df['equity'], 
                    color='green', linewidth=2, label='Portfolio Equity')
            ax2.axhline(y=equity_df['equity'].iloc[0], color='red', 
                       linestyle='--', alpha=0.5, label='Initial')
            ax2.set_ylabel('Equity ($)')
            ax2.set_xlabel('Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig