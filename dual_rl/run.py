import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("=== DUAL-MODEL TRADING SYSTEM ===")
    print("1. Train LSTM model")
    print("2. Train RL model")
    print("3. Run backtest")
    print("4. Run live trading")
    print("5. Exit")
    
    choice = input("\nSelect option: ")
    
    if choice == '1':
        from src.lstm_train import train_lstm
        train_lstm()
    
    elif choice == '2':
        from src.rl_train import train_rl
        train_rl()
    
    elif choice == '3':
        from src.backtester import run_backtest
        run_backtest()
    
    elif choice == '4':
        from src.live_trading import live_trading
        live_trading()
    
    elif choice == '5':
        sys.exit()
    
    else:
        print("Invalid choice")

if __name__ == '__main__':
    main()