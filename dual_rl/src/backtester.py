import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import tensorflow as tf
from stable_baselines3 import DQN

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    sys.path.append(os.getcwd())

def load_data(filepath, start_date=None, end_date=None):
    df = pd.read_csv(filepath)
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df.set_index('datetime', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
    
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep='first')]
    
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    }, inplace=True)
    
    if start_date:
        start_date = pd.Timestamp(start_date)
        df = df[df.index >= start_date]
    if end_date:
        end_date = pd.Timestamp(end_date)
        df = df[df.index <= end_date]
    
    return df

class LSTMPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.window_size = 100
    
    def predict(self, data_window):
        if len(data_window) < self.window_size:
            return 0
        
        seq = np.zeros((self.window_size, 5), dtype=np.float32)
        for i in range(self.window_size):
            row = data_window.iloc[i]
            seq[i] = [
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                np.log(float(row['Volume']) + 1)
            ]
        
        last_close = seq[-1, 3]
        if last_close > 0:
            seq[:, 0:4] = (seq[:, 0:4] - last_close) / last_close
        
        pred = self.model.predict(seq.reshape(1, self.window_size, 5), verbose=0)[0][0]
        return 1 if pred > 0 else -1

class RLPredictor:
    def __init__(self, model_path):
        self.model = DQN.load(model_path)

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

class Backtester:
    def __init__(self, lstm_path, rl_path, data, initial_balance=10000):
        self.lstm = LSTMPredictor(lstm_path)
        self.rl = RLPredictor(rl_path)
        
        self.data_1m = data.copy()
        self.data_1m = self.data_1m[~self.data_1m.index.duplicated(keep='first')]
        
        self.data_5m = self.data_1m.resample('5min', closed='right', label='right').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.cumulative_pnl = 0.0
        self.cumulative_pnl_pct = 0.0
        self.position = 0
        self.entry_price = 0.0
        self.position_age = 0
        self.position_size = 0
        self.entry_time = None
        
        self.trades = []
        self.equity_curve = []
        self.signals = []
        self.rl_decisions = []
    
    def run(self):
        if len(self.data_5m) < 100:
            print(f"ERROR: Not enough 5-minute data")
            return
        
        print("\nRunning backtest...")
        print("=" * 60)
        
        five_min_times = self.data_5m.index
        
        for i in range(100, len(five_min_times)-1):
            current_5m_time = five_min_times[i]
            
            m5_window = self.data_5m.iloc[i-100:i]
            lstm_signal = self.lstm.predict(m5_window)
            
            if lstm_signal != 0:
                self.signals.append({
                    'time': current_5m_time,
                    'signal': lstm_signal,
                    'price': self.data_5m['Close'].iloc[i]
                })
                
                if self.position != 0 and lstm_signal != self.position:
                    close_data = self.data_1m.loc[:current_5m_time]
                    if not close_data.empty:
                        close_price = close_data['Close'].iloc[-1]
                        self._close_position(close_price, current_5m_time, 'OPPOSITE_SIGNAL')
                
                if self.position == 0:
                    entry_data = self.data_1m.loc[:current_5m_time]
                    if not entry_data.empty:
                        entry_price = entry_data['Close'].iloc[-1]
                        self._open_position(lstm_signal, entry_price, current_5m_time)
            
            if self.position != 0 and i + 1 < len(five_min_times):
                next_5m_time = five_min_times[i+1]
                
                mask = (self.data_1m.index > current_5m_time) & (self.data_1m.index < next_5m_time)
                m1_data = self.data_1m.loc[mask]
                
                if not m1_data.empty:
                    for idx, row in m1_data.iterrows():
                        self.position_age += 1
                        current_price = row['Close']
                        
                        self._update_equity(current_price, idx)
                        
                        recent_mask = (self.data_1m.index >= idx - timedelta(minutes=30)) & (self.data_1m.index <= idx)
                        m1_window = self.data_1m.loc[recent_mask]
                        
                        if len(m1_window) >= 30:
                            obs = self._build_observation(m1_window, current_price)
                            exit_signal = self.rl.predict(obs)
                            
                            self.rl_decisions.append({
                                'time': idx,
                                'exit_signal': exit_signal,
                                'price': current_price,
                                'position_age': self.position_age
                            })
                            
                            if exit_signal == 1:
                                self._close_position(current_price, idx, 'RL_SIGNAL')
                                break
            
            if i % 100 == 0:
                progress = (i - 100) / (len(five_min_times) - 100) * 100
                current_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.balance
                print(f"Progress: {progress:.1f}% | Equity: ${current_equity:.2f} | Trades: {len([t for t in self.trades if t['action'] == 'CLOSE'])}")
        
        if self.position != 0:
            end_price = self.data_1m['Close'].iloc[-1]
            end_time = self.data_1m.index[-1]
            self._close_position(end_price, end_time, 'END')
        
        self._generate_report()
    
    def _build_observation(self, m1_window, current_price):
        obs = np.zeros((30, 5), dtype=np.float32)
        
        for i in range(30):
            if i < len(m1_window):
                row = m1_window.iloc[i]
                obs[i] = [
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    np.log(float(row['Volume']) + 1)
                ]
        
        last_close = obs[-1, 3]
        if last_close > 0:
            obs[:, 0:4] = (obs[:, 0:4] - last_close) / last_close
        
        obs_flat = obs.flatten()
        
        pnl_pct = 0.0
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:
                pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            else:
                pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
        
        obs_flat = np.concatenate([obs_flat, np.array([current_price, pnl_pct], dtype=np.float32)])
        
        if len(obs_flat) < 240:
            obs_flat = np.pad(obs_flat, (0, 240 - len(obs_flat)), 'constant')
        
        return obs_flat
    
    def _open_position(self, side, price, time):
        self.position = side
        self.entry_price = price
        self.entry_time = time
        self.position_age = 0
        
        trade_value = self.balance * 0.5
        self.position_size = trade_value / price
        
        self.trades.append({
            'time': time,
            'action': 'OPEN',
            'side': 'LONG' if side == 1 else 'SHORT',
            'price': price,
            'balance': self.balance,
            'size': self.position_size,
            'position_age': 0
        })
        
        self._update_equity(price, time)
        print(f"[{time}] OPEN {'LONG' if side == 1 else 'SHORT'} @ ${price:.2f}")
    
    def _close_position(self, price, time, reason):
        if self.position == 0 or self.entry_price <= 0:
            return
        
        if self.position == 1:
            pnl = (price - self.entry_price) * self.position_size
            pnl_pct = ((price - self.entry_price) / self.entry_price) * 100
        else:
            pnl = (self.entry_price - price) * self.position_size
            pnl_pct = ((self.entry_price - price) / self.entry_price) * 100
        
        trading_cost = abs(pnl) * 0.0004
        pnl -= trading_cost
        self.balance += pnl
        self.cumulative_pnl += pnl
        self.cumulative_pnl_pct += pnl_pct
        
        self.trades.append({
            'time': time,
            'action': 'CLOSE',
            'side': 'LONG' if self.position == 1 else 'SHORT',
            'price': price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'balance': self.balance,
            'cumulative_pnl': self.cumulative_pnl,
            'cumulative_pnl_pct': self.cumulative_pnl_pct,
            'reason': reason,
            'duration': self.position_age,
            'entry_price': self.entry_price,
            'exit_price': price
        })
        
        self._update_equity(price, time)
        
        print(f"[{time}] CLOSE {'LONG' if self.position == 1 else 'SHORT'} "
              f"@ ${price:.2f}, PnL: ${pnl:.2f} ({pnl_pct:.2f}%), "
              f"Cum PnL: ${self.cumulative_pnl:.2f}, Reason: {reason}")
        
        self.position = 0
        self.entry_price = 0.0
        self.position_age = 0
        self.position_size = 0
    
    def _update_equity(self, price, time):
        equity = self.balance
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:
                unrealized = (price - self.entry_price) * self.position_size
            else:
                unrealized = (self.entry_price - price) * self.position_size
            equity += unrealized
        
        self.equity_curve.append({
            'time': time,
            'equity': equity,
            'price': price,
            'position': self.position,
            'unrealized': equity - self.balance if self.position != 0 else 0
        })
    
    def _generate_report(self):
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        print(f"\n{'='*60}")
        print("BACKTEST COMPLETE")
        print(f"{'='*60}")
        
        if not trades_df.empty:
            closing_trades = trades_df[trades_df['action'] == 'CLOSE']
            
            if not closing_trades.empty:
                total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
                winning_trades = closing_trades[closing_trades['pnl'] > 0]
                losing_trades = closing_trades[closing_trades['pnl'] <= 0]
                
                win_rate = len(winning_trades) / len(closing_trades) * 100
                avg_win = winning_trades['pnl'].mean()
                avg_loss = losing_trades['pnl'].mean()
                
                profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty else float('inf')
                
                print(f"Initial Balance: ${self.initial_balance:,.2f}")
                print(f"Final Balance: ${self.balance:,.2f}")
                print(f"Total Return: {total_return:.2f}%")
                print(f"Cumulative PnL: ${self.cumulative_pnl:.2f}")
                print(f"Cumulative PnL%: {self.cumulative_pnl_pct:.2f}%")
                print(f"\nTrade Statistics:")
                print(f"  Total Trades: {len(closing_trades)}")
                print(f"  Win Rate: {win_rate:.1f}%")
                print(f"  Profit Factor: {profit_factor:.2f}")
                print(f"  Avg Win: ${avg_win:.2f}")
                print(f"  Avg Loss: ${avg_loss:.2f}")
                
                if 'reason' in closing_trades.columns:
                    reason_counts = closing_trades['reason'].value_counts()
                    print(f"\nExit Reasons:")
                    for reason, count in reason_counts.items():
                        print(f"  {reason}: {count}")
        
        os.makedirs('backtest_results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not trades_df.empty:
            trades_df.to_csv(f'backtest_results/trades_{timestamp}.csv', index=False)
        
        if not equity_df.empty:
            equity_df.to_csv(f'backtest_results/equity_{timestamp}.csv', index=False)
        
        print(f"\nResults saved to backtest_results/")

def run_backtest():
    data_path = 'data/btcusdt_clean.csv'
    lstm_path = 'models/lstm_model.keras'
    rl_path = 'models/rl_model.zip'
    
    data = load_data(data_path, start_date='2024-06-01', end_date='2024-06-10')
    
    backtester = Backtester(lstm_path, rl_path, data, initial_balance=10000)
    backtester.run()

if __name__ == '__main__':
    run_backtest()