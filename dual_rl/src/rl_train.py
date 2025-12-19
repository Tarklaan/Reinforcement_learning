import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import tensorflow as tf
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class LSTM_RL_TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df_1m, df_5m, lstm_predictor, max_trade_duration=120):
        super().__init__()
        
        self.df_1m = df_1m.reset_index(drop=True)
        self.df_5m = df_5m.reset_index(drop=True)
        self.lstm = lstm_predictor
        
        self.prices_1m = self.df_1m['Close'].values
        self.prices_5m = self.df_5m['Close'].values
        
        self.action_space = spaces.Discrete(2)  # 0: HOLD, 1: EXIT
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(240,),  # MUST match your Orchestrator's build_rl_obs output
            dtype=np.float32
        )
        
        self.max_trade_duration = max_trade_duration
        self.trade_cost = 0.0004
        self.reset()
    
    def _build_observation(self):
        if self.current_step < 30:
            return np.zeros(240, dtype=np.float32)
        
        obs = np.zeros((30, 5), dtype=np.float32)
        
        start_idx = self.current_step - 30
        for i in range(30):
            idx = start_idx + i
            row = self.df_1m.iloc[idx]
            obs[i] = [
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                np.log(float(row['Volume']) + 1)
            ]
        
        last_close = obs[-1, 3]
        if last_close != 0:
            obs[:, 0:4] = (obs[:, 0:4] - last_close) / last_close
        
        obs_flat = obs.flatten()
        
        current_price = self.prices_1m[self.current_step]
        pnl_pct = self._calculate_pnl_percent(current_price)
        
        obs_flat = np.concatenate([obs_flat, np.array([current_price, pnl_pct], dtype=np.float32)])
        
        if len(obs_flat) < 240:
            obs_flat = np.pad(obs_flat, (0, 240 - len(obs_flat)), 'constant')
        elif len(obs_flat) > 240:
            obs_flat = obs_flat[:240]
        
        return obs_flat
    
    def _calculate_pnl_percent(self, current_price):
        if self.position == 0 or self.entry_price == 0:
            return 0.0
        
        if self.position == 1:  # LONG
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - current_price) / self.entry_price) * 100
    
    def _get_5m_signal_for_step(self, step_1m_idx):
        if step_1m_idx < 0 or step_1m_idx >= len(self.df_1m):
            return 0
        
        ts_1m = self.df_1m.index[step_1m_idx]
        ts_5m = ts_1m.floor('5min')
        
        if ts_5m not in self.df_5m.index:
            return 0
        
        idx_5m = self.df_5m.index.get_loc(ts_5m)
        if idx_5m < 100:
            return 0
        
        window_5m = self.df_5m.iloc[idx_5m-100:idx_5m]
        return self.lstm.predict(window_5m)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 100
        self.position = 0
        self.entry_price = 0.0
        self.position_age = 0
        self.entry_signal = 0
        
        while self.current_step < len(self.df_1m) - self.max_trade_duration - 10 and self.position == 0:
            signal = self._get_5m_signal_for_step(self.current_step)
            if signal != 0:
                self.position = signal
                self.entry_price = self.prices_1m[self.current_step]
                self.position_age = 1
                self.entry_signal = signal
                break
            self.current_step += 5
        
        if self.position == 0:
            self.current_step = len(self.df_1m) - 1
        
        return self._build_observation(), {}
    
    def step(self, action):
        if self.current_step >= len(self.df_1m) - 1:
            return self._build_observation(), 0.0, True, False, {}
        
        self.position_age += 1
        current_price = self.prices_1m[self.current_step]
        reward = 0.0
        terminated = False
        close_reason = ""
        
        pnl_pct = self._calculate_pnl_percent(current_price)
        
        current_signal = self._get_5m_signal_for_step(self.current_step)
        
        forced_close = False
        
        if current_signal != 0 and current_signal != self.position:
            forced_close = True
            close_reason = "OPPOSITE_SIGNAL"
            reward = pnl_pct * 0.1
        elif pnl_pct <= -1.5:
            forced_close = True
            close_reason = "STOP_LOSS"
            reward = pnl_pct
        elif pnl_pct >= 3.0:
            forced_close = True
            close_reason = "PROFIT_TARGET"
            reward = pnl_pct
        elif self.position_age > self.max_trade_duration:
            forced_close = True
            close_reason = "TIMEOUT"
            reward = pnl_pct * 0.5
        
        if forced_close:
            terminated = True
        
        elif action == 1:
            close_reason = "RL_EXIT"
            reward = pnl_pct
            terminated = True
        
        else:
            reward = pnl_pct * 0.01
        
        if terminated:
            self.position = 0
            self.entry_price = 0.0
            self.position_age = 0
        
        info = {
            'position': self.position,
            'pnl_pct': pnl_pct,
            'age': self.position_age,
            'close_reason': close_reason if terminated else "HOLDING",
            'price': current_price,
            'step': self.current_step
        }
        
        self.current_step += 1
        
        return self._build_observation(), reward, terminated, False, info

def train_rl():
    print("=== DQN TRAINING WITH LSTM SIGNALS ===")
    
    lstm_path = 'models/lstm_model.keras'
    
    df_1m = load_data('data/btcusdt_clean.csv', '2024-06-01', '2024-07-31')
    
    df_5m = df_1m.resample('5min', closed='right', label='right').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    lstm_predictor = LSTMPredictor(lstm_path)
    
    def make_env():
        return LSTM_RL_TradingEnv(df_1m.copy(), df_5m.copy(), lstm_predictor, max_trade_duration=120)
    
    env = DummyVecEnv([make_env])
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=128,
        gamma=0.95,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        verbose=1
    )
    
    print("Starting training...")
    model.learn(total_timesteps=200_000, progress_bar=True, log_interval=1000)
    
    model.save("models/rl_model")
    print("Model saved as 'models/rl_model.zip'")
    
    env.close()