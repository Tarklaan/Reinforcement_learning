import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df, window_size=30):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.prices = df['Close'].values
        self.max_step = len(self.df) - 1
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(240,),  # Changed to match RL model
            dtype=np.float32
        )
        
        self.trade_cost = 0.0004
        self.initial_balance = 10000
        self.reset()
    
    def _get_obs(self):
        if self.current_step < self.window_size:
            return np.zeros(240, dtype=np.float32)
        
        obs = np.zeros((30, 5), dtype=np.float32)
        
        start_idx = max(0, self.current_step - 30)
        for i in range(30):
            idx = start_idx + i
            if idx >= len(self.df):
                idx = len(self.df) - 1
            
            row = self.df.iloc[idx]
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
        
        current_price = self.prices[self.current_step]
        pnl_pct = 0.0
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:
                pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            else:
                pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
        
        obs_flat = np.concatenate([obs_flat, np.array([current_price, pnl_pct], dtype=np.float32)])
        
        if len(obs_flat) < 240:
            obs_flat = np.pad(obs_flat, (0, 240 - len(obs_flat)), 'constant')
        elif len(obs_flat) > 240:
            obs_flat = obs_flat[:240]
        
        return obs_flat
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.position_age = 0
        self.balance = self.initial_balance
        self.cumulative_pnl = 0.0
        self.cumulative_pnl_pct = 0.0
        self.trades = []
        return self._get_obs(), {}
    
    def step(self, action):
        if self.current_step >= len(self.prices):
            return self._get_obs(), 0.0, True, False, {}
        
        current_price = self.prices[self.current_step]
        reward = 0.0
        
        if self.position != 0:
            self.position_age += 1
        
        if action == 1 and self.position != 0:
            if self.position == 1:
                raw_pnl = current_price - self.entry_price
            else:
                raw_pnl = self.entry_price - current_price
            
            realized_pnl = raw_pnl - abs(raw_pnl) * self.trade_cost
            pnl_pct = (realized_pnl / self.entry_price) * 100
            reward = pnl_pct
            self.balance += realized_pnl
            self.cumulative_pnl += realized_pnl
            self.cumulative_pnl_pct += pnl_pct
            
            self.trades.append({
                'step': self.current_step,
                'action': 'CLOSE',
                'side': 'LONG' if self.position == 1 else 'SHORT',
                'price': current_price,
                'pnl': realized_pnl,
                'pnl_pct': pnl_pct,
                'cumulative_pnl': self.cumulative_pnl,
                'cumulative_pnl_pct': self.cumulative_pnl_pct,
                'reward': reward
            })
            
            self.position = 0
            self.entry_price = 0.0
            self.position_age = 0
        
        self.current_step += 1
        terminated = self.current_step >= self.max_step
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'position_age': self.position_age,
            'cumulative_pnl': self.cumulative_pnl,
            'cumulative_pnl_pct': self.cumulative_pnl_pct
        }
        
        return self._get_obs(), reward, terminated, False, info
    
    def open_position(self, side, price):
        self.position = side
        self.entry_price = price
        self.position_age = 1
        
        self.trades.append({
            'step': self.current_step,
            'action': 'OPEN',
            'side': 'LONG' if side == 1 else 'SHORT',
            'price': price
        })