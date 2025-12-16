import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=30, render_mode=None):
        super().__init__()
        self.df = df.reset_index()
        self.window_size = window_size
        self.render_mode = render_mode
        self.prices = df["Close"].values
        self.max_step = len(self.df) - 1
        
        # Action space: 0=Short, 1=Hold, 2=Long
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, 5), 
            dtype=np.float32
        )
        
        self.position = 0  # -1=short, 0=neutral, 1=long
        self.entry_price = None
        self.trade_cost = 0.0004
        self.initial_balance = 1000
        self.balance = self.initial_balance
        self.trades = []
        
        # For normalization
        self.price_mean = np.mean(self.prices)
        self.price_std = np.std(self.prices)

    def _get_obs(self):
        """Get normalized observation window"""
        window_data = self.df.iloc[
            self.current_step - self.window_size:self.current_step
        ]
        
        # Normalize OHLC by percentage change
        obs = []
        for idx in range(len(window_data)):
            row = window_data.iloc[idx]
            base_price = row["Close"]
            
            normalized_row = [
                (row["Open"] - base_price) / base_price,
                (row["High"] - base_price) / base_price,
                (row["Low"] - base_price) / base_price,
                0.0,  # Close is always 0 (reference point)
                row["Volume"] / (row["Volume"] + 1e-8)  # Normalize volume
            ]
            obs.append(normalized_row)
        
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = None
        self.balance = self.initial_balance
        self.trades = []
        return self._get_obs(), {}

    def step(self, action):
        current_price = self.prices[self.current_step]
        reward = 0
        
        # Map actions: 0=Short, 1=Hold, 2=Long
        target_position = action - 1  # Convert to: -1, 0, 1
        
        # Calculate unrealized P&L if position exists
        if self.position != 0 and self.entry_price is not None:
            if self.position == 1:  # Long position
                unrealized_pnl = current_price - self.entry_price
            else:  # Short position
                unrealized_pnl = self.entry_price - current_price
            
            # Reward for holding: normalized unrealized P&L
            reward = unrealized_pnl / self.entry_price  # Percentage return
        
        # Handle position changes
        if target_position != self.position:
            # Close existing position if any
            if self.position != 0:
                if self.position == 1:  # Close long
                    pnl = current_price - self.entry_price
                else:  # Close short
                    pnl = self.entry_price - current_price
                
                # Apply trading costs
                pnl_with_cost = pnl - abs(pnl) * self.trade_cost
                self.balance += pnl_with_cost
                
                # Normalized reward for closing
                reward = pnl / self.entry_price
                
                self.trades.append({
                    "event": f"{'LONG' if self.position == 1 else 'SHORT'}_CLOSE",
                    "price": current_price,
                    "pnl": pnl_with_cost
                })
            
            # Open new position if not going to hold
            if target_position != 0:
                self.position = target_position
                self.entry_price = current_price
                self.trades.append({
                    "event": f"{'LONG' if target_position == 1 else 'SHORT'}_OPEN",
                    "price": current_price,
                    "pnl": 0
                })
            else:
                self.position = 0
                self.entry_price = None
        
        # Small penalty for being out of the market (encourages participation)
        if self.position == 0:
            reward = -0.001
        
        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= self.max_step
        truncated = False
        
        # Close any open position at end
        if terminated and self.position != 0:
            if self.position == 1:
                pnl = current_price - self.entry_price
            else:
                pnl = self.entry_price - current_price
            pnl_with_cost = pnl - abs(pnl) * self.trade_cost
            self.balance += pnl_with_cost
        
        return self._get_obs(), reward, terminated, truncated, {
            "balance": self.balance,
            "position": self.position
        }

    def export_trades(self):
        return pd.DataFrame(self.trades)