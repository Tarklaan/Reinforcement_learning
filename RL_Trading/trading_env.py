import gymnasium as gym      
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=30):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.prices = df["Close"].values
        self.max_step = len(self.df) - 1

        # 0 = Short, 1 = Long
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, 5),
            dtype=np.float32
        )

        self.trade_cost = 0.0004
        self.initial_balance = 1000

        self.reset()

    def _get_obs(self):
        window = self.df.iloc[
            self.current_step - self.window_size:self.current_step
        ]

        obs = []
        for _, row in window.iterrows():
            base = row["Close"]
            obs.append([
                (row["Open"] - base) / base,
                (row["High"] - base) / base,
                (row["Low"] - base) / base,
                0.0,
                np.log(row["Volume"] + 1)
            ])
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position = 1   # Start LONG
        self.entry_price = self.prices[self.current_step]
        self.balance = self.initial_balance
        self.trades = []
        return self._get_obs(), {}

    def step(self, action):
        current_price = self.prices[self.current_step]
        reward = 0.0

        target_position = 1 if action == 1 else -1

        # Close existing position
        pnl = (
            current_price - self.entry_price
            if self.position == 1
            else self.entry_price - current_price
        )

        pnl -= abs(pnl) * self.trade_cost
        reward = pnl / self.entry_price
        self.balance += pnl

        self.trades.append({
            "event": "CLOSE",
            "side": "LONG" if self.position == 1 else "SHORT",
            "price": current_price,
            "pnl": pnl
        })

        # Open new position immediately
        self.position = target_position
        self.entry_price = current_price

        self.trades.append({
            "event": "OPEN",
            "side": "LONG" if target_position == 1 else "SHORT",
            "price": current_price,
            "pnl": 0.0
        })

        self.current_step += 1
        terminated = self.current_step >= self.max_step

        return self._get_obs(), reward, terminated, False, {
            "balance": self.balance,
            "position": self.position
        }

    def export_trades(self):
        return pd.DataFrame(self.trades)
