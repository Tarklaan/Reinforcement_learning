import numpy as np
import tensorflow as tf
from stable_baselines3 import DQN
import MetaTrader5 as mt5
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, lstm_path='models/lstm_model.keras', rl_path='models/rl_model.zip'):
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"LSTM model not found at {lstm_path}")
        self.lstm = tf.keras.models.load_model(lstm_path, compile=False)
        logger.info("LSTM model loaded successfully")

        if os.path.exists(rl_path):
            self.rl = DQN.load(rl_path)
            logger.info("DQN RL model loaded successfully")
        else:
            self.rl = None
            logger.warning("RL model not found, continuing without it")

        self.lstm_window = 100
        self.position = 0
        self.entry_price = 0.0
        self.position_lot = 0.0
        self.position_ticket = 0
        self.position_age = 0
        self.cum_pnl = 0.0
        self._warm_up_lstm()
        logger.info("Orchestrator initialized successfully")

    def _warm_up_lstm(self):
        dummy = np.zeros((self.lstm_window,5), dtype=np.float32)
        _ = self.lstm.predict(dummy.reshape(1,self.lstm_window,5), verbose=0)
        logger.info("LSTM model warmed up")

    def sync_position_from_mt5(self, symbol):
        try:
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                self.position = 0
                self.entry_price = 0.0
                self.position_lot = 0.0
                self.position_ticket = 0
                self.position_age = 0
                return False
            pos = positions[0]
            self.position = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            self.entry_price = pos.price_open
            self.position_lot = pos.volume
            self.position_ticket = pos.ticket
            self.position_age = int((datetime.now() - datetime.fromtimestamp(pos.time)).total_seconds() // 60)
            if self.position_age < 0:
                self.position_age = 0
            return True
        except Exception as e:
            logger.error(f"Error syncing position: {e}")
            return False

    def get_lstm_signal(self, bars):
        if bars is None or len(bars) < self.lstm_window:
            return 0
        data = np.zeros((self.lstm_window,5), dtype=np.float32)
        for i in range(self.lstm_window):
            bar = bars[-self.lstm_window + i]
            data[i] = [
                float(bar['open']),
                float(bar['high']),
                float(bar['low']),
                float(bar['close']),
                np.log(float(bar['tick_volume'])+1)
            ]
        last_close = data[-1,3]
        data[:,0:4] = (data[:,0:4]-last_close)/last_close
        pred = self.lstm.predict(data.reshape(1,self.lstm_window,5), verbose=0)[0][0]
        return 1 if pred > 0.5 else -1

    def calculate_lot(self, symbol, risk_pct=2.0):
        acc_info = mt5.account_info()
        symbol_info = mt5.symbol_info(symbol)
        if not acc_info or not symbol_info:
            return 0.01
        
        free_margin = acc_info.margin_free
        risk_amount = free_margin * (risk_pct/100)
        
        margin_per_lot = symbol_info.margin_initial
        if margin_per_lot <= 0:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                current_price = tick.ask
                margin_per_lot = current_price * 1.0 / 100
            else:
                margin_per_lot = 500
        
        lot = risk_amount / margin_per_lot
        volume_step = symbol_info.volume_step if symbol_info.volume_step > 0 else 0.01
        volume_min = symbol_info.volume_min if symbol_info.volume_min > 0 else 0.01
        volume_max = symbol_info.volume_max if symbol_info.volume_max > 0 else 100.0
        
        lot = np.floor(lot / volume_step) * volume_step
        lot = max(volume_min, min(lot, volume_max))
        
        logger.info(f"Lot: {lot:.4f} (Free Margin: ${free_margin:.2f}, Risk: ${risk_amount:.2f})")
        return lot

    def send_order(self, symbol, signal, lot):
        if lot <= 0:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        order_type = mt5.ORDER_TYPE_BUY if signal==1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if signal==1 else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "LSTM_RL_Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment}")
            return None
        logger.info(f"Order successful! Ticket: {result.order}, Price: {result.price}")
        return result

    def build_rl_obs(self, m1_bars, current_price, pnl_pct):
        # Build fixed-size observation for DQN: 30 bars x 5 features + 2 extra = 152 -> need 240 for your model
        obs = np.zeros((30,5), dtype=np.float32)
        for i in range(30):
            bar = m1_bars[-30+i] if i<len(m1_bars) else m1_bars[-1]
            obs[i] = [
                float(bar['open']),
                float(bar['high']),
                float(bar['low']),
                float(bar['close']),
                np.log(float(bar['tick_volume'])+1)
            ]
        last_close = obs[-1,3]
        obs[:,0:4] = (obs[:,0:4] - last_close)/last_close
        obs_flat = obs.flatten()
        obs_flat = np.concatenate([obs_flat, np.array([current_price, pnl_pct], dtype=np.float32)])
        # Pad to match expected shape (240,)
        if len(obs_flat) < 240:
            obs_flat = np.pad(obs_flat, (0,240-len(obs_flat)), 'constant')
        elif len(obs_flat) > 240:
            obs_flat = obs_flat[:240]
        return obs_flat
