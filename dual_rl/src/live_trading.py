import MetaTrader5 as mt5
import time
from datetime import datetime
import os
import sys
import traceback
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.orchestrator import Orchestrator

def live_trading(symbol='BTCUSD', account=245162, password='Khan@1122', server='FusionMarkets-Demo'):
    logger.info("=== STARTING LIVE TRADING ===")
    if not mt5.initialize():
        logger.error(f"MT5 initialize failed, error code: {mt5.last_error()}")
        return
    if not mt5.login(account, password, server):
        logger.error(f"Login failed, error code: {mt5.last_error()}")
        mt5.shutdown()
        return
    logger.info(f"Connected to account #{account}")
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Failed to select {symbol}")
        mt5.shutdown()
        return
    logger.info(f"Symbol {symbol} selected")
    
    orchestrator = Orchestrator()
    last_m5_time = None
    last_m1_time = None
    
    try:
        while True:
            current_time = datetime.now()
            # 5-min bars
            m5_bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 105)
            if m5_bars is not None and len(m5_bars) >= 100:
                m5_time = datetime.fromtimestamp(m5_bars[-1]['time'])
                if last_m5_time is None or m5_time > last_m5_time:
                    last_m5_time = m5_time
                    signal = orchestrator.get_lstm_signal(m5_bars)
                    print(f"[{current_time}] LSTM Signal: {signal}, Current Position: {orchestrator.position}")
                    if orchestrator.position == 0 and signal != 0:
                        lot = orchestrator.calculate_lot(symbol, risk_pct=2.0)  # 2% per trade
                        print(f"Opening {'LONG' if signal==1 else 'SHORT'} Trade with lot {lot}")
                        if lot > 0:
                            result = orchestrator.send_order(symbol, signal, lot)
                            if result:
                                orchestrator.sync_position_from_mt5(symbol)
            orchestrator.sync_position_from_mt5(symbol)

            # 1-min RL exit check
            if orchestrator.position != 0:
                m1_bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 35)
                tick = mt5.symbol_info_tick(symbol)
                if m1_bars is not None and tick is not None:
                    current_price = tick.bid if orchestrator.position == 1 else tick.ask
                    positions = mt5.positions_get(symbol=symbol)
                    if positions:
                        pnl = positions[0].profit
                        entry_price = positions[0].price_open
                        pnl_pct = (pnl / (entry_price * positions[0].volume)) * 100
                        # Hard stops
                        if pnl_pct <= -1.0 or pnl_pct >= 2.0 or orchestrator.position_age > 120:
                            orchestrator.send_order(symbol, -orchestrator.position, positions[0].volume)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Trading stopped by user")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    live_trading()
