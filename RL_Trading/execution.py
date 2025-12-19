import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from stable_baselines3 import PPO
from trading_env import TradingEnv

MT5_LOGIN = 245162
MT5_PASSWORD = "Khan@1122"
MT5_SERVER = "FusionMarkets-Demo"

if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
    print("MT5 initialization failed")
    mt5.shutdown()
    exit()
print(f"Connected to MT5 account {MT5_LOGIN} on server {MT5_SERVER}")

symbol = "BTCUSD"
window_size = 30
lot_size = 0.01
magic_number = 123456
min_profit = 0.5
signal_interval = 60
check_interval = 30

if not mt5.symbol_select(symbol, True):
    print(f"Failed to select {symbol}")
    mt5.shutdown()
    exit()

symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    print(f"Symbol {symbol} not found")
    mt5.shutdown()
    exit()

if not symbol_info.visible:
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to enable {symbol}")
        mt5.shutdown()
        exit()

print(f"Symbol: {symbol}")
print(f"Volume min: {symbol_info.volume_min}")
print(f"Volume max: {symbol_info.volume_max}")
print(f"Volume step: {symbol_info.volume_step}")

lot_size = max(symbol_info.volume_min, lot_size)
lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step

print(f"Using lot size: {lot_size}")

model = PPO.load("models/ppo_trading_model_final")

def get_filling_mode(symbol_info):
    filling = symbol_info.filling_mode
    if filling & 1:
        return mt5.ORDER_FILLING_FOK
    elif filling & 2:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN

filling_mode = get_filling_mode(symbol_info)
print(f"Using filling mode: {filling_mode}")

def get_latest_candles(symbol, n=window_size):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    return df

def get_max_positions():
    info = mt5.account_info()
    if info is None:
        return 0
    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if symbol_info is None or tick is None:
        return 0
    
    free_margin = info.margin_free
    contract_size = symbol_info.trade_contract_size
    leverage = info.leverage if info.leverage > 0 else 1
    price = tick.ask
    
    margin_per_lot = (lot_size * contract_size * price) / leverage
    if margin_per_lot == 0:
        return 0
    
    max_positions = int(free_margin / margin_per_lot)
    return max(0, max_positions)

def close_profitable_positions():
    positions = mt5.positions_get(symbol=symbol, magic=magic_number)
    if not positions:
        return 0
    
    closed_count = 0
    for pos in positions:
        profit = pos.profit
        if profit >= min_profit:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": magic_number,
                "comment": "Close profit",
                "type_filling": filling_mode,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Closed position {pos.ticket} with profit: ${profit:.2f}")
                closed_count += 1
    
    return closed_count

def close_all_positions():
    positions = mt5.positions_get(symbol=symbol, magic=magic_number)
    if not positions:
        return
    
    for pos in positions:
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": magic_number,
            "comment": "Close all",
            "type_filling": filling_mode,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Closed position {pos.ticket} with profit: ${pos.profit:.2f}")

def open_position(action):
    order_type = mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("No tick data")
        return False
        
    price = tick.ask if action == 1 else tick.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": magic_number,
        "comment": "RL scalper",
        "type_filling": filling_mode,
    }
    
    result = mt5.order_send(request)
    if result is None:
        print("order_send returned None")
        return False
        
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: retcode={result.retcode}, comment={result.comment}")
        return False
    else:
        return True

running = True
last_signal_time = 0

while running:
    try:
        current_time = time.time()
        
        if current_time - last_signal_time >= signal_interval:
            close_all_positions()
            
            df_obs = get_latest_candles(symbol, window_size)
            if len(df_obs) < window_size:
                time.sleep(1)
                continue

            obs = []
            for idx, row in df_obs.iterrows():
                base = row["Close"]
                obs.append([
                    (row["Open"] - base) / base,
                    (row["High"] - base) / base,
                    (row["Low"] - base) / base,
                    0.0,
                    np.log(row["Volume"] + 1)
                ])
            obs = np.array([obs], dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action[0])
            signal = "BUY" if action == 1 else "SELL"
            print(f"\n{'='*50}")
            print(f"NEW SIGNAL: {signal}")
            print(f"{'='*50}")

            max_positions = get_max_positions()
            print(f"Opening {max_positions} positions...")
            
            opened = 0
            for i in range(max_positions):
                if open_position(action):
                    opened += 1
                time.sleep(0.1)
            
            print(f"Opened {opened} positions")
            last_signal_time = current_time
        
        else:
            time_since_signal = current_time - last_signal_time
            if time_since_signal >= check_interval:
                positions = mt5.positions_get(symbol=symbol, magic=magic_number)
                if positions:
                    print(f"\nChecking {len(positions)} positions for profit...")
                    closed = close_profitable_positions()
                    if closed > 0:
                        print(f"Closed {closed} profitable positions")
        
        time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nStopping live trading...")
        close_all_positions()
        running = False
    except Exception as e:
        print("Error:", e)
        time.sleep(1)

mt5.shutdown()