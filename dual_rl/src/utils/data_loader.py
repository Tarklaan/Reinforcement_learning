import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath, start_date=None, end_date=None):
    df = pd.read_csv(filepath)
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    }, inplace=True)
    
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    
    return df

def create_sequences(df, window_size):
    sequences = []
    targets = []
    
    for i in range(window_size, len(df)-1):
        window = df.iloc[i-window_size:i]
        last_close = window['Close'].iloc[-1]
        
        seq = []
        for _, row in window.iterrows():
            features = [
                (row['Open'] - last_close) / last_close,
                (row['High'] - last_close) / last_close,
                (row['Low'] - last_close) / last_close,
                (row['Close'] - last_close) / last_close,
                np.log(row['Volume'] + 1)
            ]
            seq.append(features)
        
        sequences.append(seq)
        target = 1 if df['Close'].iloc[i] > df['Close'].iloc[i-1] else -1
        targets.append(target)
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)