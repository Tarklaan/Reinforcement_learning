import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import joblib
import optuna
from optuna.integration import TFKerasPruningCallback
import os
import pandas as pd
import warnings
import gc
warnings.filterwarnings('ignore')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def load_data(filepath, start_date=None, end_date=None):
    df = pd.read_csv(filepath)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df.set_index('datetime', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
    df.index = df.index.tz_localize(None)
    df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    print(f"✓ Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    return df

def create_sequences(df, window_size):
    sequences, targets = [], []
    for i in range(window_size, len(df)-1):
        window = df.iloc[i-window_size:i]
        last_close = window['Close'].iloc[-1]
        seq = []
        for _, row in window.iterrows():
            features = [
                (row['Open'] - last_close)/last_close,
                (row['High'] - last_close)/last_close,
                (row['Low'] - last_close)/last_close,
                (row['Close'] - last_close)/last_close,
                np.log(row['Volume'] + 1)/10
            ]
            seq.append(features)
        sequences.append(seq)
        
        current_close = df['Close'].iloc[i-1]
        next_close = df['Close'].iloc[i]
        next_high = df['High'].iloc[i]
        next_low = df['Low'].iloc[i]
        
        if next_close > current_close or next_high > current_close:
            targets.append(1)
        else:
            targets.append(-1)
    
    print(f"✓ Created {len(sequences)} sequences")
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

def create_model(trial, window_size, n_features):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    units_l1 = trial.suggest_int('units_l1', 32, 256, step=32)
    dropout_l1 = trial.suggest_float('dropout_l1', 0.1, 0.5)
    use_bidirectional = trial.suggest_categorical('use_bidirectional', [True, False])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(window_size, n_features)))

    if use_bidirectional:
        model.add(Bidirectional(LSTM(units_l1, return_sequences=(n_layers > 1))))
    else:
        model.add(LSTM(units_l1, return_sequences=(n_layers > 1)))
    model.add(Dropout(dropout_l1))

    if n_layers > 1:
        units_l2 = trial.suggest_int('units_l2', 32, 128, step=32)
        dropout_l2 = trial.suggest_float('dropout_l2', 0.1, 0.4)
        model.add(LSTM(units_l2, return_sequences=(n_layers > 2)))
        model.add(Dropout(dropout_l2))
    if n_layers > 2:
        units_l3 = trial.suggest_int('units_l3', 16, 64, step=16)
        dropout_l3 = trial.suggest_float('dropout_l3', 0.1, 0.3)
        model.add(LSTM(units_l3, return_sequences=False))
        model.add(Dropout(dropout_l3))

    dense_units = trial.suggest_int('dense_units', 16, 64, step=16)
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def objective(trial, X_train, X_val, y_train, y_val):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    tf.keras.backend.clear_session()
    gc.collect()
    model = create_model(trial, X_train.shape[1], X_train.shape[2])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=0)
    pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=30, batch_size=batch_size,
                        callbacks=[early_stop, reduce_lr, pruning_callback],
                        verbose=0)
    
    preds = model.predict(X_val, verbose=0).flatten()
    pred_signals = np.where(preds > 0, 1, -1)
    directional_accuracy = np.mean(pred_signals == y_val)
    
    return 1 - directional_accuracy

def train_with_optuna():
    data_path = 'data/btcusdt.csv'
    df = load_data(data_path, start_date='2025-02-01')
    df_5m = df.resample('5min').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    window_size = 50
    X, y = create_sequences(df_5m, window_size)
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, shuffle=False)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=50)

    best_trial = study.best_trial
    final_model = create_model(best_trial, X_train.shape[1], X_train.shape[2])
    batch_size = best_trial.params['batch_size']

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

    X_train_combined = np.concatenate([X_train, X_val])
    y_train_combined = np.concatenate([y_train, y_val])

    history = final_model.fit(X_train_combined, y_train_combined,
                              validation_data=(X_test, y_test),
                              epochs=100,
                              batch_size=batch_size,
                              callbacks=[early_stop, reduce_lr],
                              verbose=1)

    test_loss, test_mae = final_model.evaluate(X_test, y_test, verbose=0)
    predictions = final_model.predict(X_test, verbose=0).flatten()
    pred_signals = np.where(predictions > 0, 1, -1)
    directional_accuracy = np.mean(pred_signals == y_test)
    
    print(f"Test Loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}, Directional Accuracy: {directional_accuracy*100:.2f}%")

    os.makedirs('models', exist_ok=True)
    final_model.save('models/lstm_model.keras')
    joblib.dump({'best_params': best_trial.params, 'directional_accuracy': directional_accuracy}, 'models/lstm_config.pkl')
    return final_model

if __name__ == '__main__':
    train_with_optuna()