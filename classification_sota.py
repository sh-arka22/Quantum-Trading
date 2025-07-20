import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Input
from tensorflow.keras import Model
import pandas as pd
from datetime import timedelta

def build_sequences(df, seq_len, horizon):
    X = []
    y = []
    dates = []
    for i in range(len(df) - seq_len - horizon + 1):
        X.append(df[['Open', 'High', 'Low', 'Close']].iloc[i:i + seq_len].values)
        y.append(df['Close'].iloc[i + seq_len + horizon - 1])
        dates.append(df['Date'].iloc[i + seq_len + horizon - 1])
    return np.array(X), np.array(y), dates

def build_attention_lstm_regressor(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=True)(x)
    attention = Attention()([x, x])
    x = LSTM(50)(attention)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_classification_sota(train_df, test_df, lstm_epochs: int = 10, batch_size: int = 32, forecast_days: int = 30):
    seq_len = 30
    horizon = forecast_days
    
    # Ensure sufficient data
    if len(train_df) < seq_len + horizon + 20 or len(test_df) < seq_len + horizon + 10:
        raise ValueError(f"Insufficient data. Need at least {seq_len + horizon + 20} training rows and {seq_len + horizon + 10} test rows.")
    
    # Build sequences for LSTM
    X_train, y_train, _ = build_sequences(train_df, seq_len, horizon)
    X_test, y_test, test_dates = build_sequences(test_df, seq_len, horizon)
    
    print(f"LSTM training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"LSTM test data shape: X={X_test.shape}, y={y_test.shape}")

    # 1. Attention-LSTM Regressor
    lstm = build_attention_lstm_regressor(input_shape=(seq_len, 4))
    lstm.fit(X_train, y_train,
             epochs=lstm_epochs,
             batch_size=batch_size,
             validation_split=0.1,
             verbose=0)
    
    preds_lstm = lstm.predict(X_test).flatten()

    # 2. LightGBM on engineered features
    import lightgbm as lgb
    import ta
    train_feat = train_df.copy()
    
    # Feature engineering
    for col in ['Close', 'Volume']:
        for span in [5, 10, 20]:
            train_feat[f'{col}_sma_{span}'] = train_feat[col].rolling(span).mean()
            train_feat[f'{col}_rsi_{span}'] = ta.momentum.RSIIndicator(train_feat[col], span).rsi()
    
    # Create target variable BEFORE dropping NAs
    train_feat['target'] = train_feat['Close'].shift(-horizon)
    
    # Drop rows with NaN values
    train_feat = train_feat.dropna()
    
    # Separate features and target
    feats = [c for c in train_feat.columns if c not in ['Date', 'Close', 'target']]
    Xgb = train_feat[feats]
    ygb = train_feat['target']
    
    print(f"LightGBM training data shape: X={Xgb.shape}, y={ygb.shape}")
    
    # Verify lengths match
    if len(Xgb) != len(ygb):
        raise ValueError(f"Feature and target lengths don't match: {len(Xgb)} vs {len(ygb)}")
    
    # Train LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500, 
        max_depth=-1, 
        learning_rate=0.02,
        random_state=42
    )
    lgb_model.fit(Xgb, ygb)

    # Same feature engineering for test data
    test_feat = test_df.copy()
    for col in ['Close', 'Volume']:
        for span in [5, 10, 20]:
            test_feat[f'{col}_sma_{span}'] = test_feat[col].rolling(span).mean()
            test_feat[f'{col}_rsi_{span}'] = ta.momentum.RSIIndicator(test_feat[col], span).rsi()
    
    # Create target for test
    test_feat['target'] = test_feat['Close'].shift(-horizon)
    test_feat = test_feat.dropna()
    
    if len(test_feat) == 0:
        raise ValueError("No test data remaining after feature engineering.")
    
    # Get LightGBM predictions
    preds_lgb = lgb_model.predict(test_feat[feats])
    
    print(f"LSTM predictions length: {len(preds_lstm)}")
    print(f"LightGBM predictions length: {len(preds_lgb)}")
    
    # Align prediction lengths for ensemble
    min_len = min(len(preds_lstm), len(preds_lgb), len(y_test))
    preds_lstm = preds_lstm[:min_len]
    preds_lgb = preds_lgb[:min_len]
    y_test_aligned = y_test[:min_len]
    test_dates = test_dates[:min_len]
    
    print(f"Aligned length for ensemble: {min_len}")

    # 3. Ensemble (simple average)
    ensemble_preds = 0.5 * (preds_lstm + preds_lgb)
    mae = mean_absolute_error(y_test_aligned, ensemble_preds)

    # Convert to binary classification for AUC calculation
    from sklearn.metrics import roc_auc_score
    
    # Create binary targets: 1 if price goes up, 0 if down
    price_changes_actual = np.diff(y_test_aligned) > 0
    price_changes_pred = np.diff(ensemble_preds) > 0
    
    # Calculate AUC if we have enough samples
    if len(price_changes_actual) > 10:
        try:
            pred_proba = (ensemble_preds[1:] - ensemble_preds[:-1]) / ensemble_preds[:-1]
            pred_proba = (pred_proba - pred_proba.min()) / (pred_proba.max() - pred_proba.min())
            auc_score = roc_auc_score(price_changes_actual.astype(int), pred_proba)
        except:
            auc_score = 0.5
    else:
        auc_score = 0.5
    
    # Convert predictions to binary for compatibility with app
    binary_predictions = price_changes_pred.astype(int)
    binary_actual = price_changes_actual.astype(int)
    pred_probabilities = pred_proba if len(price_changes_actual) > 10 else np.random.random(len(binary_predictions))

    # Future forecast
    full_df = pd.concat([train_df, test_df])
    last_seq = full_df[['Open', 'High', 'Low', 'Close']].iloc[-seq_len:].values
    pred_lstm_future = lstm.predict(last_seq[None, :, :])[0][0]

    # LightGBM future
    last_features = test_feat[feats].iloc[-1:]
    pred_lgb_future = lgb_model.predict(last_features)[0]

    ensemble_future = 0.5 * (pred_lstm_future + pred_lgb_future)
    last_close = full_df['Close'].iloc[-1]
    step = (ensemble_future - last_close) / forecast_days
    future_preds = [last_close + step * (i + 1) for i in range(forecast_days)]
    future_dates = [full_df['Date'].max() + timedelta(days=i + 1) for i in range(forecast_days)]

    # Create the graph
    fig = plt.figure(figsize=(12, 6))
    plt.plot(full_df['Date'], full_df['Close'], label='Historical Prices', color='blue')
    plt.plot(future_dates, future_preds, label='Predicted Future Prices', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'Stock Price Prediction for {forecast_days} Days Ahead')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    return {
        'lstm': lstm,
        'lgb': lgb_model,
        'mae': mae,
        'auc': auc_score,
        'pred': binary_predictions,
        'y_true': binary_actual,
        'proba': pred_probabilities,
        'preds': ensemble_preds,
        'y_true_prices': y_test_aligned,
        'forecast': future_preds,
        'future_dates': future_dates,
        'fig': fig
    }