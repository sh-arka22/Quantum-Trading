import warnings
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error
from neuralprophet import set_log_level

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrameGroupBy.apply operated on the grouping columns")
warnings.filterwarnings("ignore", category=FutureWarning, message="Series.view is deprecated")

set_log_level("ERROR")

def build_np_model(epochs: int = 50, n_lags: int = 10, ar_layers_config: list = [8, 8], batch_size: int = 32):
    m = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        n_lags=n_lags,
        ar_layers=ar_layers_config,
        epochs=epochs,
        normalize='soft',
        impute_missing=True,
        batch_size=batch_size
    )
    return m

def calculate_optimal_lags(df_length, n_lags):
    # Simple check: n_lags must be smaller than dataset length
    if n_lags >= df_length:
        raise ValueError(f"n_lags ({n_lags}) must be smaller than dataset length ({df_length})")
    return n_lags

def train_regression_sota(train_df, test_df, forecast_days, epochs=50, n_lags=10, ar_layers=None, batch_size=32):
    if ar_layers is None:
        ar_layers = [8, 8]

    # Rename columns
    train_np = train_df.rename(columns={'Date': 'ds', 'Close': 'y'})[['ds', 'y']]
    test_np = test_df.rename(columns={'Date': 'ds', 'Close': 'y'})[['ds', 'y']]

    if train_np.empty or test_np.empty:
        raise ValueError("One of the DataFrames is empty after renaming.")

    train_np = train_np.drop_duplicates(subset=['ds']).sort_values('ds')
    test_np = test_np.drop_duplicates(subset=['ds']).sort_values('ds')

    if train_np.empty or test_np.empty:
        raise ValueError("One of the DataFrames is empty after deduplication.")

    total_length = len(train_np)
    
    # Simple validation: only check if n_lags < dataset length
    final_lags = calculate_optimal_lags(total_length, n_lags)  # This just validates and returns n_lags

    # Adjust for better LR finder (increase effective batches if possible)
    if total_length < 226 * 32:  # Minimum for LR finder
        print("Warning: Dataset small for optimal LR finding. Consider adding more data.")
    
    m = build_np_model(epochs, n_lags=final_lags, ar_layers_config=ar_layers, batch_size=batch_size)
    
    try:
        metrics = m.fit(train_np, freq='D', progress="none")
        print(metrics)
        
        # Predict on full data to get test predictions with history
        full_np = pd.concat([train_np, test_np])
        full_forecast = m.predict(full_np)
        
        # Extract test part
        test_forecast = full_forecast[full_forecast['ds'].isin(test_np['ds'])]
        
        if len(test_forecast) > 0 and 'yhat1' in test_forecast.columns:
            test_merged = test_np.merge(test_forecast[['ds', 'yhat1']], on='ds', how='inner')
            
            if len(test_merged) > 0:
                try:
                    test_mae = mean_absolute_error(test_merged['y'], test_merged['yhat1'])
                    direction_acc = 0.0
                    if len(test_merged) > 1:
                        actual_changes = np.diff(test_merged['y'].values) > 0
                        pred_changes = np.diff(test_merged['yhat1'].values) > 0
                        direction_acc = np.mean(actual_changes == pred_changes)
                except Exception as e:
                    test_mae = float('inf')
                    direction_acc = 0.0
                    print(f"Error calculating metrics: {e}")
            else:
                test_mae = float('inf')
                direction_acc = 0.0
        else:
            test_mae = float('inf')
            direction_acc = 0.0
        
        future = m.make_future_dataframe(train_np, periods=forecast_days, n_historic_predictions=True)
        forecast = m.predict(future)
        
        # Post-processing: interpolate any missing yhat1, fill remaining with ffill/bfill
        forecast['yhat1'] = (
            forecast['yhat1']
            .interpolate(method='linear')
            .fillna(method='bfill')
            .fillna(method='ffill')
        )
        # zero-fill any missing components
        components = ['ar1', 'trend', 'season_yearly', 'season_weekly', 'season_daily']
        for col in components:
            if col in forecast.columns:
                forecast[col] = forecast[col].fillna(0)
        
        # take last forecast_days rows and pad if needed
        future_predictions = forecast.tail(forecast_days)

        if len(future_predictions) < forecast_days:
            last_val = future_predictions['yhat1'].iloc[-1] if len(future_predictions) > 0 else 0.0
            pad = pd.DataFrame({
                'yhat1': [last_val] * (forecast_days - len(future_predictions))
            })
            future_predictions = pd.concat([future_predictions, pad], ignore_index=True)

        future_predictions_values = future_predictions['yhat1'].values
        
        return {
            'forecast': future_predictions_values,
            'mae_test': test_mae,
            'direction_accuracy': direction_acc,
            'full_forecast': future_predictions,
            'n_lags_used': final_lags
        }
        
    except Exception as e:
        if "less than n_forecasts + n_lags" in str(e):
            # Simple fallback: use much smaller lags, ensure still valid
            fallback_lags = max(1, min(5, total_length - 10))
            if fallback_lags >= total_length:
                fallback_lags = max(1, total_length // 2)
            
            fallback_forecast = min(forecast_days, total_length // 5)
            
            m_fallback = build_np_model(epochs, n_lags=fallback_lags, ar_layers_config=ar_layers)
            metrics = m_fallback.fit(train_np, freq='D', progress="none")
            
            # Predict on full for test
            full_np = pd.concat([train_np, test_np])
            full_forecast = m_fallback.predict(full_np)
            test_forecast = full_forecast[full_forecast['ds'].isin(test_np['ds'])]
            
            if len(test_forecast) > 0 and 'yhat1' in test_forecast.columns:
                test_merged = test_np.merge(test_forecast[['ds', 'yhat1']], on='ds', how='inner')
                
                if len(test_merged) > 0:
                    try:
                        test_mae = mean_absolute_error(test_merged['y'], test_merged['yhat1'])
                        direction_acc = 0.0
                        if len(test_merged) > 1:
                            actual_changes = np.diff(test_merged['y'].values) > 0
                            pred_changes = np.diff(test_merged['yhat1'].values) > 0
                            direction_acc = np.mean(actual_changes == pred_changes)
                    except Exception as e:
                        test_mae = float('inf')
                        direction_acc = 0.0
                        print(f"Error calculating metrics: {e}")
                else:
                    test_mae = float('inf')
                    direction_acc = 0.0
            else:
                test_mae = float('inf')
                direction_acc = 0.0
            
            future = m_fallback.make_future_dataframe(train_np, periods=fallback_forecast)
            forecast = m_fallback.predict(future)
            
            # Apply same post-processing as main path
            forecast['yhat1'] = (
                forecast['yhat1']
                .interpolate(method='linear')
                .fillna(method='bfill')
                .fillna(method='ffill')
            )
            
            components = ['ar1', 'trend', 'season_yearly', 'season_weekly', 'season_daily']
            for col in components:
                if col in forecast.columns:
                    forecast[col] = forecast[col].fillna(0)
            
            future_predictions = forecast.tail(fallback_forecast)
            # Pad to match requested forecast_days
            if len(future_predictions) < forecast_days:
                last_val = future_predictions['yhat1'].iloc[-1] if len(future_predictions) > 0 else 0.0
                pad = pd.DataFrame({
                    'yhat1': [last_val] * (forecast_days - len(future_predictions))
                })
                future_predictions = pd.concat([future_predictions, pad], ignore_index=True)
            
            future_predictions_values = future_predictions['yhat1'].values[:forecast_days]
            
            return {
                'forecast': future_predictions_values,
                'mae_test': test_mae,
                'direction_accuracy': direction_acc,
                'full_forecast': future_predictions,
                'n_lags_used': fallback_lags,
                'warning': 'Used fallback parameters due to data limitations'
            }
        else:
            raise e