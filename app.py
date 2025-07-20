# app.py – Complete Stock Forecasting Application with All Fixes

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf

# Clear TensorFlow session to prevent memory issues
tf.keras.backend.clear_session()

# Local module imports
from yfinance_utils import fetch_stock
from regression_sota import train_regression_sota
from classification_sota import train_classification_sota
from agents_sota import (
    RSIMeanReversion,
    MACDTrend,
    BollingerBreak,
    ATRTrailing,
    run_backtest
)

# GPU Configuration and Detection
def setup_gpu():
    """Configure GPU if available and provide user feedback"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            st.sidebar.success(f"✅ Using GPU: {gpus[0].name}")
            return True
        except RuntimeError as e:
            st.sidebar.warning(f"⚠️ GPU config failed: {e}")
            return False
    else:
        st.sidebar.info("⚠️ No GPU found, using CPU")
        return False

# Application Configuration
st.set_page_config(
    page_title="Quantum Trading",
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title
st.title("📈 Stock Forecasting & Trading Platform")
st.markdown("*Advanced Machine Learning for Financial Markets*")

# Setup GPU
gpu_available = setup_gpu()

# Sidebar Configuration
st.sidebar.header("🎯 Model Configuration")

# 1. Mode Selection
mode = st.sidebar.radio(
    "🤖 Select AI Mode", 
    ["Regression", "Classification"],
    help="Regression: Predict future prices | Classification: Predict up/down movements"
)

# 2. Data Selection Parameters
st.sidebar.subheader("🔍 Data Configuration")

ticker = st.sidebar.text_input(
    "Stock Ticker", 
    "AAPL",
    help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)"
).upper()

train_years = st.sidebar.slider(
    "Training Years", 
    min_value=1, 
    max_value=4, 
    value=2,
    help="Years of historical data for training"
)

test_months = st.sidebar.slider(
    "Test Period (months)", 
    min_value=1, 
    max_value=3, 
    value=2,
    help="Hold-out period for model validation"
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon (days)", 
    min_value=5, 
    max_value=30, 
    value=15,
    help="Number of days to predict into the future"
)

# 3. Model Parameters
st.sidebar.subheader("⚙️ Model Parameters")

epochs = st.sidebar.slider(
    "Training Epochs", 
    min_value=5, 
    max_value=300, 
    value=15,
    help="Number of training iterations"
)

batch_size = st.sidebar.slider(
    "Batch Size", 
    min_value=8, 
    max_value=32, 
    value=16,
    help="Training batch size (larger = faster but more memory)"
)

# Advanced parameters (collapsible)
with st.sidebar.expander("🔧 Advanced Settings"):
    n_lags = st.slider(
        "Autoregressive Lags", 
        min_value=5, 
        max_value=120, 
        value=10,
        help="Number of past observations to use"
    )
    
    # Custom Neural Network Architecture
    st.subheader("🏗️ Custom Architecture")
    
    # Choose between preset and custom
    arch_mode = st.radio(
        "Architecture Mode",
        ["Preset", "Custom"],
        help="Use preset architectures or build your own"
    )
    
    if arch_mode == "Preset":
        ar_layer_size = st.selectbox(
            "Preset Architecture",
            options=["Small [8,4]", "Medium [16,8]", "Large [32,16]"],
            index=1
        )
        
        # Parse architecture selection
        arch_map = {
            "Small [8,4]": [8, 4],
            "Medium [16,8]": [16, 8], 
            "Large [32,16]": [32, 16]
        }
        ar_layers = arch_map[ar_layer_size]
    
    else:  # Custom mode
        num_layers = st.slider(
            "Number of Hidden Layers",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of hidden layers in the neural network"
        )
        
        ar_layers = []
        for i in range(num_layers):
            neurons = st.slider(
                f"Layer {i+1} - Neurons",
                min_value=4,
                max_value=128,
                value=16 if i == 0 else max(4, ar_layers[i-1] // 2),
                step=4,
                help=f"Number of neurons in layer {i+1}"
            )
            ar_layers.append(neurons)
        
        # Display the custom architecture
        st.info(f"**Custom Architecture:** {ar_layers}")

# Training trigger button
train_button = st.sidebar.button(
    "🚀 Download Data & Train Models", 
    type="primary",
    help="Start the training process"
)

# Main content area
if train_button:
    # Data Download Section
    with st.spinner(f"📊 Downloading {ticker} data..."):
        try:
            train_df, test_df, future_index = fetch_stock(
                ticker, train_years, test_months, forecast_days
            )
            
            # Data Summary Display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", len(train_df))
            with col2:
                st.metric("Test Samples", len(test_df))
            with col3:
                st.metric("Total Data Points", len(train_df) + len(test_df))
            
            st.success("✅ Data successfully downloaded and processed")
            
            # Display data range information
            st.info(
                f"**Training Period:** {train_df['Date'].min().strftime('%Y-%m-%d')} "
                f"→ {train_df['Date'].max().strftime('%Y-%m-%d')}\n\n"
                f"**Testing Period:** {test_df['Date'].min().strftime('%Y-%m-%d')} "
                f"→ {test_df['Date'].max().strftime('%Y-%m-%d')}"
            )

            # Regression Mode
            if mode == "Regression":
                st.header("📈 Regression Forecasting Results")
                
                with st.spinner("🧠 Training NeuralProphet model..."):
                    try:
                        # Calculate adaptive n_lags based on data size
                        # adaptive_lags = min(n_lags, max(5, len(train_df) // 10))
                        print(f"Using {n_lags} lags for training")
                        res = train_regression_sota(
                            train_df,
                            test_df,
                            forecast_days,
                            epochs=epochs,
                            n_lags=n_lags,
                            ar_layers=ar_layers,
                            batch_size=batch_size
                        )
                        # Results Display
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "MAE achieved", 
                                f"{res['mae_test']:.2f}",
                                help="Mean Absolute Error on test set"
                            )
                        with col2:
                            if 'n_lags_used' in res:
                                st.metric(
                                    "Lags Used", 
                                    res['n_lags_used'],
                                    help="Actual number of lags used (may be adapted)"
                                )
                        with col3:
                            st.metric(
                                "Direction Accuracy", 
                                f"{res['direction_accuracy']:.1%}",
                                help="Accuracy of predicted price direction on test set"
                            )
                        
                        # Display warnings if present
                        if 'warning' in res:
                            st.warning(f"⚠️ {res['warning']}")
                        
                        # Price Forecast Chart
                        fig = go.Figure()
                        
                        # Historical data
                        historical_df = pd.concat([train_df, test_df])
                        fig.add_trace(go.Scatter(
                            x=historical_df['Date'],
                            y=historical_df['Close'],
                            mode='lines',
                            name='Historical Prices',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Future predictions
                        fig.add_trace(go.Scatter(
                            x=future_index,
                            y=res['forecast'],
                            mode='lines+markers',
                            name=f'{forecast_days}-Day Forecast',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
                        
                        # Chart styling
                        fig.update_layout(
                            title=f"{ticker} Stock Price Forecast",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            hovermode='x unified',
                            template="plotly_white",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast Summary Table
                        forecast_df = pd.DataFrame({
                            'Date': future_index,
                            'Predicted Price': [f"${price:.2f}" for price in res['forecast']],
                            'Day': range(1, forecast_days + 1)
                        })
                        
                        st.subheader("📋 Detailed Forecast")
                        st.dataframe(
                            forecast_df.set_index('Day'), 
                            use_container_width=True
                        )

                        # Trading Agents Backtesting
                        st.subheader("🤖 Algorithmic Trading Agents Performance")
                        st.markdown("*Testing various trading strategies on historical data*")
                        
                        with st.spinner("⚡ Running backtests..."):
                            profits = {}
                            agent_details = {
                                "RSI Mean Reversion": ("Buys oversold, sells overbought", RSIMeanReversion),
                                "MACD Trend Following": ("Follows MACD crossover signals", MACDTrend),
                                "Bollinger Band Breakout": ("Trades channel breakouts", BollingerBreak),
                                "ATR Trailing Stop": ("Uses volatility-based stops", ATRTrailing)
                            }
                            
                            for name, (description, strategy_class) in agent_details.items():
                                try:
                                    profit = run_backtest(
                                        pd.concat([train_df, test_df]), 
                                        strategy_class
                                    )
                                    profits[name] = profit
                                except Exception as e:
                                    st.error(f"Error running {name}: {str(e)}")
                                    profits[name] = 0
                        
                        # Trading Results Visualization
                        if profits:
                            fig2 = go.Figure([go.Bar(
                                x=list(profits.keys()),
                                y=list(profits.values()),
                                marker_color=['green' if p > 0 else 'red' for p in profits.values()],
                                text=[f"${p:,.0f}" for p in profits.values()],
                                textposition='auto'
                            )])
                            
                            fig2.update_layout(
                                title="Trading Agent Performance (Historical Backtest)",
                                xaxis_title="Trading Strategy",
                                yaxis_title="Profit/Loss ($)",
                                template="plotly_white",
                                height=400
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Best performing agent
                            best_agent = max(profits, key=profits.get)
                            best_profit = profits[best_agent]
                            
                            if best_profit > 0:
                                st.success(f"🏆 **Best Performer:** {best_agent} with ${best_profit:,.0f} profit")
                            else:
                                st.warning("⚠️ All agents showed losses on historical data")

                    except Exception as e:
                        st.error(f"❌ Regression training failed: {str(e)}")
                        st.info("💡 Try reducing forecast days or increasing training data")

            # Classification Mode
            else:
                st.header("📊 Classification Results")
                
                with st.spinner("🧠 Training ensemble classifier..."):
                    try:
                        clf_res = train_classification_sota(
                            train_df, 
                            test_df,
                            lstm_epochs=epochs,
                            batch_size=batch_size,
                            forecast_days=forecast_days
                        )
                        
                        # Performance Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Ensemble AUC", 
                                f"{clf_res['auc']:.3f}",
                                help="Area Under ROC Curve (higher is better)"
                            )
                        with col2:
                            accuracy = np.mean(clf_res['pred'] == clf_res['y_true'])
                            st.metric(
                                "Accuracy", 
                                f"{accuracy:.1%}",
                                help="Percentage of correct predictions"
                            )
                        with col3:
                            up_predictions = np.mean(clf_res['pred'])
                            st.metric(
                                "Bullish Signals", 
                                f"{up_predictions:.1%}",
                                help="Percentage of 'up' predictions"
                            )
                        with col4:
                            st.metric(
                                "MAE", 
                                f"{clf_res['mae']:.2f}",
                                help="Mean Absolute Error on test set"
                            )
                        
                        # Prediction Visualization
                        if len(clf_res['pred']) > 0:
                            # Get corresponding test dates
                            pred_dates = test_df['Date'].iloc[-len(clf_res['pred']):]
                            
                            pred_df = pd.DataFrame({
                                'Date': pred_dates,
                                'Prediction': ['📈 UP' if p == 1 else '📉 DOWN' for p in clf_res['pred']],
                                'Confidence': clf_res['proba'],
                                'Actual': ['📈 UP' if p == 1 else '📉 DOWN' for p in clf_res['y_true']]
                            })
                            
                            st.subheader("🎯 Recent Predictions vs Reality")
                            
                            # Show last 10 predictions
                            display_df = pred_df.tail(10).copy()
                            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
                            
                            st.dataframe(
                                display_df.set_index('Date'),
                                use_container_width=True
                            )
                            
                            # Prediction accuracy over time
                            accuracy_series = (clf_res['pred'] == clf_res['y_true']).astype(int)
                            rolling_accuracy = pd.Series(accuracy_series).rolling(10, min_periods=1).mean()
                            
                            fig3 = go.Figure()
                            fig3.add_trace(go.Scatter(
                                x=pred_dates,
                                y=rolling_accuracy,
                                mode='lines',
                                name='10-Day Rolling Accuracy',
                                line=dict(color='green', width=2)
                            ))
                            
                            fig3.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                                           annotation_text="Random Chance (50%)")
                            
                            fig3.update_layout(
                                title="Model Accuracy Over Time",
                                xaxis_title="Date",
                                yaxis_title="Accuracy",
                                yaxis=dict(tickformat=".0%", range=[0, 1]),
                                template="plotly_white",
                                height=400
                            )
                            
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        # Future Trend Graph
                        st.subheader("📈 Future Stock Trend Forecast")
                        st.pyplot(clf_res['fig'])
                        
                        # Forecast Summary Table
                        forecast_df = pd.DataFrame({
                            'Date': clf_res['future_dates'],
                            'Predicted Price': [f"${price:.2f}" for price in clf_res['forecast']],
                            'Day': range(1, forecast_days + 1)
                        })
                        
                        st.subheader("📋 Detailed Forecast")
                        st.dataframe(
                            forecast_df.set_index('Day'), 
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Classification training failed: {str(e)}")
                        st.info("💡 Try reducing sequence length or increasing training data")

        except Exception as e:
            st.error(f"❌ Data download failed: {str(e)}")
            st.info("💡 Please check the ticker symbol and try again")

else:
    # Landing Page Content
    st.markdown("""
    ## 🚀 Welcome to Advanced Stock Forecasting
    
    This platform combines **state-of-the-art machine learning** with **algorithmic trading strategies** 
    to provide comprehensive stock market analysis.
    
    ### 🎯 Features:
    - **🤖 Regression Mode**: Predict exact future stock prices using NeuralProphet
    - **📊 Classification Mode**: Predict market direction (up/down) with ensemble ML
    - **⚡ Real-time Data**: Live data from Yahoo Finance
    - **🔄 Backtesting**: Test trading strategies on historical data
    - **🎛️ Customizable**: Adjustable parameters for different market conditions
    
    ### 📈 Available Models:
    - **NeuralProphet**: Time series forecasting with neural networks
    - **Attention-LSTM**: Deep learning for sequence prediction  
    - **LightGBM**: Gradient boosting with engineered features
    - **Trading Agents**: RSI, MACD, Bollinger Bands, ATR strategies
    
    ### 🎮 Quick Start:
    1. **Select Mode**: Choose Regression or Classification
    2. **Configure Data**: Set ticker, training period, and forecast horizon
    3. **Adjust Parameters**: Fine-tune model settings
    4. **Train & Analyze**: Click the training button to begin
    
    ---
    
    **⚠️ Disclaimer**: This tool is for educational and research purposes only. 
    Do not use for actual trading without proper risk management.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit • Powered by TensorFlow & NeuralProphet"
    "</div>", 
    unsafe_allow_html=True
)