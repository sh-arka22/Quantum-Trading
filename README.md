# 📈 Advanced Stock Forecasting & Trading Platform

A comprehensive machine learning platform for stock price prediction and algorithmic trading, combining state-of-the-art neural networks with traditional technical analysis strategies.

## 🚀 Overview

This platform integrates multiple advanced machine learning models to provide accurate stock price forecasting and automated trading strategy backtesting. It supports both regression (exact price prediction) and classification (direction prediction) modes, with real-time data integration and interactive visualization.

## 🏗️ System Architecture

### Core Components

```
├── Data Layer (yfinance_utils.py)
│   ├── Real-time data fetching via Yahoo Finance
│   ├── Local data caching and management
│   └── Data preprocessing and validation
│
├── Model Layer
│   ├── Regression Models (regression_sota.py)
│   │   └── NeuralProphet (Time Series Forecasting)
│   ├── Classification Models (classification_sota.py)
│   │   ├── Attention-LSTM (Deep Learning)
│   │   └── LightGBM (Gradient Boosting)
│   ├── Custom Neural Networks (rnn_model.py)
│   │   └── Custom Attention Mechanism
│   └── Trading Agents (agents_sota.py)
│       ├── RSI Mean Reversion
│       ├── MACD Trend Following
│       ├── Bollinger Band Breakout
│       └── ATR Trailing Stop
│
└── Application Layer (app.py)
    ├── Streamlit Web Interface
    ├── GPU Configuration & Management
    ├── Interactive Visualizations
    └── Real-time Model Training & Prediction
```

## 🤖 Machine Learning Models

### 1. NeuralProphet (Regression Mode)

**Architecture**: Advanced time series forecasting combining traditional statistical methods with neural networks.

**Key Features**:
- **Autoregressive Components**: Uses configurable lagged values (n_lags) to capture temporal dependencies
- **Seasonality Modeling**: Automatic detection and modeling of yearly, weekly, and daily patterns
- **Neural Network Layers**: Customizable fully connected layers for non-linear pattern learning
- **Trend Analysis**: Automatic trend detection and extrapolation
- **Normalization**: Soft normalization for stable training

**Configuration**:
- Configurable AR layers: [8,4], [16,8], [32,16] or custom
- Adjustable lag periods: 5-120 historical observations
- Batch size optimization for GPU acceleration
- Automatic missing value imputation

**Prediction Process**:
1. Data preprocessing with deduplication and sorting
2. Feature engineering with lag creation
3. Seasonality decomposition
4. Neural network training with backpropagation
5. Future value prediction with trend extrapolation

### 2. Attention-LSTM (Classification Mode)

**Architecture**: Deep recurrent neural network with attention mechanism for sequence-to-sequence learning.

**Technical Implementation**:
```python
Model Architecture:
Input Shape: (sequence_length, 4)  # [Open, High, Low, Close]
├── LSTM Layer 1: 50 units, return_sequences=True
├── Dropout: 0.2
├── LSTM Layer 2: 50 units, return_sequences=True  
├── Custom Attention Layer: Weighted feature aggregation
├── LSTM Layer 3: 50 units, final hidden state
└── Dense Output: 1 unit (regression) or 2 units (classification)
```

**Attention Mechanism**:
- **Weight Calculation**: `e = tanh(X · W)` where W is learned weight matrix
- **Attention Scores**: `α = softmax(e)` for normalized importance weights
- **Weighted Output**: `output = Σ(αᵢ · xᵢ)` for context-aware representation

**Training Process**:
1. Sequence preparation with sliding windows (30-day sequences)
2. Feature normalization and scaling
3. Attention weight learning during backpropagation
4. Dropout regularization to prevent overfitting
5. Adam optimizer with MSE/categorical crossentropy loss

### 3. LightGBM Ensemble

**Architecture**: Gradient boosting decision tree with engineered technical indicators.

**Feature Engineering**:
```python
Technical Indicators:
├── Simple Moving Averages: 5, 10, 20-day periods
├── RSI Indicators: 5, 10, 20-day periods  
├── Volume Analysis: Moving averages and volatility
├── Price Momentum: Rate of change calculations
└── Statistical Features: Rolling std, min, max
```

**Model Configuration**:
- **Estimators**: 500 trees for robust learning
- **Learning Rate**: 0.02 for stable convergence
- **Max Depth**: Unlimited (-1) for complex pattern capture
- **Regularization**: Built-in L1/L2 regularization

### 4. Ensemble Integration

**Methodology**: Simple averaging of LSTM and LightGBM predictions for improved robustness.

```python
ensemble_prediction = 0.5 × (lstm_pred + lgb_pred)
```

**Benefits**:
- Reduced overfitting through model diversity
- Improved generalization across market conditions
- Balanced technical and sequence-based analysis

## 📊 Trading Strategies (agents_sota.py)

### 1. RSI Mean Reversion Strategy

**Logic**: Exploits overbought/oversold conditions using Relative Strength Index.

**Parameters**:
- RSI Period: 14 days (default)
- Oversold Threshold: 30
- Overbought Threshold: 70

**Execution**:
```python
if RSI < 30: BUY (oversold condition)
if RSI > 70: SELL (overbought condition)  
if position_open and RSI crosses 50: CLOSE
```

### 2. MACD Trend Following

**Logic**: Follows momentum using Moving Average Convergence Divergence signals.

**Components**:
- Fast EMA: 12 periods
- Slow EMA: 26 periods  
- Signal Line: 9-period EMA of MACD

**Execution**:
```python
if MACD > Signal: BUY (bullish crossover)
if MACD < Signal: SELL (bearish crossover)
if MACD × Signal < 0: CLOSE (momentum reversal)
```

### 3. Bollinger Band Breakout

**Logic**: Trades volatility breakouts using statistical bands.

**Parameters**:
- Lookback Period: 20 days
- Standard Deviations: 2.0

**Execution**:
```python
if Price > Upper_Band: BUY (upward breakout)
if Price < Lower_Band: SELL (downward breakout)
if abs(Price - Middle_Band) < 0.01: CLOSE
```

### 4. ATR Trailing Stop

**Logic**: Volatility-adjusted position management using Average True Range.

**Configuration**:
- ATR Period: 14 days
- Multiplier: 3.0

**Execution**:
```python
trailing_stop = Price - (ATR × multiplier)
if Price < trailing_stop: CLOSE position
```

## 📈 Stock Prediction Process

### Data Pipeline

1. **Data Acquisition**:
   - Real-time fetching via yfinance API
   - Local caching for improved performance
   - Automatic data validation and cleaning
   - Business day alignment for trading calendars

2. **Preprocessing**:
   - Missing value interpolation
   - Outlier detection and handling
   - Feature scaling and normalization
   - Sequence generation for temporal models

3. **Feature Engineering**:
   - Technical indicator calculations
   - Lag feature creation
   - Seasonality decomposition
   - Volatility metrics

### Model Training Pipeline

```python
Training Workflow:
1. Data splitting: Train/Validation/Test
2. GPU configuration and memory optimization
3. Hyperparameter validation
4. Model initialization with custom architecture
5. Training with early stopping and regularization
6. Performance evaluation and metrics calculation
7. Model ensembling and final prediction generation
```

### Prediction Generation

**Regression Mode**:
1. Historical pattern analysis with NeuralProphet
2. Seasonality extraction and future projection
3. Trend continuation with neural network adjustment
4. Confidence interval calculation
5. Multi-day forecast generation

**Classification Mode**:
1. Binary direction prediction (Up/Down)
2. Ensemble probability calculation
3. Threshold-based decision making
4. Direction accuracy assessment
5. Future trend visualization

## 🎯 Key Features

### Real-time Capabilities
- **Live Data Integration**: Automatic fetching of latest market data
- **GPU Acceleration**: CUDA support for faster model training
- **Streaming Predictions**: Real-time forecast updates
- **Interactive Visualization**: Dynamic charts and performance metrics

### Advanced Analytics
- **Multi-timeframe Analysis**: Support for various prediction horizons
- **Risk Assessment**: Volatility-based confidence intervals
- **Performance Metrics**: MAE, Direction Accuracy, AUC scores
- **Backtesting Engine**: Historical strategy performance evaluation

### User Experience
- **Intuitive Interface**: Streamlit-based web application
- **Customizable Parameters**: Adjustable model hyperparameters
- **Visual Analytics**: Interactive plotly charts and graphs
- **Export Capabilities**: Downloadable predictions and reports

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for acceleration)
- Minimum 8GB RAM recommended

### Installation

```bash
# Clone the repository
git clone <rhttps://github.com/sh-arka22/Quantum-Trading.gitl>
cd stock-forecasting-platform

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Dependencies

```
streamlit          # Web application framework
pandas            # Data manipulation
numpy             # Numerical computing
scikit-learn      # Machine learning utilities
tensorflow>=2.4   # Deep learning framework
plotly            # Interactive visualizations
yfinance          # Stock data fetching
backtrader        # Trading strategy backtesting
ta                # Technical analysis indicators
xgboost           # Gradient boosting
lightgbm          # LightGBM implementation
neuralprophet     # Time series forecasting
```

## 📱 Usage Guide

### Quick Start

1. **Launch Application**: Run `streamlit run app.py`
2. **Configure Parameters**:
   - Select prediction mode (Regression/Classification)
   - Enter stock ticker symbol
   - Adjust training period and forecast horizon
   - Customize model architecture
3. **Train Models**: Click "Download Data & Train Models"
4. **Analyze Results**: Review predictions, metrics, and visualizations
5. **Evaluate Strategies**: Examine trading agent performance

### Advanced Configuration

**Model Customization**:
- Adjust neural network layers and neurons
- Modify lag periods for temporal dependencies
- Configure ensemble weights
- Set training epochs and batch sizes

**Data Management**:
- Local data caching for improved performance
- Automatic data updates and validation
- Support for multiple time periods
- Business day alignment

## 📊 Performance Metrics

### Regression Metrics
- **Mean Absolute Error (MAE)**: Average prediction error
- **Direction Accuracy**: Percentage of correct trend predictions
- **R-squared**: Variance explanation coefficient

### Classification Metrics
- **AUC Score**: Area under ROC curve for binary classification
- **Accuracy**: Overall prediction correctness
- **Precision/Recall**: Class-specific performance measures

### Trading Metrics
- **Total Return**: Absolute profit/loss from strategies
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## ⚠️ Important Disclaimers

1. **Educational Purpose**: This platform is designed for research and educational use only
2. **Market Risk**: Stock trading involves substantial risk of loss
3. **No Guarantees**: Past performance does not guarantee future results
4. **Professional Advice**: Consult with financial professionals before making investment decisions

## 🔮 Future Enhancements

- Real-time streaming data integration
- Multi-asset portfolio optimization
- Advanced risk management systems
- Cloud deployment and scaling
- Mobile application development
- Alternative data source integration

## 📚 Technical References

- **NeuralProphet**: Facebook's neural network time series forecasting
- **Attention Mechanism**: Bahdanau et al. attention for sequence modeling
- **LightGBM**: Microsoft's gradient boosting framework
- **Technical Analysis**: Traditional trading indicators and strategies

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests for improvements, bug fixes, or new features.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for the quantitative finance community**