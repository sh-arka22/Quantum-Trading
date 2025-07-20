import yfinance as yf
import pandas as pd
import os
import warnings
from datetime import datetime, timedelta

# Suppress yfinance warnings
warnings.filterwarnings("ignore", message=".*YF.download.*")

def fetch_stock(ticker: str,
                train_years: int = 2,
                test_months: int = 1,
                forecast_days: int = 60,
                data_dir: str = "data"):
    """
    Returns 3 DataFrames: train_df, test_df, future_index
    
    First checks if ticker data exists locally, if not downloads it.
    If local data exists, uses the entire dataset for train/test split.
    """
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define file path for the ticker
    file_path = os.path.join(data_dir, f"{ticker.upper()}.csv")
    
    # Check if file exists locally
    if os.path.exists(file_path):
        print(f"📁 Loading {ticker} data from local file: {file_path}")
        try:
            full = pd.read_csv(file_path)
            # Ensure Date column is datetime
            full['Date'] = pd.to_datetime(full['Date'])
            full = full.sort_values('Date').reset_index(drop=True)
            
        except Exception as e:
            print(f"❌ Error reading local file: {e}")
            print("🌐 Downloading fresh data...")
            full = download_and_save_data(ticker, file_path, train_years)
    else:
        print(f"🌐 {ticker} not found locally. Downloading...")
        full = download_and_save_data(ticker, file_path, train_years)
    
    # Validate required columns
    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in expected_cols if col not in full.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    
    # Use entire dataset for train/test split
    total_rows = len(full)
    
    if total_rows < forecast_days + test_months * 21 + 50:  # Minimum data check
        raise ValueError(f"Insufficient data: {total_rows} rows. Need at least {forecast_days + test_months * 21 + 50}")
    
    # Calculate split points based on entire dataset
    test_rows = test_months * 21  # Approximate trading days per month
    train_rows = total_rows - test_rows - forecast_days
    
    # Ensure we have enough training data
    if train_rows < 100:  # Minimum training data
        train_rows = max(100, total_rows - test_rows)
        test_rows = total_rows - train_rows
    
    # Split the data
    train_df = full.iloc[:train_rows].copy()
    test_df = full.iloc[train_rows:train_rows + test_rows].copy()
    
    # Create future index for forecasting
    last_date = full['Date'].iloc[-1]
    future_index = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days, 
        freq='B'  # Business days only
    )
    
    print(f"📊 Data split: {len(train_df)} training, {len(test_df)} testing, {forecast_days} forecast days")
    print(f"📅 Training: {train_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {train_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"📅 Testing: {test_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {test_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    
    return train_df, test_df, future_index


def download_and_save_data(ticker: str, file_path: str, train_years: int = 5):
    """
    Downloads stock data and saves it locally
    """
    try:
        # Download more data for better training (5+ years)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=train_years * 365)
        
        print(f"📈 Downloading {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        full = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # Explicitly set to avoid warnings
        )
        
        if full.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Fix MultiIndex columns if present
        if isinstance(full.columns, pd.MultiIndex):
            full.columns = full.columns.get_level_values(0)
        
        # Reset index to make Date a column
        full.reset_index(inplace=True)
        
        # Ensure proper column names
        if 'Adj Close' in full.columns and 'Close' not in full.columns:
            full['Close'] = full['Adj Close']
        
        # Save to local file
        full.to_csv(file_path, index=False)
        print(f"💾 Saved {len(full)} rows to {file_path}")
        
        return full
        
    except Exception as e:
        raise ValueError(f"Failed to download {ticker}: {str(e)}")


def update_local_data(ticker: str, data_dir: str = "data"):
    """
    Force update local data by downloading fresh data
    """
    file_path = os.path.join(data_dir, f"{ticker.upper()}.csv")
    
    if os.path.exists(file_path):
        # Check last date in file
        try:
            existing_data = pd.read_csv(file_path)
            existing_data['Date'] = pd.to_datetime(existing_data['Date'])
            last_date = existing_data['Date'].max()
            days_old = (datetime.now() - last_date).days
            
            if days_old > 1:  # Update if data is more than 1 day old
                print(f"🔄 Data is {days_old} days old. Updating...")
                download_and_save_data(ticker, file_path)
            else:
                print(f"✅ Data is current (last update: {last_date.strftime('%Y-%m-%d')})")
                
        except Exception as e:
            print(f"⚠️ Error checking data age: {e}. Re-downloading...")
            download_and_save_data(ticker, file_path)
    else:
        print(f"📁 No local data found for {ticker}. Downloading...")
        download_and_save_data(ticker, file_path)


def list_available_tickers(data_dir: str = "data"):
    """
    List all locally available ticker files
    """
    if not os.path.exists(data_dir):
        return []
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    tickers = [f.replace('.csv', '') for f in csv_files]
    
    return sorted(tickers)
