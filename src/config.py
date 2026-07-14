"""
Central configuration for the stock return prediction study.
Edit values here rather than scattering magic numbers through the code.
"""

# ----------------------------------------------------------------------
# Universe of assets
# ----------------------------------------------------------------------
STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'WMT', 'JNJ',
    'BAC', 'NFLX', 'AMD', 'INTC', 'KO', 'PEP', 'DIS', 'IBM', 'ORCL', 'PFE'
]

# Market / conditioning context features (not predicted, used as inputs)
MARKET_TICKERS = {
    'sp500': '^GSPC',
    'vix': '^VIX',
}

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
DATA_PERIOD = "8y"          # pull extra history so walk-forward has room
CACHE_DIR = "data/cache"    # local parquet cache so you don't re-hit yfinance every run

# ----------------------------------------------------------------------
# Walk-forward validation
# ----------------------------------------------------------------------
TRAIN_WINDOW_DAYS = 750     # ~3 trading years per training window
TEST_WINDOW_DAYS = 63       # ~1 trading quarter per test fold
STEP_DAYS = 63              # roll forward by one quarter each fold
MIN_TOTAL_DAYS = TRAIN_WINDOW_DAYS + TEST_WINDOW_DAYS + 100  # skip assets with too little history

# ----------------------------------------------------------------------
# Sequence models
# ----------------------------------------------------------------------
TIME_STEPS = 60             # lookback window for LSTM/GRU/Transformer
LSTM_EPOCHS = 15
GRU_EPOCHS = 15
BATCH_SIZE = 32

# ----------------------------------------------------------------------
# Feature list (must match columns produced in features.py)
# ----------------------------------------------------------------------
FEATURE_COLUMNS = [
    'Log_Return', 'SMA_50', 'EMA_50', 'RSI_14',
    'Upper_Band', 'Middle_Band', 'Lower_Band', 'Volatility', 'Momentum',
    'MACD', 'MACD_Signal', 'Stoch_K', 'Stoch_D', 'ATR_14',
    'OBV', 'Volume_ROC',
    'Mkt_Log_Return', 'VIX_Level', 'VIX_Change'
]

TARGET_COLUMN = 'Target'    # next-day log return

RANDOM_SEED = 42

# ----------------------------------------------------------------------
# Output paths
# ----------------------------------------------------------------------
RESULTS_DIR = "experiments/results"
FIGURES_DIR = "experiments/figures"
