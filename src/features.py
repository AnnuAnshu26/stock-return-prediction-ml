"""
Feature engineering.

Design notes for the paper's Methodology section:
- Target is next-day LOG RETURN (stationary), not raw price — regressing on
  raw price is a classic mistake that inflates R^2 artificially via trend.
- All indicators are computed causally (rolling/shift), so no future
  information leaks into a given row's features.
- Market context (S&P 500 return, VIX level/change) is merged in to give
  the model conditioning information beyond the single asset's own history.
"""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _stochastic(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d = k.rolling(d_window).mean()
    return k, d


def _atr(high, low, close, window=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def engineer_features(df: pd.DataFrame, market_context: pd.DataFrame = None) -> pd.DataFrame:
    """
    Takes raw OHLCV DataFrame, returns feature-engineered DataFrame with
    a stationary target column ready for modeling.
    """
    df = df.copy()

    # --- Target: next-day log return (stationary) ---
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Target'] = df['Log_Return'].shift(-1)

    # --- Trend ---
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # --- Momentum ---
    df['RSI_14'] = _rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'] = _macd(df['Close'])
    df['Stoch_K'], df['Stoch_D'] = _stochastic(df['High'], df['Low'], df['Close'])
    df['Momentum'] = df['Close'] - df['Close'].shift(4)

    # --- Volatility ---
    middle = df['Close'].rolling(20).mean()
    rolling_std = df['Close'].rolling(20).std()
    df['Middle_Band'] = middle
    df['Upper_Band'] = middle + 2 * rolling_std
    df['Lower_Band'] = middle - 2 * rolling_std
    df['Volatility'] = df['Close'].rolling(21).std()
    df['ATR_14'] = _atr(df['High'], df['Low'], df['Close'], 14)

    # --- Volume ---
    df['OBV'] = _obv(df['Close'], df['Volume'])
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5)

    # --- Market context merge (left join on date index) ---
    if market_context is not None:
        df = df.join(market_context, how='left')
        df[['Mkt_Log_Return', 'VIX_Level', 'VIX_Change']] = (
            df[['Mkt_Log_Return', 'VIX_Level', 'VIX_Change']].ffill()
        )
    else:
        df['Mkt_Log_Return'] = 0.0
        df['VIX_Level'] = 0.0
        df['VIX_Change'] = 0.0

    df.dropna(inplace=True)
    return df
