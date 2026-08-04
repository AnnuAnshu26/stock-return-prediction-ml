"""
Data acquisition with local parquet caching.
Caching matters here: yfinance rate-limits, and you will re-run this
pipeline many times while debugging downstream code. Don't re-download
every time.
"""
import os
import pandas as pd
import yfinance as yf
from . import config


def _cache_path(symbol: str) -> str:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, f"{symbol.replace('^', 'IDX_')}.parquet")


def fetch_ohlcv(symbol: str, period: str = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol, using local cache when available.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    """
    period = period or config.DATA_PERIOD
    path = _cache_path(symbol)

    if os.path.exists(path) and not force_refresh:
        try:
            df = pd.read_parquet(path)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            return df
        except Exception as e:
            # Cache file exists but is corrupted/truncated (e.g. left over
            # from a previously interrupted run). Treat as a cache miss
            # rather than crashing the whole pipeline.
            print(f"[data_pipeline] Cache for {symbol} unreadable ({e}); refetching.")
            try:
                os.remove(path)
            except OSError:
                pass

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        raise ValueError(f"No data retrieved for symbol: {symbol}")

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Normalize the index: strip timezone and drop intraday time component.
    # Different tickers (especially equities vs. index tickers like ^GSPC/
    # ^VIX) can come back with subtly different tz representations or
    # datetime precision from yfinance depending on version/platform. If
    # left tz-aware, joining two separately-fetched DataFrames on their
    # DatetimeIndex can silently produce zero overlap (all-NaN result)
    # instead of an error. Normalizing here guarantees every DataFrame in
    # this pipeline uses the same plain, tz-naive daily index.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()

    df.to_parquet(path)
    return df


def fetch_market_context(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch S&P 500 and VIX series used as cross-asset conditioning features.
    Returns a DataFrame indexed by date with columns: Mkt_Log_Return, VIX_Level, VIX_Change
    """
    sp500 = fetch_ohlcv(config.MARKET_TICKERS['sp500'], force_refresh=force_refresh)
    vix = fetch_ohlcv(config.MARKET_TICKERS['vix'], force_refresh=force_refresh)

    import numpy as np
    mkt = pd.DataFrame(index=sp500.index)
    mkt['Mkt_Log_Return'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
    mkt['VIX_Level'] = vix['Close']
    mkt['VIX_Change'] = vix['Close'].pct_change()

    return mkt.dropna()


def fetch_all(stocks=None, force_refresh: bool = False) -> dict:
    """
    Fetch OHLCV for every stock in the universe plus market context.
    Returns dict: {symbol: DataFrame}
    """
    stocks = stocks or config.STOCKS
    data = {}
    for symbol in stocks:
        try:
            data[symbol] = fetch_ohlcv(symbol, force_refresh=force_refresh)
        except Exception as e:
            print(f"[data_pipeline] Failed to fetch {symbol}: {e}")
    return data