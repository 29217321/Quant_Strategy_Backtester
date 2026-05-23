"""
QuantX V4.9 — strategy parameters, data loaders, and indicator helpers.
"""
import os
import glob
from datetime import time

import numpy as np
import pandas as pd

try:
    import polars as pl  # optional, only needed for .parquet inputs
except Exception:
    pl = None

# ------------------------------------------------------------------
# Run configuration (override via env vars or by editing here)
# ------------------------------------------------------------------
RUN_MODE  = "RECENT"   # 写死,不再读环境变量。可选: JAN / FEB / JAN_FEB / JAN_JUN / JUL_SEP / OCT_DEC / FULL / RECENT
DATA_ROOT = os.environ.get("QUANTX_DATA_ROOT", "./data")

# Dow 30 + a few large caps (matches the README universe size of 32).
# ALL_TICKERS = [
#     "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "JPM",
#     "V", "UNH", "HD", "PG", "MA", "DIS", "BAC", "XOM",
#     "KO", "PFE", "CSCO", "WMT", "INTC", "VZ", "CVX", "MRK",
#     "MCD", "NKE", "CRM", "BA", "IBM", "GS", "MMM", "CAT",
# ]
ALL_TICKERS = ["NVDA"]

# ---------------
# Strategy params
# ---------------
DAILY_TREND_WINDOW    = 5
INTRADAY_LOOKBACK     = 15
ATR_WINDOW            = 14

Z_THRESHOLD           = 0.65
CONFIRM_BARS          = 0
VOLUME_MIN_FACTOR     = 0.35

RISK_PER_TRADE        = 0.035
MAX_POSITION_FRACTION = 0.05
MAX_GROSS_EXPOSURE    = 2.2
MAX_OPEN_POSITIONS_TOTAL = 14

STOP_LOSS_PCT         = 0.022
TAKE_PROFIT_PCT       = 0.10

# ---------------------------------------------------------------
# Longbridge US-stock fee schedule (USD) —以面额的方式成交,不再使用简单百分比
#   平台费 : 0.005 / 股, 单笔最低 1 USD                     (买+卖)
#   交易费 : 0.003 / 股                                       (买+卖)
#   SEC 费: 0.0000229 * 交易金额, 最低 0.01 USD              (仅卖)
#   TAF    : 0.00013 / 股, 最低 0.01, 最高 6.49 USD            (仅卖)
# ---------------------------------------------------------------
LB_PLATFORM_FEE_PER_SHARE    = 0.005
LB_PLATFORM_FEE_MIN          = 1.00
LB_TRANSACTION_FEE_PER_SHARE = 0.003
LB_SEC_FEE_RATE              = 0.0000229   # sell only
LB_SEC_FEE_MIN               = 0.01
LB_TAF_PER_SHARE             = 0.00013     # sell only
LB_TAF_MIN                   = 0.01
LB_TAF_MAX                   = 6.49

# 滑点仍然以百分比模拟(与佣金独立)
SLIPPAGE_PCT          = 0.0005

# 保留旧变量名,防止别处引用;新逻辑不再使用该值
TRANSACTION_COST_PCT  = 0.0

MINUTES_PER_DAY       = 390
SKIP_FIRST_MINUTES    = 3
SKIP_LAST_MINUTES     = 5

INITIAL_CAPITAL       = 1_000_000.0


def compute_commission(shares, price, side):
    """计算单笔成交的长桥美股佣金(USD)。side='buy'|'sell'。返回总额>=0。"""
    n = abs(int(shares))
    if n <= 0:
        return 0.0
    notional = n * float(price)
    platform    = max(LB_PLATFORM_FEE_PER_SHARE * n, LB_PLATFORM_FEE_MIN)
    transaction = LB_TRANSACTION_FEE_PER_SHARE * n
    fee = platform + transaction
    if side == 'sell':
        sec = max(LB_SEC_FEE_RATE * notional, LB_SEC_FEE_MIN)
        taf = min(max(LB_TAF_PER_SHARE * n, LB_TAF_MIN), LB_TAF_MAX)
        fee += sec + taf
    return float(fee)

ROLL_STD_FLOOR = 1e-4
VOL_FLOOR = 1e-4
ATR_FLOOR = 1e-4

# Output paths & behaviour
OUT_DIR = "./quantx_reports"
os.makedirs(OUT_DIR, exist_ok=True)
GENERATE_PDF = True
MAX_TICKER_CHARTS = 32            # max per-ticker images
MAX_TRADES_PER_TICKER_ZOOM = 0    # 0 = disabled (avoid 322 extra charts)

# ------------------------
# RUN_MODE -> dates
# ------------------------
def _detect_data_date_range(ticker):
    """扫描 DATA_ROOT/<ticker>/ 下所有 *_YYYYMMDD.{parquet,parq,csv,pkl} 文件,
    返回 (start_date_str, end_date_str), 格式 'YYYY-MM-DD'。

    若目录不存在或没有可识别的文件名, 返回 (None, None)。
    """
    import re
    folder = os.path.join(DATA_ROOT, ticker)
    if not os.path.isdir(folder):
        return None, None
    pat = re.compile(r"(\d{8})")  # 任意位置出现的 8 位数字 = YYYYMMDD
    dates = set()
    for fn in os.listdir(folder):
        if not fn.lower().endswith((".parquet", ".parq", ".csv", ".pkl")):
            continue
        m = pat.search(fn)
        if not m:
            continue
        try:
            d = pd.to_datetime(m.group(1), format="%Y%m%d")
            dates.add(d)
        except Exception:
            continue
    if not dates:
        return None, None
    return min(dates).strftime("%Y-%m-%d"), max(dates).strftime("%Y-%m-%d")


if RUN_MODE == "JAN":
    START_DATE, END_DATE = "2024-01-02", "2024-01-31"
elif RUN_MODE == "FEB":
    START_DATE, END_DATE = "2024-02-01", "2024-02-29"
elif RUN_MODE == "JAN_FEB":
    START_DATE, END_DATE = "2024-01-02", "2024-02-29"
elif RUN_MODE == "JAN_JUN":
    START_DATE, END_DATE = "2024-01-02", "2024-06-28"
elif RUN_MODE == "JUL_SEP":
    START_DATE, END_DATE = "2024-07-01", "2024-09-30"
elif RUN_MODE == "OCT_DEC":
    START_DATE, END_DATE = "2024-10-01", "2024-12-31"
elif RUN_MODE == "RECENT":
    # 自动按 data/<ticker>/ 下已下载的文件名 (含 YYYYMMDD) 推导日期窗口。
    # 假设 ALL_TICKERS 只填一个代码 (多只时取第一只作为基准)。
    _probe_ticker = ALL_TICKERS[0] if ALL_TICKERS else None
    _auto_start, _auto_end = (None, None)
    if _probe_ticker:
        _auto_start, _auto_end = _detect_data_date_range(_probe_ticker)
    if _auto_start and _auto_end:
        START_DATE, END_DATE = _auto_start, _auto_end
        print(f"[RECENT] auto-detected date range from data/{_probe_ticker}/: "
              f"{START_DATE} → {END_DATE}")
    else:
        # 兜底:目录为空或不存在, 用最近一年的硬编码值
        START_DATE, END_DATE = "2025-05-23", "2026-05-22"
        print(f"[RECENT] data/{_probe_ticker}/ 找不到带 YYYYMMDD 的数据文件, "
              f"fallback 到 {START_DATE} → {END_DATE}")
else:
    START_DATE, END_DATE = "2024-01-02", "2024-12-31"

TICKERS = ALL_TICKERS

print(f"QuantX FINAL Backtest | RUN_MODE={RUN_MODE}")
print(f"Running {len(TICKERS)} tickers: {TICKERS}")
print(f"Date range: {START_DATE} → {END_DATE}\n")

# ------------------------
# IO helper (robust & cached)
# ------------------------
_day_cache = {}
def _find_file_for_day(ticker, date_str):
    """
    Look inside DATA_ROOT/ticker for any file containing date_str in its filename and return full path.
    Accepts parquet, csv, pkl. Returns None if nothing found.
    """
    folder = os.path.join(DATA_ROOT, ticker)
    if not os.path.isdir(folder):
        return None
    # search for common extensions (csv 优先, 方便人工核对; 找不到再回退到 parquet)
    for ext in ("*.csv", "*.parquet", "*.parq", "*.pkl"):
        for fn in glob.glob(os.path.join(folder, f"*{date_str}*{ext.replace('*','')}")):
            return fn
    # fallback: any file with date_str substring
    for fn in os.listdir(folder):
        if date_str in fn:
            return os.path.join(folder, fn)
    return None

def load_minute_parquet_for_day(ticker, date_str):
    """
    Return pandas DataFrame with columns: timestamp, open, high, low, close, volume, ms_of_day
    or None if not available.
    """
    key = (ticker, date_str)
    if key in _day_cache:
        return _day_cache[key]

    candidate = _find_file_for_day(ticker, date_str)
    if candidate is None:
        _day_cache[key] = None
        return None
    try:
        if candidate.endswith((".parquet",".parq")):
            df_pl = pl.read_parquet(candidate)
            df = df_pl.to_pandas()
        elif candidate.endswith(".csv"):
            df = pd.read_csv(candidate)
        elif candidate.endswith(".pkl"):
            df = pd.read_pickle(candidate)
        else:
            # try read parquet/csv heuristics
            try:
                df_pl = pl.read_parquet(candidate)
                df = df_pl.to_pandas()
            except Exception:
                df = pd.read_csv(candidate)
    except Exception:
        _day_cache[key] = None
        return None

    # normalize columns and construct timestamp (support a couple of naming conventions)
    if 'date' not in df.columns or 'ms_of_day' not in df.columns:
        # try to infer
        if 'timestamp' in df.columns:
            # assume timestamp is epoch ms or ISO string
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception:
                pass
        # if we don't have required columns, give up
    # If 'date' exists and 'ms_of_day' exists -> create timestamp
    if 'date' in df.columns and 'ms_of_day' in df.columns:
        df['date_dt'] = pd.to_datetime(df['date'].astype(str), format="%Y%m%d", errors='coerce')
        df['timestamp'] = df['date_dt'] + pd.to_timedelta(df['ms_of_day'], unit='ms')
    # else try parse 'timestamp' column (already)
    if 'timestamp' not in df.columns:
        _day_cache[key] = None
        return None

    # restrict to market hours if we can
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.loc[
            (df['timestamp'].dt.time >= time(9,30)) &
            (df['timestamp'].dt.time <= time(16,0))
        ]
    except Exception:
        pass

    # standardize numeric columns existence
    for c in ['open','high','low','close','volume','ms_of_day']:
        if c not in df.columns:
            df[c] = np.nan

    df = df.sort_values('timestamp').reset_index(drop=True)
    if df.empty:
        _day_cache[key] = None
        return None
    _day_cache[key] = df[['timestamp','open','high','low','close','volume','ms_of_day']].copy()
    return _day_cache[key]


# ------------------------------------------------------------------
# Indicator helpers (restored from the original notebook)
# ------------------------------------------------------------------
def compute_daily_trend(ticker, dates):
    """Return (daily_close_series, daily_trend_series) indexed by ``dates``.

    The trend is a rolling SMA of the daily close (window=DAILY_TREND_WINDOW).
    Daily close is derived from the last minute bar of each day's minute file.
    Days with no data simply produce NaN — the backtest gracefully skips them.
    """
    closes = {}
    for d in dates:
        date_str = pd.Timestamp(d).strftime("%Y%m%d")
        df = load_minute_parquet_for_day(ticker, date_str)
        if df is None or df.empty:
            continue
        last_close = df['close'].dropna()
        if last_close.empty:
            continue
        closes[pd.Timestamp(d)] = float(last_close.iloc[-1])

    s = pd.Series(closes, dtype=float).reindex(pd.DatetimeIndex(dates)).sort_index()
    trend = s.rolling(window=DAILY_TREND_WINDOW, min_periods=1).mean()
    return s, trend


def compute_intraday_indicators(df):
    """Add the columns the backtest needs: z, vol15, volatility, atr.

    Returns ``None`` when the input is None/empty so the caller can short-circuit.
    """
    if df is None or df.empty:
        return None

    df = df.copy()
    close = df['close'].astype(float)

    # Rolling mean / std of close, then standardized z-score
    roll_mean = close.rolling(INTRADAY_LOOKBACK, min_periods=2).mean()
    roll_std  = close.rolling(INTRADAY_LOOKBACK, min_periods=2).std().clip(lower=ROLL_STD_FLOOR)
    df['z'] = (close - roll_mean) / roll_std

    # Rolling 15-bar volume sum, used by the volume-confirmation filter
    df['vol15'] = df['volume'].rolling(INTRADAY_LOOKBACK, min_periods=1).sum()

    # Realised return volatility (rolling std of pct change)
    rets = close.pct_change()
    df['volatility'] = rets.rolling(INTRADAY_LOOKBACK, min_periods=2).std().fillna(VOL_FLOOR)

    # True Range / ATR (Wilder's smoothing approximated by simple MA)
    high = df['high'].astype(float)
    low  = df['low'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_WINDOW, min_periods=1).mean().fillna(ATR_FLOOR)

    return df