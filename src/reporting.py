"""
QuantX V4.9 — performance summary helpers.
"""
import math

from strategy import MINUTES_PER_DAY


# ---------------------------------
# Performance helpers and reporting
# ---------------------------------
def summarize_performance(series, trades_df):
    total_return = series.iloc[-1] / series.iloc[0] - 1
    days = max(1, (series.index[-1].date() - series.index[0].date()).days)
    annualized = (1 + total_return) ** (252/days) - 1
    rolling_max = series.expanding(min_periods=1).max()
    dd = (series - rolling_max) / rolling_max
    max_dd = dd.min()
    rets = series.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * math.sqrt(252*MINUTES_PER_DAY) if rets.std() > 0 else 0
    return {'total_return': total_return, 'annualized_return': annualized, 'max_drawdown': max_dd, 'sharpe': sharpe}