"""
QuantX V4.9 — sequential backtest engine.
"""
import math
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd

from strategy import (
    INITIAL_CAPITAL,
    SKIP_FIRST_MINUTES, SKIP_LAST_MINUTES,
    VOLUME_MIN_FACTOR, Z_THRESHOLD, CONFIRM_BARS,
    MAX_GROSS_EXPOSURE, MAX_POSITION_FRACTION,
    RISK_PER_TRADE, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    SLIPPAGE_PCT,
    VOL_FLOOR, ATR_FLOOR,
    compute_daily_trend, compute_intraday_indicators,
    load_minute_parquet_for_day,
    compute_commission,
)


# ------------------------
# Backtest core (sequential)
# ------------------------
def run_backtest(tickers, start_date, end_date, stop_on_negative_cash=False):
    dates = pd.bdate_range(start=start_date, end=end_date)
    daily_closes, daily_trends = {}, {}
    for t in tickers:
        s, trend = compute_daily_trend(t, dates)
        daily_closes[t], daily_trends[t] = s, trend

    cash = float(INITIAL_CAPITAL)
    positions = {t: {'shares':0, 'entry_price':0.0, 'entry_time':None, 'z':np.nan} for t in tickers}
    portfolio_history = [(pd.to_datetime(start_date), cash)]
    trades = []
    stats = defaultdict(int)

    for current_date in dates:
        date_str = current_date.strftime("%Y%m%d")
        print(f"Processing {date_str}")
        try:
            intraday = {t: compute_intraday_indicators(load_minute_parquet_for_day(t, date_str)) for t in tickers}
            if all(v is None for v in intraday.values()):
                portfolio_history.append((pd.to_datetime(current_date) + pd.Timedelta(hours=16), cash))
                continue

            entered_today = {t: False for t in tickers}
            max_len = max((len(df) for df in intraday.values() if df is not None), default=0)
            start_i = SKIP_FIRST_MINUTES
            end_i = max_len - SKIP_LAST_MINUTES - 1
            if end_i <= start_i:
                portfolio_history.append((pd.to_datetime(current_date) + pd.Timedelta(hours=16), cash))
                continue

            for i in range(start_i, end_i):
                # mark-to-market price mapping
                current_prices = {}
                for t, df in intraday.items():
                    if df is None or i+1 >= len(df): continue
                    p = df['open'].iloc[i+1]
                    if np.isfinite(p) and p > 0: current_prices[t] = p

                total_value = cash + sum(pos['shares'] * current_prices.get(sym, pos['entry_price'])
                                         for sym,pos in positions.items() if pos['shares'] != 0)
                gross_exposure = sum(abs(pos['shares']) * current_prices.get(sym, pos['entry_price'])
                                     for sym,pos in positions.items() if pos['shares'] != 0)

                open_positions_count = sum(1 for p in positions.values() if p['shares'] != 0)
                for t, df in intraday.items():
                    if df is None or i+1 >= len(df):
                        stats['missing_data'] += 1
                        continue

                    exec_price = df['open'].iloc[i+1]
                    if not (np.isfinite(exec_price) and exec_price > 0):
                        stats['price_fail'] += 1
                        continue

                    pos = positions[t]

                    # EXIT first (intraday)
                    if pos['shares'] != 0:
                        pnl_pct = (exec_price - pos['entry_price']) / (pos['entry_price'] if pos['entry_price']!=0 else 1e-8)
                        if (pnl_pct <= -STOP_LOSS_PCT) or (pnl_pct >= TAKE_PROFIT_PCT):
                            gross = pos['shares'] * exec_price * (1.0 - SLIPPAGE_PCT)
                            proceeds = gross - compute_commission(pos['shares'], exec_price, 'sell')
                            cash += proceeds
                            trades.append({
                                'symbol': t,
                                'entry_price': pos['entry_price'],
                                'exit_price': exec_price,
                                'entry_time': pos.get('entry_time'),
                                'exit_time': df['timestamp'].iloc[i+1],
                                'shares': pos['shares'],
                                'pnl': (exec_price - pos['entry_price']) * pos['shares'],
                                'pnl_pct': pnl_pct,
                                'z_score': pos.get('z', np.nan)
                            })
                            positions[t] = {'shares':0,'entry_price':0.0,'entry_time':None,'z':np.nan}
                            stats['intraday_exits'] += 1
                            continue

                    # ENTRY (only if no open position and not entered today)
                    if pos['shares'] == 0 and not entered_today.get(t, False):
                        z = df['z'].iloc[i] if 'z' in df.columns else np.nan
                        vol15 = df['vol15'].iloc[i] if 'vol15' in df.columns else 0
                        median_vol = df['volume'].median() if len(df)>0 else 0
                        vol_ok = (median_vol > 0) and (vol15 >= median_vol * VOLUME_MIN_FACTOR)

                        today_close = daily_closes[t].loc[current_date] if current_date in daily_closes[t].index else np.nan
                        trend_val = daily_trends[t].loc[current_date] if current_date in daily_trends[t].index else np.nan
                        trend_bull = pd.notna(today_close) and pd.notna(trend_val) and (today_close > trend_val)

                        confirm_pass = True
                        if CONFIRM_BARS > 0:
                            confirm_pass = True
                            for j in range(CONFIRM_BARS):
                                idx = i - j
                                if idx < 0 or idx >= len(df) or not np.isfinite(df['z'].iloc[idx]) or df['z'].iloc[idx] > -Z_THRESHOLD:
                                    confirm_pass = False
                                    break

                        if not np.isfinite(z):
                            stats['z_fail'] += 1; continue
                        if z > -Z_THRESHOLD:
                            stats['z_fail'] += 1; continue
                        if not vol_ok:
                            stats['vol_fail'] += 1; continue
                        if not trend_bull:
                            stats['trend_fail'] += 1; continue
                        if not confirm_pass:
                            stats['confirm_fail'] += 1; continue

                        remaining_capacity = max(0.0, (total_value * MAX_GROSS_EXPOSURE) - gross_exposure)
                        if remaining_capacity <= 0:
                            stats['size_fail'] += 1; continue

                        if not np.isfinite(exec_price) or exec_price <= 0:
                            stats['price_fail'] += 1; continue

                        vol = df['volatility'].iloc[i] if 'volatility' in df.columns else VOL_FLOOR
                        atr = df['atr'].iloc[i] if 'atr' in df.columns else ATR_FLOOR
                        if not np.isfinite(vol) or vol <= 0: vol = VOL_FLOOR
                        if not np.isfinite(atr) or atr <= 0: atr = ATR_FLOOR

                        dollar_risk_per_share = max(atr, exec_price * vol, ATR_FLOOR)
                        risk_budget = total_value * RISK_PER_TRADE

                        if dollar_risk_per_share <= 0 or not np.isfinite(dollar_risk_per_share):
                            stats['size_fail'] += 1; continue

                        approx_shares = int(math.floor(risk_budget / dollar_risk_per_share))
                        max_shares_by_fraction = int(math.floor((total_value * MAX_POSITION_FRACTION) / max(exec_price, 1e-6)))
                        approx_shares = max(0, min(approx_shares, max_shares_by_fraction))
                        if approx_shares <= 0:
                            stats['size_fail'] += 1; continue

                        allowed_value = min(approx_shares * exec_price, remaining_capacity)
                        n_shares = int(allowed_value // exec_price)
                        if n_shares <= 0:
                            stats['size_fail'] += 1; continue

                        cost = n_shares * exec_price * (1.0 + SLIPPAGE_PCT) + compute_commission(n_shares, exec_price, 'buy')
                        if cost > cash:
                            stats['cash_fail'] += 1; continue

                        if cost > cash * 0.75:
                            stats['size_fail'] += 1; continue

                        # EXECUTE buy -- store z-score inside position
                        cash -= cost
                        positions[t] = {'shares': n_shares, 'entry_price': exec_price, 'entry_time': df['timestamp'].iloc[i+1], 'z': float(z)}
                        entered_today[t] = True
                        stats['entries'] += 1

                # mark-to-market snapshot
                total_value = cash + sum(pos['shares'] * current_prices.get(sym, pos['entry_price'])
                                         for sym,pos in positions.items() if pos['shares'] != 0)
                ts = None
                for df in intraday.values():
                    if df is not None and i+1 < len(df):
                        ts = df['timestamp'].iloc[i+1]; break
                if ts is not None:
                    portfolio_history.append((ts, total_value))

                if cash < INITIAL_CAPITAL * 0.01:
                    print(f"[WARN] Cash very low: {cash:.2f} on {date_str} i={i}. Continuing but check sizing.")

            # End of day: close all positions
            for sym, pos in list(positions.items()):
                if pos['shares'] != 0:
                    df = intraday.get(sym)
                    if df is not None and len(df) > 0:
                        close_price = df['close'].iloc[-1]
                        gross = pos['shares'] * close_price * (1.0 - SLIPPAGE_PCT)
                        proceeds = gross - compute_commission(pos['shares'], close_price, 'sell')
                        cash += proceeds
                        trades.append({
                            'symbol': sym,
                            'entry_price': pos['entry_price'],
                            'exit_price': close_price,
                            'entry_time': pos.get('entry_time'),
                            'exit_time': pd.to_datetime(current_date) + pd.Timedelta(hours=16),
                            'shares': pos['shares'],
                            'pnl': (close_price - pos['entry_price']) * pos['shares'],
                            'pnl_pct': (close_price - pos['entry_price']) / (pos['entry_price'] if pos['entry_price']!=0 else 1e-8),
                            'z_score': pos.get('z', np.nan)
                        })
                        stats['eod_closes'] += 1
                    positions[sym] = {'shares':0, 'entry_price':0.0, 'entry_time':None, 'z':np.nan}

            portfolio_history.append((pd.to_datetime(current_date) + pd.Timedelta(hours=16), cash))

        except Exception as e:
            print("Exception on date", date_str, e)
            traceback.print_exc()
            portfolio_history.append((pd.to_datetime(current_date) + pd.Timedelta(hours=16), cash))
            continue

    idx = pd.DatetimeIndex([t for t,_ in portfolio_history])
    vals = [v for _,v in portfolio_history]
    series = pd.Series(vals, index=idx).sort_index().resample('1min').last().ffill().fillna(INITIAL_CAPITAL)
    trades_df = pd.DataFrame(trades)
    # return also daily_closes/daily_trends for reporting
    return series, trades_df, stats, daily_closes, daily_trends