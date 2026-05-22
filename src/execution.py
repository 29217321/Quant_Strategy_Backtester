"""
QuantX V4.9 — main entry point. Runs the backtest and writes reports to OUT_DIR.
"""
import os
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from strategy import (
    TICKERS, START_DATE, END_DATE, RUN_MODE,
    OUT_DIR, GENERATE_PDF, MAX_TICKER_CHARTS,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
)
from backtest import run_backtest
from reporting import summarize_performance


# ----------
# Run (main)
# ----------
if __name__ == "__main__":
    # run backtest
    series, trades_df, stats, daily_closes, daily_trends = run_backtest(TICKERS, START_DATE, END_DATE)

    perf = summarize_performance(series, trades_df)

    # Export trades CSV in MD's exact format
    if not trades_df.empty:
        trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
        def safe_daily_sma(row):
            try:
                sym = row['symbol']; d = pd.Timestamp(row['date'])
                if sym in daily_trends and d in daily_trends[sym].index:
                    return daily_trends[sym].loc[d]
            except Exception:
                return np.nan
            return np.nan
        trades_df['daily_sma'] = trades_df.apply(safe_daily_sma, axis=1)
        trades_df['stop_loss'] = trades_df['entry_price'] * (1 - STOP_LOSS_PCT)
        trades_df['target'] = trades_df['entry_price'] * (1 + TAKE_PROFIT_PCT)
        # z_score should already be present from run_backtest; if not, fill with NaN
        trades_df['z_score'] = trades_df.get('z_score', np.nan).astype(float)

        final_trades = trades_df[['entry_time','symbol','entry_price','stop_loss','target','pnl','daily_sma','z_score']].copy()
        final_trades.columns = ['TimeOfSignal','Ticker','EntryPrice','StopLoss','TargetPrice','RealizedPnL','DailySMAValue','ZScore']
        out_csv = os.path.join(OUT_DIR, "QuantX_V4.9_Trades_Report.csv")
        final_trades.to_csv(out_csv, index=False)
        print(f"✅ Trade report exported: {out_csv}")
    else:
        final_trades = pd.DataFrame(columns=['TimeOfSignal','Ticker','EntryPrice','StopLoss','TargetPrice','RealizedPnL','DailySMAValue','ZScore'])
        print("ℹ️ No trades executed — empty trade report created.")

    # Diagnostics summary printed
    print("\n--- Diagnostics Summary ---")
    print(f"entries: {stats.get('entries',0)}")
    print(f"intraday_exits: {stats.get('intraday_exits',0)}")
    print(f"eod_closes: {stats.get('eod_closes',0)}")
    print(f"z_fail: {stats.get('z_fail',0)}")
    print(f"vol_fail: {stats.get('vol_fail',0)}")
    print(f"trend_fail: {stats.get('trend_fail',0)}")
    print(f"confirm_fail: {stats.get('confirm_fail',0)}")
    print(f"size_fail: {stats.get('size_fail',0)}")
    print(f"cash_fail: {stats.get('cash_fail',0)}")
    print(f"price_fail: {stats.get('price_fail',0)}")
    print(f"missing_data: {stats.get('missing_data',0)}")
    print("----------------------------\n")

    print(f"Initial capital: ${series.iloc[0]:,.2f}")
    print(f"Final capital:   ${series.iloc[-1]:,.2f}")
    print(f"Total return:    {perf['total_return']*100:.2f}%")
    print(f"Annualized:      {perf['annualized_return']*100:.2f}%")
    print(f"Max drawdown:    {perf['max_drawdown']*100:.2f}%")
    print(f"Sharpe (ann):    {perf['sharpe']:.2f}")
    print(f"Trades executed: {len(trades_df)}\n")

    if not trades_df.empty:
        win_rate = (trades_df['pnl'] > 0).mean()
        avg_win = trades_df.loc[trades_df['pnl']>0, 'pnl'].mean()
        avg_loss = trades_df.loc[trades_df['pnl']<=0, 'pnl'].mean()
        print(f"Win rate:        {win_rate*100:.2f}%")
        print(f"Avg Win:         {avg_win:.2f},  Avg Loss: {avg_loss:.2f}")

    # Save equity & PnL hist
    try:
        eq_png = os.path.join(OUT_DIR, "QuantX_V4.9_EquityCurve.png")
        plt.figure(figsize=(12,5))
        plt.plot(series.index, series.values, linewidth=1)
        plt.title(f"Equity Curve ({RUN_MODE}): {START_DATE} → {END_DATE}")
        plt.ylabel("Portfolio value ($)")
        plt.grid(True)
        plt.savefig(eq_png, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"✅ Equity curve saved: {eq_png}")

        if not trades_df.empty:
            hist_png = os.path.join(OUT_DIR, "QuantX_V4.9_PnL_Hist.png")
            plt.figure(figsize=(8,4))
            plt.hist(trades_df['pnl'].dropna(), bins=50)
            plt.title("Trade PnL Distribution")
            plt.xlabel("PnL")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(hist_png, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"✅ PnL hist saved: {hist_png}")
    except Exception as e:
        print("Failed to save equity/pnl charts:", e)

    # Per-ticker charts (one chart per ticker with all entries/exits marked)
    saved_charts = []
    try:
        traded_tickers = sorted(final_trades['Ticker'].unique()) if not final_trades.empty else []
        # If you want *all* tickers regardless of trades, change to TICKERS
        tickers_to_plot = list(traded_tickers)[:MAX_TICKER_CHARTS]

        for sym in tickers_to_plot:
            sym_trades = final_trades[final_trades['Ticker'] == sym]
            # build daily series for plotting if available, else skip
            if sym not in daily_closes or daily_closes[sym].dropna().empty:
                # skip if no daily data
                continue
            price_series = daily_closes[sym].dropna()
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(price_series.index, price_series.values, '-', linewidth=1, label='Daily Close')
            # plot markers for each trade
            for _, r in sym_trades.iterrows():
                try:
                    t_entry = pd.to_datetime(r['TimeOfSignal'])
                except Exception:
                    t_entry = pd.to_datetime(r['TimeOfSignal'], errors='coerce')
                entry_y = r['EntryPrice']
                ax.scatter([t_entry], [entry_y], marker='^', color='green', s=50, label='Entry' if _==sym_trades.index[0] else "")
                # for exit, we don't have explicit exit_time in final_trades columns (we stripped it earlier)
                # find original record in trades_df to get exit_time/exit_price
                orig = trades_df[(trades_df['symbol']==sym) & (pd.to_datetime(trades_df['entry_time'])==t_entry)]
                if not orig.empty:
                    exit_time = orig['exit_time'].iloc[0]
                    exit_price = orig['exit_price'].iloc[0]
                    try:
                        exit_time = pd.to_datetime(exit_time)
                    except Exception:
                        exit_time = pd.to_datetime(exit_time, errors='coerce')
                    ax.scatter([exit_time], [exit_price], marker='v', color='red', s=50, label='Exit' if _==sym_trades.index[0] else "")
            ax.set_title(f"{sym} | All trades ({len(sym_trades)} trades)")
            ax.legend(loc='upper left')
            ax.grid(True)
            fname = os.path.join(OUT_DIR, f"chart_{sym}_all_trades.png")
            fig.savefig(fname, bbox_inches='tight', dpi=150)
            plt.close(fig)
            saved_charts.append(fname)
        print(f"✅ Per-ticker charts saved (up to {MAX_TICKER_CHARTS}) in {OUT_DIR}")
    except Exception as e:
        print("Failed to save per-ticker charts:", e)
        traceback.print_exc()

    # Optional: assemble PDF
    try:
        if GENERATE_PDF:
            pdf_path = os.path.join(OUT_DIR, "QuantX_V4.9_Report.pdf")
            with PdfPages(pdf_path) as pdf:
                # first page: text summary
                fig, ax = plt.subplots(figsize=(11,8.5))
                ax.axis('off')
                txt = f"QuantX V4.9 Report\nMode: {RUN_MODE}\nDate range: {START_DATE} → {END_DATE}\n\n"
                txt += f"Initial capital: ${series.iloc[0]:,.2f}\nFinal capital: ${series.iloc[-1]:,.2f}\nTotal return: {perf['total_return']*100:.2f}%\nAnnualized: {perf['annualized_return']*100:.2f}%\nMax drawdown: {perf['max_drawdown']*100:.2f}%\nSharpe (ann): {perf['sharpe']:.2f}\n\n"
                txt += f"Trades executed: {len(trades_df)}\nEntries: {stats.get('entries',0)}  intraday exits: {stats.get('intraday_exits',0)}  eod closes: {stats.get('eod_closes',0)}\n"
                ax.text(0.01, 0.99, txt, va='top', ha='left', fontsize=10, family='monospace')
                pdf.savefig(fig); plt.close(fig)

                # equity figure
                if os.path.exists(eq_png):
                    fig = plt.figure(figsize=(11,6))
                    img = plt.imread(eq_png)
                    plt.imshow(img); plt.axis('off')
                    pdf.savefig(fig); plt.close(fig)

                # per-ticker charts (append)
                for p in saved_charts:
                    fig = plt.figure(figsize=(11,6))
                    img = plt.imread(p)
                    plt.imshow(img); plt.axis('off')
                    pdf.savefig(fig); plt.close(fig)

                # include small table page (top 20 trades)
                if not final_trades.empty:
                    fig, ax = plt.subplots(figsize=(11,8.5))
                    ax.axis('off')
                    sample_table = final_trades.head(40)
                    table = ax.table(cellText=sample_table.values,
                                     colLabels=sample_table.columns,
                                     cellLoc='center',
                                     loc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1, 1.5)
                    pdf.savefig(fig); plt.close(fig)

            print(f"✅ Combined PDF report saved: {pdf_path}")
    except Exception as e:
        print("Failed to assemble PDF:", e)