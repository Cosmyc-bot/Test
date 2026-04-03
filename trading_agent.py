"""
SPY Market Bottom Detection Agent
==================================
Strategy: Mean Reversion & Sentiment Exhaustion
Target:    SPY (S&P 500 ETF)
Author:    Algorithmic Trading Agent

How Sentiment (VIX) is Weighed Against Technicals (RSI / Bollinger Bands):
---------------------------------------------------------------------------
The agent uses a composite scoring system. Each condition awards points:

  Condition                        Points  Weight Rationale
  ─────────────────────────────────────────────────────────
  Price ≤ Lower Bollinger Band        1    Price dislocation from mean
  RSI < 25                            1    Extreme oversold momentum
  Bullish RSI Divergence              1    Momentum loss in sellers (bonus)
  VIX > 30                            1    Crowd fear elevated
  VIX Peak (today < yesterday)        1    Fear likely turning (key timing)
  Selling Climax (Vol ≥ 1.5x MA20)   1    Capitulation volume
  Price > 2% below SMA200             1    "Rubber Band" stretched far

Minimum score to emit a BUY signal: 5 out of 7 points.

VIX contributes up to 2 points (elevated + peaking). Technicals contribute
up to 5 points. This means sentiment alone can never trigger a signal — it
acts as a multiplier/confirmer of the technical setup.  Conversely, a
technically perfect setup without elevated fear also won't fire.
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta  # pip install ta


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TICKER          = "SPY"
VIX_TICKER      = "^VIX"
LOOKBACK_YEARS  = 5          # Backtesting window
BB_PERIOD       = 20         # Bollinger Band period
BB_STD          = 2          # Bollinger Band std-dev multiplier
RSI_PERIOD      = 14         # RSI period
RSI_THRESHOLD   = 25         # Extreme oversold threshold
VIX_THRESHOLD   = 30         # Fear threshold
VOLUME_MULT     = 1.5        # Selling climax: vol must be ≥ this × MA20
SMA200_OFFSET   = 0.02       # 2% below 200-day SMA triggers rubber-band bonus
ATR_PERIOD      = 14         # ATR period for dynamic stop-loss
ATR_MULT        = 1.5        # Stop-loss = entry − ATR_MULT × ATR
MIN_SCORE       = 5          # Minimum signal score to trigger BUY (out of 7)
HOLD_DAYS       = 20         # Max holding period if neither TP nor SL hit


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHER
# ─────────────────────────────────────────────────────────────────────────────
def fetch_data(ticker: str, years: int = LOOKBACK_YEARS + 1) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def fetch_vix(years: int = LOOKBACK_YEARS + 1) -> pd.Series:
    """Download VIX close prices."""
    df = fetch_data(VIX_TICKER, years=years)
    return df["Close"].rename("VIX")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def compute_indicators(spy: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """
    Attach all required indicators to the SPY DataFrame.
    Returns an enriched DataFrame (rows aligned on date).
    """
    df = spy.copy()

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(
        close=df["Close"], window=BB_PERIOD, window_dev=BB_STD
    )
    df["BB_lower"]  = bb.bollinger_lband()
    df["BB_middle"] = bb.bollinger_mavg()   # 20-day SMA == TP target
    df["BB_upper"]  = bb.bollinger_hband()

    # ── RSI ──────────────────────────────────────────────────────────────────
    df["RSI"] = ta.momentum.RSIIndicator(
        close=df["Close"], window=RSI_PERIOD
    ).rsi()

    # ── ATR (for dynamic stop-loss) ──────────────────────────────────────────
    df["ATR"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=ATR_PERIOD
    ).average_true_range()

    # ── 200-day SMA ──────────────────────────────────────────────────────────
    df["SMA200"] = df["Close"].rolling(200).mean()

    # ── Volume MA20 ──────────────────────────────────────────────────────────
    df["Vol_MA20"] = df["Volume"].rolling(BB_PERIOD).mean()

    # ── VIX (aligned on trading days) ────────────────────────────────────────
    df = df.join(vix, how="left")
    df["VIX"] = df["VIX"].ffill()
    df["VIX_prev"] = df["VIX"].shift(1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# DIVERGENCE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
def detect_bullish_rsi_divergence(
    df: pd.DataFrame, window: int = 10
) -> pd.Series:
    """
    Bullish RSI Divergence:
      - Price makes a lower low  within `window` bars
      - RSI  makes a higher low  within the same period
    Returns a boolean Series indexed like df.
    """
    divergence = pd.Series(False, index=df.index)

    for i in range(window, len(df)):
        current_close = df["Close"].iloc[i]
        current_rsi   = df["RSI"].iloc[i]

        prior_close = df["Close"].iloc[i - window : i].min()
        prior_rsi   = df["RSI"].iloc[i - window : i].min()

        if pd.isna(current_rsi) or pd.isna(prior_rsi):
            continue

        price_lower_low = current_close < prior_close
        rsi_higher_low  = current_rsi   > prior_rsi

        divergence.iloc[i] = price_lower_low and rsi_higher_low

    return divergence


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL SCORER
# ─────────────────────────────────────────────────────────────────────────────
def score_row(row: pd.Series) -> dict:
    """
    Score a single bar (row) against all strategy conditions.
    Returns a dict with individual flags, total score, and component details.
    """
    scores = {}

    # 1. Price ≤ Lower Bollinger Band
    scores["bb_touch"] = (
        not pd.isna(row["BB_lower"]) and row["Close"] <= row["BB_lower"]
    )

    # 2. RSI < 25 (Extreme Oversold)
    scores["rsi_oversold"] = (
        not pd.isna(row["RSI"]) and row["RSI"] < RSI_THRESHOLD
    )

    # 3. Bullish RSI Divergence (pre-computed column)
    scores["rsi_divergence"] = bool(row.get("RSI_divergence", False))

    # 4. VIX > 30 (Elevated Fear)
    scores["vix_elevated"] = (
        not pd.isna(row["VIX"]) and row["VIX"] > VIX_THRESHOLD
    )

    # 5. VIX Peak: today < yesterday after a spike (fear turning)
    scores["vix_peak"] = (
        not pd.isna(row["VIX"]) and
        not pd.isna(row["VIX_prev"]) and
        row["VIX"] < row["VIX_prev"] and
        row["VIX_prev"] > VIX_THRESHOLD
    )

    # 6. Selling Climax (Volume ≥ 1.5× MA20)
    scores["selling_climax"] = (
        not pd.isna(row["Vol_MA20"]) and
        row["Volume"] >= VOLUME_MULT * row["Vol_MA20"]
    )

    # 7. Price > 2% below SMA200 (Rubber Band stretched)
    scores["rubber_band"] = (
        not pd.isna(row["SMA200"]) and
        row["Close"] < row["SMA200"] * (1 - SMA200_OFFSET)
    )

    total_score = sum(scores.values())
    signal      = "BUY" if total_score >= MIN_SCORE else "WAIT"

    return {**scores, "score": total_score, "signal": signal}


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATOR  (full DataFrame)
# ─────────────────────────────────────────────────────────────────────────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI divergence, then score every bar.
    Returns df augmented with signal columns.
    """
    df = df.copy()
    df["RSI_divergence"] = detect_bullish_rsi_divergence(df)

    results = df.apply(score_row, axis=1, result_type="expand")
    df = pd.concat([df, results], axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────
class Backtester:
    """
    Simple event-driven backtester for the market-bottom strategy.

    Entry  : Day after a BUY signal (open price).
    Exit   : First of —
               • Take-Profit  : Close ≥ BB_middle (20-day SMA)
               • Stop-Loss    : Close ≤ entry − ATR_MULT × ATR
               • Time-stop    : After HOLD_DAYS bars if neither hit
    """

    def __init__(self, df: pd.DataFrame):
        self.df     = df.copy()
        self.trades = []

    def run(self) -> pd.DataFrame:
        """Execute the backtest and return a DataFrame of trades."""
        df      = self.df
        in_trade = False
        entry_price = stop_loss = take_profit = None
        entry_date  = None
        entry_atr   = None
        hold_count  = 0

        for i in range(1, len(df)):
            row = df.iloc[i]

            if not in_trade:
                # Check previous bar for a BUY signal (enter at today's open)
                prev = df.iloc[i - 1]
                if prev.get("signal") == "BUY":
                    in_trade    = True
                    entry_price = row["Open"]
                    entry_date  = row.name
                    entry_atr   = prev["ATR"]
                    stop_loss   = entry_price - ATR_MULT * entry_atr
                    take_profit = prev["BB_middle"]   # mean reversion target
                    hold_count  = 0
            else:
                hold_count += 1
                exit_price = None
                exit_reason = None

                # Take-Profit check (using Close)
                if row["Close"] >= take_profit:
                    exit_price  = take_profit
                    exit_reason = "TP"

                # Stop-Loss check (using Close; intraday would use Low)
                elif row["Close"] <= stop_loss:
                    exit_price  = stop_loss
                    exit_reason = "SL"

                # Time-stop
                elif hold_count >= HOLD_DAYS:
                    exit_price  = row["Close"]
                    exit_reason = "TIME"

                if exit_price is not None:
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    self.trades.append({
                        "entry_date":   entry_date,
                        "exit_date":    row.name,
                        "entry_price":  round(entry_price, 2),
                        "exit_price":   round(exit_price, 2),
                        "stop_loss":    round(stop_loss, 2),
                        "take_profit":  round(take_profit, 2),
                        "hold_days":    hold_count,
                        "pnl_pct":      round(pnl_pct, 2),
                        "exit_reason":  exit_reason,
                        "score":        df.iloc[i - hold_count - 1 + 1].get("score", 0),
                    })
                    in_trade = False

        return pd.DataFrame(self.trades)

    def summary(self, trades: pd.DataFrame) -> dict:
        """Compute summary statistics for the backtest results."""
        if trades.empty:
            return {"error": "No trades generated in backtest period."}

        total  = len(trades)
        wins   = (trades["pnl_pct"] > 0).sum()
        losses = total - wins

        avg_win  = trades.loc[trades["pnl_pct"] > 0, "pnl_pct"].mean() if wins   else 0
        avg_loss = trades.loc[trades["pnl_pct"] < 0, "pnl_pct"].mean() if losses else 0

        # Profit Factor = gross profit / gross loss (absolute)
        gross_profit = trades.loc[trades["pnl_pct"] > 0, "pnl_pct"].sum()
        gross_loss   = abs(trades.loc[trades["pnl_pct"] < 0, "pnl_pct"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Expectancy per trade (in %)
        expectancy = trades["pnl_pct"].mean()

        # Max drawdown (peak-to-trough on cumulative PnL)
        cumulative = trades["pnl_pct"].cumsum()
        rolling_max = cumulative.cummax()
        drawdown    = cumulative - rolling_max
        max_dd      = drawdown.min()

        exit_counts = trades["exit_reason"].value_counts().to_dict()

        return {
            "total_trades":   total,
            "win_rate_%":     round(wins / total * 100, 1),
            "avg_win_%":      round(avg_win, 2),
            "avg_loss_%":     round(avg_loss, 2),
            "profit_factor":  round(profit_factor, 2),
            "expectancy_%":   round(expectancy, 2),
            "max_drawdown_%": round(max_dd, 2),
            "total_pnl_%":    round(trades["pnl_pct"].sum(), 2),
            "exit_breakdown": exit_counts,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LIVE SIGNAL REPORTER
# ─────────────────────────────────────────────────────────────────────────────
def print_live_signal(df: pd.DataFrame) -> None:
    """Print the signal assessment for the most recent available trading day."""
    latest = df.iloc[-1]
    date   = latest.name.strftime("%Y-%m-%d")

    conditions = [
        ("Price ≤ Lower Bollinger Band",   latest.get("bb_touch", False)),
        ("RSI < 25 (Extreme Oversold)",     latest.get("rsi_oversold", False)),
        ("Bullish RSI Divergence",          latest.get("rsi_divergence", False)),
        ("VIX > 30 (Elevated Fear)",        latest.get("vix_elevated", False)),
        ("VIX Peaking (fear turning)",      latest.get("vix_peak", False)),
        ("Selling Climax (Vol ≥ 1.5×MA20)", latest.get("selling_climax", False)),
        ("Price > 2% below SMA200",         latest.get("rubber_band", False)),
    ]

    score  = latest.get("score", 0)
    signal = latest.get("signal", "WAIT")

    width = 60
    print("\n" + "═" * width)
    print(f"  SPY Market-Bottom Agent — Signal Report")
    print(f"  Date: {date}")
    print("═" * width)

    print(f"\n  {'Condition':<40} {'Status':>8}")
    print("  " + "─" * (width - 4))
    for label, flag in conditions:
        status = "✓  TRUE" if flag else "✗  false"
        print(f"  {label:<40} {status:>8}")

    print("  " + "─" * (width - 4))
    print(f"\n  Composite Score : {score} / {len(conditions)}")
    print(f"  Minimum to BUY  : {MIN_SCORE}")
    print()

    if signal == "BUY":
        entry   = round(float(latest["Close"]), 2)
        atr     = latest["ATR"]
        sl      = round(entry - ATR_MULT * float(atr), 2) if not pd.isna(atr) else "N/A"
        tp      = round(float(latest["BB_middle"]), 2) if not pd.isna(latest["BB_middle"]) else "N/A"
        rr      = round((tp - entry) / (entry - sl), 2) if isinstance(sl, float) and isinstance(tp, float) else "N/A"

        print(f"  ╔{'═' * (width - 6)}╗")
        print(f"  ║  *** SIGNAL: BUY ***{' ' * (width - 26)}║")
        print(f"  ║  Entry  ~  ${entry:<10} (current close){' ' * (width - 48)}║")
        print(f"  ║  Stop   ~  ${sl:<10} (−{ATR_MULT}× ATR){' ' * (width - 47)}║")
        print(f"  ║  Target ~  ${tp:<10} (20-day SMA){' ' * (width - 46)}║")
        print(f"  ║  R:R   ~   {rr}x{' ' * (width - 17)}║")
        print(f"  ╚{'═' * (width - 6)}╝")
    else:
        print(f"  ┌{'─' * (width - 6)}┐")
        print(f"  │  SIGNAL: WAIT  —  conditions not met ({score}/{MIN_SCORE})  │")
        print(f"  └{'─' * (width - 6)}┘")

    print()

    # Key metrics snapshot
    print(f"  Key Metrics (latest bar):")
    print(f"    SPY Close  : ${float(latest['Close']):.2f}")
    rsi_val = latest.get("RSI", float("nan"))
    vix_val = latest.get("VIX", float("nan"))
    bb_l    = latest.get("BB_lower", float("nan"))
    sma200  = latest.get("SMA200", float("nan"))
    atr_val = latest.get("ATR", float("nan"))
    print(f"    RSI (14)   : {rsi_val:.1f}" if not pd.isna(rsi_val) else "    RSI (14)   : N/A")
    print(f"    VIX        : {vix_val:.1f}" if not pd.isna(vix_val) else "    VIX        : N/A")
    print(f"    BB Lower   : ${bb_l:.2f}"   if not pd.isna(bb_l)    else "    BB Lower   : N/A")
    print(f"    SMA200     : ${sma200:.2f}"  if not pd.isna(sma200)  else "    SMA200     : N/A")
    print(f"    ATR (14)   : ${atr_val:.2f}" if not pd.isna(atr_val) else "    ATR (14)   : N/A")
    print("═" * width + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────
def print_backtest_report(trades: pd.DataFrame, stats: dict) -> None:
    """Pretty-print backtest results."""
    width = 60
    print("\n" + "═" * width)
    print("  BACKTEST RESULTS  —  SPY Market-Bottom Strategy")
    print(f"  Period : last {LOOKBACK_YEARS} years   |   Ticker: {TICKER}")
    print("═" * width)

    if "error" in stats:
        print(f"\n  {stats['error']}\n")
        return

    print(f"\n  {'Metric':<32} {'Value':>10}")
    print("  " + "─" * (width - 4))

    rows = [
        ("Total Trades",        stats["total_trades"]),
        ("Win Rate",            f"{stats['win_rate_%']}%"),
        ("Avg Win",             f"{stats['avg_win_%']}%"),
        ("Avg Loss",            f"{stats['avg_loss_%']}%"),
        ("Profit Factor",       stats["profit_factor"]),
        ("Expectancy / Trade",  f"{stats['expectancy_%']}%"),
        ("Max Drawdown",        f"{stats['max_drawdown_%']}%"),
        ("Total PnL (sum %)",   f"{stats['total_pnl_%']}%"),
    ]

    for label, value in rows:
        print(f"  {label:<32} {str(value):>10}")

    print("  " + "─" * (width - 4))
    print(f"\n  Exit Breakdown:")
    for reason, count in stats.get("exit_breakdown", {}).items():
        print(f"    {reason:<12}: {count} trades")

    if not trades.empty:
        print(f"\n  Last 5 trades:")
        cols = ["entry_date", "exit_date", "entry_price", "exit_price", "pnl_pct", "exit_reason"]
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", width + 40)
        print(trades[cols].tail(5).to_string(index=False))

    print("═" * width + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\nFetching data …")
    spy = fetch_data(TICKER)
    vix = fetch_vix()

    print("Computing indicators …")
    df = compute_indicators(spy, vix)

    print("Generating signals …")
    df = generate_signals(df)

    # ── Trim to backtest window (keep extra for indicator warm-up) ────────────
    cutoff = datetime.today() - timedelta(days=LOOKBACK_YEARS * 365)
    df_bt  = df[df.index >= cutoff].copy()

    # ── Backtest ──────────────────────────────────────────────────────────────
    print("Running backtest …\n")
    bt     = Backtester(df_bt)
    trades = bt.run()
    stats  = bt.summary(trades)
    print_backtest_report(trades, stats)

    # ── Live / latest signal ──────────────────────────────────────────────────
    print_live_signal(df)


if __name__ == "__main__":
    main()
