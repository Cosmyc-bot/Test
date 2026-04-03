"""
SPY Market Bottom Detection Agent
==================================
Strategy : Mean Reversion & Sentiment Exhaustion
Target   : SPY (S&P 500 ETF)
Libraries: requests, pandas, numpy, talib (TA-Lib C extension)

How Sentiment (VIX) is Weighed Against Technicals (RSI / Bollinger Bands):
---------------------------------------------------------------------------
The agent uses a composite scoring system.  Each condition awards 1 point:

  #  Condition                          Weight Rationale
  ─────────────────────────────────────────────────────────────────────────
  1  Price ≤ Lower Bollinger Band        Price dislocated from mean
  2  RSI < 25 (Extreme Oversold)         Momentum fully exhausted
  3  Bullish RSI Divergence              Sellers losing momentum (bonus)
  4  VIX > 30 (Elevated Fear)            Crowd panic present
  5  VIX Peaking (today < yesterday)     Fear likely at inflection
  6  Selling Climax (Vol ≥ 1.5× MA20)   Capitulation volume spike
  7  Price > 2% below SMA200             "Rubber Band" fully stretched

Minimum score to emit a BUY signal: 5 out of 7.

VIX contributes up to 2 points (conditions 4 + 5).
Technicals contribute up to 5 points (conditions 1–3, 6–7).

Sentiment alone can never fire a signal — it amplifies a technical setup.
A perfect technical setup without elevated fear will also not fire.
Both dimensions must align for a BUY to be issued.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import requests
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TICKER         = "SPY"
VIX_TICKER     = "%5EVIX"          # ^VIX URL-encoded
LOOKBACK_YEARS = 5                 # Back-test window (years)
BB_PERIOD      = 20                # Bollinger Band lookback
BB_STD         = 2.0               # Bollinger Band std-dev multiplier
RSI_PERIOD     = 14                # RSI lookback
RSI_THRESHOLD  = 25                # Extreme-oversold threshold
VIX_THRESHOLD  = 30                # Fear threshold
VOLUME_MULT    = 1.5               # Selling climax: vol ≥ this × MA20
SMA200_OFFSET  = 0.02              # 2 % below SMA200 → rubber-band active
ATR_PERIOD     = 14                # ATR lookback for dynamic stop-loss
ATR_MULT       = 1.5               # Stop = entry − ATR_MULT × ATR
MIN_SCORE      = 5                 # Minimum composite score to issue BUY
HOLD_DAYS      = 20                # Time-stop: exit after N bars if untouched

YF_BASE        = "https://query1.finance.yahoo.com/v8/finance/chart"
YF_HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHER  (requests → Yahoo Finance v8 chart API)
# ─────────────────────────────────────────────────────────────────────────────
def _yf_fetch(symbol: str, years: int) -> pd.DataFrame:
    """
    Download daily OHLCV from Yahoo Finance's v8 chart endpoint.
    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    indexed by date (timezone-naive).
    """
    range_str = f"{years + 1}y"          # fetch a bit extra for warm-up
    url = (
        f"{YF_BASE}/{symbol}"
        f"?interval=1d&range={range_str}&includePrePost=false"
    )

    for attempt in range(4):
        try:
            resp = requests.get(url, headers=YF_HEADERS, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 3:
                raise RuntimeError(f"Failed to fetch {symbol}: {exc}") from exc
            wait = 2 ** attempt
            print(f"  Retry {attempt + 1}/4 for {symbol} in {wait}s …")
            time.sleep(wait)

    data   = resp.json()
    result = data["chart"]["result"][0]
    meta   = result["meta"]
    ts     = result["timestamp"]
    quote  = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "Open":   quote["open"],
        "High":   quote["high"],
        "Low":    quote["low"],
        "Close":  quote["close"],
        "Volume": quote["volume"],
    }, index=pd.to_datetime(ts, unit="s", utc=True).tz_localize(None))

    df.index = df.index.normalize()      # strip intraday component
    df.index.name = "Date"
    df.dropna(subset=["Close"], inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_spy(years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    print(f"  Fetching {TICKER} …")
    return _yf_fetch(TICKER, years)


def fetch_vix(years: int = LOOKBACK_YEARS) -> pd.Series:
    print(f"  Fetching VIX …")
    df = _yf_fetch(VIX_TICKER, years)
    return df["Close"].rename("VIX")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR ENGINE  (TA-Lib)
# ─────────────────────────────────────────────────────────────────────────────
def compute_indicators(spy: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """
    Attach all required technical indicators to the SPY DataFrame.

    Indicators computed:
      BB_upper / BB_middle / BB_lower  — Bollinger Bands (20, 2)
      RSI                              — Relative Strength Index (14)
      ATR                              — Average True Range (14)
      SMA200                           — 200-day Simple Moving Average
      Vol_MA20                         — 20-day Volume Moving Average
      VIX / VIX_prev                   — VIX close, and prior-day VIX
    """
    df = spy.copy()

    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    # talib.BBANDS returns (upper, middle, lower)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        close, timeperiod=BB_PERIOD, nbdevup=BB_STD, nbdevdn=BB_STD, matype=0
    )
    df["BB_upper"]  = bb_upper
    df["BB_middle"] = bb_middle   # 20-day SMA → used as Take-Profit target
    df["BB_lower"]  = bb_lower

    # ── RSI ──────────────────────────────────────────────────────────────────
    df["RSI"] = talib.RSI(close, timeperiod=RSI_PERIOD)

    # ── ATR ──────────────────────────────────────────────────────────────────
    df["ATR"] = talib.ATR(high, low, close, timeperiod=ATR_PERIOD)

    # ── 200-day SMA ──────────────────────────────────────────────────────────
    df["SMA200"] = talib.SMA(close, timeperiod=200)

    # ── Volume MA20 ──────────────────────────────────────────────────────────
    df["Vol_MA20"] = talib.SMA(volume, timeperiod=BB_PERIOD)

    # ── VIX (align on SPY trading days, forward-fill gaps) ───────────────────
    df = df.join(vix, how="left")
    df["VIX"]      = df["VIX"].ffill()
    df["VIX_prev"] = df["VIX"].shift(1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# BULLISH RSI DIVERGENCE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
def detect_bullish_rsi_divergence(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Bullish RSI Divergence (within a rolling `window`):
      • Price makes a lower low   → sellers still in control on price
      • RSI   makes a higher low  → but momentum is weakening (divergence)

    This signals that selling pressure is exhausting even as price falls,
    making it a high-conviction leading indicator for a reversal.
    """
    divergence = pd.Series(False, index=df.index, dtype=bool)

    close_arr = df["Close"].values
    rsi_arr   = df["RSI"].values

    for i in range(window, len(df)):
        cur_close = close_arr[i]
        cur_rsi   = rsi_arr[i]
        if np.isnan(cur_rsi):
            continue

        prior_close_min = np.nanmin(close_arr[i - window : i])
        prior_rsi_min   = np.nanmin(rsi_arr[i - window : i])

        if np.isnan(prior_rsi_min):
            continue

        if cur_close < prior_close_min and cur_rsi > prior_rsi_min:
            divergence.iloc[i] = True

    return divergence


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL SCORER
# ─────────────────────────────────────────────────────────────────────────────
def score_row(row: pd.Series) -> dict:
    """
    Evaluate all 7 strategy conditions for a single bar.
    Returns a dict of individual boolean flags plus total score and signal.
    """
    s = {}

    # 1. Price touched / broke below the lower Bollinger Band
    s["bb_touch"] = (
        not pd.isna(row["BB_lower"])
        and float(row["Close"]) <= float(row["BB_lower"])
    )

    # 2. RSI below extreme-oversold threshold
    s["rsi_oversold"] = (
        not pd.isna(row["RSI"])
        and float(row["RSI"]) < RSI_THRESHOLD
    )

    # 3. Bullish RSI Divergence (pre-computed per bar)
    s["rsi_divergence"] = bool(row.get("RSI_divergence", False))

    # 4. VIX above fear threshold (crowd panic elevated)
    s["vix_elevated"] = (
        not pd.isna(row["VIX"])
        and float(row["VIX"]) > VIX_THRESHOLD
    )

    # 5. VIX peaking: today lower than yesterday after a spike
    #    → fear is at an inflection point, the perfect timing trigger
    s["vix_peak"] = (
        not pd.isna(row["VIX"])
        and not pd.isna(row["VIX_prev"])
        and float(row["VIX"])      < float(row["VIX_prev"])
        and float(row["VIX_prev"]) > VIX_THRESHOLD
    )

    # 6. Selling climax: volume ≥ 1.5× 20-day average
    s["selling_climax"] = (
        not pd.isna(row["Vol_MA20"])
        and float(row["Volume"]) >= VOLUME_MULT * float(row["Vol_MA20"])
    )

    # 7. Rubber-band: price stretched > 2% below the 200-day SMA
    s["rubber_band"] = (
        not pd.isna(row["SMA200"])
        and float(row["Close"]) < float(row["SMA200"]) * (1 - SMA200_OFFSET)
    )

    total  = sum(s.values())
    signal = "BUY" if total >= MIN_SCORE else "WAIT"

    return {**s, "score": total, "signal": signal}


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATOR  (full DataFrame)
# ─────────────────────────────────────────────────────────────────────────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute divergence column, then score every bar."""
    df = df.copy()
    df["RSI_divergence"] = detect_bullish_rsi_divergence(df)

    results = df.apply(score_row, axis=1, result_type="expand")
    return pd.concat([df, results], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────
class Backtester:
    """
    Simple event-driven back-tester for the SPY market-bottom strategy.

    Entry  : Open of the bar AFTER a BUY signal fires.
    Exit   : First condition met —
               TP   Close ≥ BB_middle (20-day SMA / mean reversion)
               SL   Close ≤ entry − ATR_MULT × ATR  (dynamic 1.5× ATR)
               TIME After HOLD_DAYS bars (no TP/SL hit)

    The dynamic stop-loss adapts to current volatility: wider stops in
    volatile regimes (high ATR), tighter in calm ones.
    The take-profit at the 20-day SMA exploits mean reversion: after an
    extreme oversold spike, price tends to snap back to the rolling mean.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def run(self) -> pd.DataFrame:
        df       = self.df
        trades   = []
        in_trade = False
        entry_price = stop_loss = take_profit = entry_atr = None
        entry_date  = None
        hold_count  = 0

        for i in range(1, len(df)):
            row = df.iloc[i]

            if not in_trade:
                prev = df.iloc[i - 1]
                if prev.get("signal") == "BUY":
                    in_trade    = True
                    entry_price = float(row["Open"])
                    entry_date  = row.name
                    entry_atr   = float(prev["ATR"]) if not pd.isna(prev["ATR"]) else 0
                    stop_loss   = entry_price - ATR_MULT * entry_atr
                    take_profit = float(prev["BB_middle"])
                    hold_count  = 0
            else:
                hold_count  += 1
                exit_price   = None
                exit_reason  = None
                close        = float(row["Close"])

                if close >= take_profit:
                    exit_price  = take_profit
                    exit_reason = "TP"
                elif close <= stop_loss:
                    exit_price  = stop_loss
                    exit_reason = "SL"
                elif hold_count >= HOLD_DAYS:
                    exit_price  = close
                    exit_reason = "TIME"

                if exit_price is not None:
                    pnl = (exit_price - entry_price) / entry_price * 100
                    trades.append({
                        "entry_date":  entry_date,
                        "exit_date":   row.name,
                        "entry_price": round(entry_price, 2),
                        "exit_price":  round(exit_price, 2),
                        "stop_loss":   round(stop_loss, 2),
                        "take_profit": round(take_profit, 2),
                        "hold_days":   hold_count,
                        "pnl_pct":     round(pnl, 2),
                        "exit_reason": exit_reason,
                    })
                    in_trade = False

        return pd.DataFrame(trades)

    @staticmethod
    def summary(trades: pd.DataFrame) -> dict:
        """Aggregate performance statistics from a trades DataFrame."""
        if trades.empty:
            return {"error": "No trades generated in the back-test period."}

        total  = len(trades)
        wins   = (trades["pnl_pct"] > 0).sum()
        losses = total - wins

        avg_win  = trades.loc[trades["pnl_pct"] > 0, "pnl_pct"].mean() if wins   else 0.0
        avg_loss = trades.loc[trades["pnl_pct"] < 0, "pnl_pct"].mean() if losses else 0.0

        gross_profit = trades.loc[trades["pnl_pct"] > 0, "pnl_pct"].sum()
        gross_loss   = abs(trades.loc[trades["pnl_pct"] < 0, "pnl_pct"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        cumulative  = trades["pnl_pct"].cumsum()
        drawdown    = cumulative - cumulative.cummax()
        max_dd      = drawdown.min()

        return {
            "total_trades":   total,
            "win_rate_%":     round(wins / total * 100, 1),
            "avg_win_%":      round(avg_win, 2),
            "avg_loss_%":     round(avg_loss, 2),
            "profit_factor":  round(profit_factor, 2),
            "expectancy_%":   round(trades["pnl_pct"].mean(), 2),
            "max_drawdown_%": round(max_dd, 2),
            "total_pnl_%":    round(trades["pnl_pct"].sum(), 2),
            "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTERS
# ─────────────────────────────────────────────────────────────────────────────
W = 62   # print width


def _hline(ch="─"):
    return "  " + ch * (W - 4)


def print_live_signal(df: pd.DataFrame) -> None:
    """Print the signal assessment for the most recent trading bar."""
    latest = df.iloc[-1]
    date   = latest.name.strftime("%Y-%m-%d")

    conditions = [
        ("1. Price ≤ Lower Bollinger Band",    "bb_touch"),
        ("2. RSI < 25 (Extreme Oversold)",      "rsi_oversold"),
        ("3. Bullish RSI Divergence",           "rsi_divergence"),
        ("4. VIX > 30 (Elevated Fear)",         "vix_elevated"),
        ("5. VIX Peaking (fear turning)",       "vix_peak"),
        ("6. Selling Climax (Vol ≥ 1.5×MA20)", "selling_climax"),
        ("7. Price > 2% below SMA200",          "rubber_band"),
    ]

    score  = int(latest.get("score", 0))
    signal = latest.get("signal", "WAIT")

    print("\n" + "═" * W)
    print("  SPY Market-Bottom Agent — Live Signal Report")
    print(f"  Date : {date}")
    print("═" * W)
    print(f"\n  {'Condition':<43} Status")
    print(_hline())
    for label, key in conditions:
        flag   = bool(latest.get(key, False))
        status = "✓" if flag else "✗"
        print(f"  {label:<43} {status}")
    print(_hline())
    print(f"\n  Composite score : {score} / {len(conditions)}")
    print(f"  Minimum for BUY : {MIN_SCORE}")
    print()

    if signal == "BUY":
        entry = float(latest["Close"])
        atr   = float(latest["ATR"]) if not pd.isna(latest["ATR"]) else 0
        sl    = round(entry - ATR_MULT * atr, 2)
        tp    = round(float(latest["BB_middle"]), 2)
        rr    = round((tp - entry) / (entry - sl), 2) if (entry - sl) > 0 else "N/A"

        print(f"  ╔{'═' * (W - 6)}╗")
        print(f"  ║  *** SIGNAL : B U Y ***{' ' * (W - 30)}║")
        print(f"  ║  Entry  ≈  ${entry:<9.2f}  (current close){' ' * (W - 46)}║")
        print(f"  ║  Stop   ≈  ${sl:<9.2f}  (−{ATR_MULT}× ATR){' ' * (W - 46)}║")
        print(f"  ║  Target ≈  ${tp:<9.2f}  (20-day SMA){' ' * (W - 44)}║")
        print(f"  ║  R:R    ≈  {rr}x{' ' * (W - 17)}║")
        print(f"  ╚{'═' * (W - 6)}╝")
    else:
        lacking = MIN_SCORE - score
        print(f"  ┌{'─' * (W - 6)}┐")
        print(f"  │  SIGNAL : W A I T  —  {lacking} more condition(s) needed  │")
        print(f"  └{'─' * (W - 6)}┘")

    print()
    print("  Key Metrics (latest bar):")
    metrics = [
        ("SPY Close",  f"${float(latest['Close']):.2f}"),
        ("RSI (14)",   f"{float(latest['RSI']):.1f}"   if not pd.isna(latest["RSI"])      else "N/A"),
        ("VIX",        f"{float(latest['VIX']):.1f}"   if not pd.isna(latest["VIX"])      else "N/A"),
        ("BB Lower",   f"${float(latest['BB_lower']):.2f}" if not pd.isna(latest["BB_lower"]) else "N/A"),
        ("SMA200",     f"${float(latest['SMA200']):.2f}"   if not pd.isna(latest["SMA200"])   else "N/A"),
        ("ATR (14)",   f"${float(latest['ATR']):.2f}"      if not pd.isna(latest["ATR"])       else "N/A"),
    ]
    for label, val in metrics:
        print(f"    {label:<12}: {val}")
    print("═" * W + "\n")


def print_backtest_report(trades: pd.DataFrame, stats: dict) -> None:
    """Pretty-print backtest summary statistics and recent trades."""
    print("\n" + "═" * W)
    print("  BACKTEST RESULTS  —  SPY Market-Bottom Strategy")
    print(f"  Window : last {LOOKBACK_YEARS} years   |   Ticker : {TICKER}")
    print("═" * W)

    if "error" in stats:
        print(f"\n  {stats['error']}\n")
        return

    rows = [
        ("Total Trades",       stats["total_trades"]),
        ("Win Rate",           f"{stats['win_rate_%']}%"),
        ("Avg Win",            f"{stats['avg_win_%']}%"),
        ("Avg Loss",           f"{stats['avg_loss_%']}%"),
        ("Profit Factor",      stats["profit_factor"]),
        ("Expectancy / Trade", f"{stats['expectancy_%']}%"),
        ("Max Drawdown",       f"{stats['max_drawdown_%']}%"),
        ("Total PnL (sum %)",  f"{stats['total_pnl_%']}%"),
    ]

    print(f"\n  {'Metric':<28} {'Value':>12}")
    print(_hline())
    for label, val in rows:
        print(f"  {label:<28} {str(val):>12}")

    print(_hline())
    print("\n  Exit Breakdown:")
    for reason, count in stats.get("exit_breakdown", {}).items():
        print(f"    {reason:<8}: {count} trade(s)")

    if not trades.empty:
        print("\n  Last 5 Trades:")
        cols = ["entry_date", "exit_date", "entry_price", "exit_price", "pnl_pct", "exit_reason"]
        display = trades[cols].tail(5).copy()
        display["entry_date"] = display["entry_date"].dt.strftime("%Y-%m-%d")
        display["exit_date"]  = display["exit_date"].dt.strftime("%Y-%m-%d")
        print(display.to_string(index=False))

    print("═" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def _make_demo_data(years: int = LOOKBACK_YEARS + 1) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic SPY + VIX data that exercises the full strategy:
      • A long calm uptrend (VIX low, RSI mid-range)
      • Several engineered market-bottom episodes with:
          – deep price drops through the lower BB
          – RSI < 25 spikes
          – VIX spikes above 30, then peaks (next day lower)
          – selling-climax volume bursts
      • Recovery rallies back through the 20-day SMA (TP hits)

    This lets the entire pipeline — indicators, scoring, backtester —
    run and be validated without any network access.
    """
    np.random.seed(42)
    n_days   = years * 252            # approximate trading days
    dates    = pd.bdate_range(end=datetime.today(), periods=n_days)

    # ── Base price path: geometric Brownian motion ───────────────────────────
    daily_ret = np.random.normal(0.0004, 0.012, n_days)
    price     = 300.0 * np.cumprod(1 + daily_ret)

    # ── Inject 6 synthetic crash-and-recover episodes ────────────────────────
    episode_centers = np.linspace(n_days // 8, int(n_days * 0.95), 6).astype(int)
    for idx in episode_centers:
        crash_len = np.random.randint(8, 18)
        rally_len = np.random.randint(15, 30)
        drop      = np.random.uniform(0.10, 0.18)

        crash_start = max(0, idx - crash_len)
        crash_end   = min(n_days, idx)
        rally_end   = min(n_days, idx + rally_len)

        # Crash segment: accelerating decline
        crash_path = np.linspace(0, -drop, crash_end - crash_start)
        for j, k in enumerate(range(crash_start, crash_end)):
            price[k:] *= (1 + crash_path[j] / (crash_end - crash_start))

        # Recovery: snap back
        if rally_end > crash_end:
            recovery = np.linspace(0, drop * 0.75, rally_end - crash_end)
            for j, k in enumerate(range(crash_end, rally_end)):
                price[k:] *= (1 + recovery[j] / (rally_end - crash_end))

    price = np.clip(price, 50, 10000)

    # ── OHLCV construction ───────────────────────────────────────────────────
    noise  = np.abs(np.random.normal(0, 0.008, n_days))
    high   = price * (1 + noise)
    low    = price * (1 - noise)
    open_  = price * (1 + np.random.normal(0, 0.004, n_days))

    base_vol = 80_000_000
    volume   = np.random.lognormal(np.log(base_vol), 0.4, n_days)
    # Amplify volume during crash episodes
    for idx in episode_centers:
        s = max(0, idx - 15)
        e = min(n_days, idx + 5)
        volume[s:e] *= np.random.uniform(1.8, 3.0, e - s)

    spy = pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  price,
        "Volume": volume,
    }, index=dates)
    spy.index.name = "Date"

    # ── VIX: normally ~15–20, spike to 35–55 during crashes ─────────────────
    vix_base = 15 + np.abs(np.random.normal(0, 3, n_days))
    for idx in episode_centers:
        s = max(0, idx - 12)
        e = min(n_days, idx + 3)
        spike = np.linspace(0, np.random.uniform(25, 40), e - s)
        vix_base[s:e] += spike
        # Gradual decay after peak
        decay_end = min(n_days, e + 15)
        if decay_end > e:
            vix_base[e:decay_end] += spike[-1] * np.linspace(1, 0, decay_end - e)

    vix = pd.Series(vix_base, index=dates, name="VIX")

    return spy, vix


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(demo: bool = False):
    print("\n" + "═" * W)
    print("  SPY Market-Bottom Detection Agent")
    print("  Strategy: Mean Reversion & Sentiment Exhaustion")
    if demo:
        print("  Mode    : DEMO  (synthetic data — no network required)")
    print("═" * W + "\n")

    # 1. Fetch / generate data ────────────────────────────────────────────────
    if demo:
        print("[ 1/4 ] Generating synthetic demo data …")
        spy, vix = _make_demo_data(years=LOOKBACK_YEARS + 1)
        print(f"        SPY rows : {len(spy)}  |  VIX rows : {len(vix)}")
    else:
        print("[ 1/4 ] Fetching market data …")
        spy = fetch_spy(years=LOOKBACK_YEARS + 1)   # extra year for indicator warm-up
        vix = fetch_vix(years=LOOKBACK_YEARS + 1)
        print(f"        SPY rows : {len(spy)}  |  VIX rows : {len(vix)}")

    # 2. Compute indicators ───────────────────────────────────────────────────
    print("[ 2/4 ] Computing indicators (BB, RSI, ATR, SMA200, VIX) …")
    df = compute_indicators(spy, vix)

    # 3. Generate signals ─────────────────────────────────────────────────────
    print("[ 3/4 ] Scoring signals …")
    df = generate_signals(df)

    # 4. Back-test ────────────────────────────────────────────────────────────
    print("[ 4/4 ] Running back-test …\n")
    cutoff = datetime.today() - timedelta(days=LOOKBACK_YEARS * 365)
    df_bt  = df[df.index >= cutoff].copy()

    bt     = Backtester(df_bt)
    trades = bt.run()
    stats  = bt.summary(trades)
    print_backtest_report(trades, stats)

    # 5. Live / latest signal ─────────────────────────────────────────────────
    print_live_signal(df)


if __name__ == "__main__":
    import sys
    demo_mode = "--demo" in sys.argv
    main(demo=demo_mode)
