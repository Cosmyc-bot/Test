"""
SPY Market-Bottom Agent — HTTP API
===================================
Exposes a single endpoint:

  GET /signal          → full signal report as JSON
  GET /signal?demo=1   → same, but uses synthetic data (no network needed)
  GET /health          → liveness probe

Run directly:
  python api.py

Or via gunicorn (as the Dockerfile does):
  gunicorn api:app --bind 0.0.0.0:8080 --timeout 120
"""

import os
import traceback
from datetime import datetime

from flask import Flask, jsonify, request

from trading_agent import (
    LOOKBACK_YEARS,
    MIN_SCORE,
    ATR_MULT,
    compute_indicators,
    fetch_spy,
    fetch_vix,
    generate_signals,
    Backtester,
    _make_demo_data,
)

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _build_signal_response(df) -> dict:
    """
    Extract the latest bar from a fully enriched DataFrame and
    return a structured JSON-serialisable dict.
    """
    import pandas as pd

    latest = df.iloc[-1]

    def _f(val, decimals=2):
        """Safe float conversion; None for NaN."""
        try:
            v = float(val)
            return None if pd.isna(v) else round(v, decimals)
        except (TypeError, ValueError):
            return None

    score  = int(latest.get("score", 0))
    signal = str(latest.get("signal", "WAIT"))

    entry  = _f(latest["Close"])
    atr    = _f(latest["ATR"])
    tp     = _f(latest["BB_middle"])
    sl     = round(entry - ATR_MULT * atr, 2) if entry and atr else None
    rr     = round((tp - entry) / (entry - sl), 2) if tp and sl and (entry - sl) > 0 else None

    conditions = {
        "bb_touch":       bool(latest.get("bb_touch",       False)),
        "rsi_oversold":   bool(latest.get("rsi_oversold",   False)),
        "rsi_divergence": bool(latest.get("rsi_divergence", False)),
        "vix_elevated":   bool(latest.get("vix_elevated",   False)),
        "vix_peak":       bool(latest.get("vix_peak",       False)),
        "selling_climax": bool(latest.get("selling_climax", False)),
        "rubber_band":    bool(latest.get("rubber_band",    False)),
    }

    risk_management = None
    if signal == "BUY" and entry and sl and tp:
        risk_management = {
            "entry":       entry,
            "stop_loss":   sl,
            "take_profit": tp,
            "risk_reward": rr,
            "stop_method": f"entry − {ATR_MULT}× ATR",
            "tp_method":   "20-day SMA (mean reversion)",
        }

    return {
        "ticker":       "SPY",
        "date":         latest.name.strftime("%Y-%m-%d"),
        "signal":       signal,
        "score":        score,
        "max_score":    len(conditions),
        "min_to_buy":   MIN_SCORE,
        "conditions":   conditions,
        "metrics": {
            "close":    entry,
            "rsi_14":   _f(latest["RSI"], 1),
            "vix":      _f(latest["VIX"], 1),
            "bb_lower": _f(latest["BB_lower"]),
            "sma_200":  _f(latest["SMA200"]),
            "atr_14":   _f(latest["ATR"]),
        },
        "risk_management": risk_management,
        "generated_at":    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "spy-market-bottom-agent"})


@app.get("/signal")
def signal():
    """
    Return the current BUY / WAIT signal for SPY.

    Query params:
      demo=1        Use synthetic data (no network call to Yahoo Finance)
      backtest=1    Include backtest summary in the response
    """
    demo     = request.args.get("demo",     "0") == "1"
    do_bt    = request.args.get("backtest", "0") == "1"

    try:
        # ── 1. Data ───────────────────────────────────────────────────────────
        if demo:
            spy, vix = _make_demo_data(years=LOOKBACK_YEARS + 1)
        else:
            spy = fetch_spy(years=LOOKBACK_YEARS + 1)
            vix = fetch_vix(years=LOOKBACK_YEARS + 1)

        # ── 2. Indicators + signals ───────────────────────────────────────────
        df = compute_indicators(spy, vix)
        df = generate_signals(df)

        # ── 3. Build response ────────────────────────────────────────────────
        payload = _build_signal_response(df)

        # ── 4. Optional backtest block ───────────────────────────────────────
        if do_bt:
            from datetime import timedelta
            cutoff = datetime.today() - timedelta(days=LOOKBACK_YEARS * 365)
            df_bt  = df[df.index >= cutoff].copy()
            bt     = Backtester(df_bt)
            trades = bt.run()
            stats  = bt.summary(trades)
            payload["backtest"] = stats

        payload["demo"] = demo
        return jsonify(payload), 200

    except Exception:
        return jsonify({
            "error":   "internal_error",
            "detail":  traceback.format_exc(),
        }), 500


# ─────────────────────────────────────────────────────────────────────────────
# DEV SERVER
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
