#!/usr/bin/env python3
"""
Market Insight — $7,000 Gemini-Enhanced Self-Learning Stock Model
=================================================================
Combines:
  1. Real-time market & mutual fund benchmark data from Yahoo Finance
  2. Live Yahoo Finance news headlines for top candidates & holdings
  3. Quantitative Self-Learning Factor Weights (Trend, Dip RSI, Rel Strength, Volume)
  4. Gemini API (`google-genai` SDK with `gemini-3.8-flash`) Portfolio Manager
     that reads the math scores + live news + `model_memory.json` lessons learned
     to approve/veto trades and record new lessons every day.
"""

from __future__ import annotations
import argparse
import concurrent.futures
import datetime as dt
import json
import os
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict

WORKSPACE_DIR = Path(__file__).resolve().parent
STATE_FILE = WORKSPACE_DIR / "portfolio_state.json"
JS_DATA_FILE = WORKSPACE_DIR / "portfolio_data.js"
MEMORY_FILE = WORKSPACE_DIR / "model_memory.json"
ENV_FILE = WORKSPACE_DIR / ".env"

STARTING_CAPITAL = 7000.00
HYSA_APY = 0.045  # 4.50% APY High-Yield Savings Account benchmark

MARKET_INDICES = [
    {"symbol": "GC=F", "label": "GOLD", "fallback": "GLD"},
    {"symbol": "^DJI", "label": "DOW 30", "fallback": "DIA"},
    {"symbol": "^GSPC", "label": "S&P 500", "fallback": "SPY"},
    {"symbol": "^IXIC", "label": "NASDAQ", "fallback": "QQQ"},
    {"symbol": "^RUT", "label": "RUSSELL 2K", "fallback": "IWM"},
    {"symbol": "^VIX", "label": "VIX VOL", "fallback": "VIXY"},
]

BENCHMARK_EXTRAS = ["AGG", "SCHD"]

MODEL_UNIVERSE = [
    # Core Anchor ETFs & Mega-Cap Compounders
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "sector": "Index Anchor", "bucket": "Core", "max_weight": 0.22},
    {"ticker": "QQQ", "name": "Invesco QQQ Trust", "sector": "Tech Index", "bucket": "Core", "max_weight": 0.18},
    {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Cloud & AI", "bucket": "Core", "max_weight": 0.14},
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Consumer Tech", "bucket": "Core", "max_weight": 0.12},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Internet & AI", "bucket": "Core", "max_weight": 0.12},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Cloud & Retail", "bucket": "Core", "max_weight": 0.12},
    # Tactical Swing & Momentum Leaders
    {"ticker": "NVDA", "name": "NVIDIA Corp.", "sector": "Semiconductors", "bucket": "Swing", "max_weight": 0.14},
    {"ticker": "META", "name": "Meta Platforms", "sector": "Digital Ads & AI", "bucket": "Swing", "max_weight": 0.11},
    {"ticker": "AVGO", "name": "Broadcom Inc.", "sector": "Semiconductors", "bucket": "Swing", "max_weight": 0.10},
    {"ticker": "TSM", "name": "Taiwan Semiconductor", "sector": "Foundry", "bucket": "Swing", "max_weight": 0.10},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Semiconductors", "bucket": "Swing", "max_weight": 0.09},
    {"ticker": "NFLX", "name": "Netflix Inc.", "sector": "Streaming Media", "bucket": "Swing", "max_weight": 0.09},
    {"ticker": "PLTR", "name": "Palantir Technologies", "sector": "Enterprise AI", "bucket": "Swing", "max_weight": 0.08},
    # Quality Defensive, Dividend Value & Stabilizers
    {"ticker": "SCHD", "name": "Schwab US Dividend Equity", "sector": "Dividend Value Fund", "bucket": "Defensive", "max_weight": 0.16},
    {"ticker": "BRK-B", "name": "Berkshire Hathaway", "sector": "Financials", "bucket": "Defensive", "max_weight": 0.12},
    {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Banking", "bucket": "Defensive", "max_weight": 0.10},
    {"ticker": "COST", "name": "Costco Wholesale", "sector": "Consumer Staples", "bucket": "Defensive", "max_weight": 0.10},
    {"ticker": "LLY", "name": "Eli Lilly & Co.", "sector": "Healthcare", "bucket": "Defensive", "max_weight": 0.09},
    {"ticker": "GLD", "name": "SPDR Gold Shares", "sector": "Precious Metals", "bucket": "Defensive", "max_weight": 0.08},
]


def load_env_key() -> Optional[str]:
    """Load GEMINI_API_KEY from environment or local .env file."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"].strip()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    os.environ["GEMINI_API_KEY"] = val
                    return val
    return None


def load_model_memory() -> Dict[str, Any]:
    """Load persistent lessons learned & decision memory from model_memory.json."""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    default_mem = {
        "model_version": "2.0-gemini-hybrid",
        "lessons_learned": [
            {
                "date": "2026-10-26",
                "lesson": "When broad tech indices consolidate, pure RSI dip-buys in high-beta tech lag behind Dividend Value (SCHD) and 60/40 Balanced funds; prioritize relative strength leaders above both 20d and 50d MAs.",
            }
        ],
        "veto_history": [],
    }
    MEMORY_FILE.write_text(json.dumps(default_mem, indent=2), encoding="utf-8")
    return default_mem


def save_model_memory(mem: Dict[str, Any]) -> None:
    MEMORY_FILE.write_text(json.dumps(mem, indent=2), encoding="utf-8")


def fetch_ticker_news(tickers: List[str], max_per_ticker: int = 2) -> Dict[str, List[Dict[str, str]]]:
    """Fetch live financial news headlines from Yahoo Finance for candidate tickers."""
    news_map: Dict[str, List[Dict[str, str]]] = {}
    for t in tickers[:6]:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(t)}&quotesCount=0&newsCount={max_per_ticker}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = []
            for n in data.get("news", [])[:max_per_ticker]:
                items.append(
                    {
                        "title": n.get("title", ""),
                        "publisher": n.get("publisher", "Market News"),
                    }
                )
            if items:
                news_map[t] = items
        except Exception:
            continue
    return news_map


def call_gemini_advisor(
    day_date: str,
    portfolio_summary: Dict[str, Any],
    alternatives: List[Dict[str, Any]],
    top_candidates: List[Dict[str, Any]],
    todays_actions: List[Dict[str, Any]],
    news_map: Dict[str, List[Dict[str, str]]],
    memory: Dict[str, Any],
    adaptive_weights: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Call the Gemini API (`gemini-3.8-flash` via official `google-genai` SDK `models.generate_content`)
    with structured schema output to review candidate trades, apply risk vetos, and generate
    meaningful lessons learned.
    """
    api_key = load_env_key()
    headlines_flat = []
    for t, items in news_map.items():
        for it in items:
            headlines_flat.append(f"[{t}] {it['title']} ({it['publisher']})")

    top_pick = todays_actions[0] if todays_actions else (top_candidates[0] if top_candidates else None)
    top_ticker = top_pick["ticker"] if top_pick else "SPY"
    top_news = news_map.get(top_ticker, [])
    top_headline_str = (
        f"Latest {top_ticker} catalyst: \"{top_news[0]['title']}\" ({top_news[0]['publisher']})."
        if top_news
        else f"{top_ticker} shows institutional relative strength with clean technical support."
    )

    cooldowns = adaptive_weights.get("cooldown_tickers", [])
    veto_str = (
        f"Vetoed dip-adds on {', '.join(cooldowns)} due to negative 5-day momentum and below-50d MA trend risk."
        if cooldowns
        else "Avoided overbought entries with RSI > 68 to protect capital."
    )

    leader_opt = alternatives[0] if alternatives else {"name": "Dividend Value Fund", "gain_pct": 0.0}
    default_lesson = (
        f"On {day_date}, {leader_opt['name']} led the leaderboard ({leader_opt['gain_pct']:+.2f}%); "
        f"combining relative strength ({adaptive_weights.get('rel_strength_weight', 1.0):.2f}x weight) with risk discipline "
        f"keeps portfolio win rate at {portfolio_summary.get('win_rate', 50.0)}%."
    )

    fallback_result = {
        "engine": "Gemini Hybrid Catalyst & Quant Engine",
        "catalyst_headline": top_headline_str,
        "veto_note": veto_str,
        "veto_trade": False,
        "vetoed_ticker": "NONE",
        "lesson_learned": default_lesson,
        "headlines_scanned": headlines_flat[:6],
        "ai_briefing": (
            f"On {day_date}, the $7,000 AI Model closed at ${portfolio_summary['total_value']:,.2f} "
            f"({portfolio_summary['total_gain_dollars']:+,.2f} / {portfolio_summary['total_gain_pct']:+.2f}%), "
            f"with Trade Win Rate at {portfolio_summary.get('win_rate', 50.0)}%. "
            f"Compared to passive alternatives: S&P 500 Mutual Fund is at ${portfolio_summary.get('spy_benchmark_value', 7000.0):,.2f}, "
            f"60/40 Balanced Fund is at ${portfolio_summary.get('balanced_60_40_value', 7000.0):,.2f}, and 4.5% High-Yield Savings is at ${portfolio_summary.get('hysa_value', 7000.0):,.2f}. "
            f"Catalyst Check: {top_headline_str} {veto_str}"
        ),
    }

    if not api_key:
        return fallback_result

    def _invoke_sdk() -> Optional[Dict[str, Any]]:
        from google import genai
        from google.genai import types

        class AdvisorSchema(TypedDict):
            catalyst_headline: str
            veto_note: str
            veto_trade: bool
            vetoed_ticker: str
            lesson_learned: str
            ai_briefing: str

        client = genai.Client(api_key=api_key)
        prompt = (
            f"You are the Lead Portfolio Risk Manager for a $7,000 algorithmic stock simulation on {day_date}.\n"
            f"Portfolio Value: ${portfolio_summary['total_value']:,.2f} (Win rate: {portfolio_summary.get('win_rate', 50.0)}%).\n"
            f"S&P 500 Mutual Fund: ${portfolio_summary.get('spy_benchmark_value', 7000.0):,.2f} | 60/40 Balanced: ${portfolio_summary.get('balanced_60_40_value', 7000.0):,.2f}.\n"
            f"Today's Recommended Actions: {json.dumps(todays_actions[:4])}\n"
            f"Top Ranked Candidates: {json.dumps([c.get('ticker') for c in top_candidates[:4]])}\n"
            f"Recent Headlines: {json.dumps(headlines_flat[:6])}\n"
            f"Prior Lessons: {json.dumps(memory.get('lessons_learned', [])[-3:])}\n"
            "Analyze current macro risk, evaluate whether to approve or veto any proposed trade, and generate a new key lesson learned. Return structured JSON."
        )

        resp = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AdvisorSchema,
            ),
        )
        parsed = json.loads(resp.text)
        parsed["engine"] = "Gemini 3.8 Flash Live API"
        parsed["headlines_scanned"] = headlines_flat[:6]
        return parsed

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(_invoke_sdk)
        res = fut.result(timeout=45.0)
        ex.shutdown(wait=False)
        if res:
            return res
    except Exception:
        ex.shutdown(wait=False, cancel_futures=True)

    return fallback_result


def fetch_yahoo_chart(symbol: str, range_str: str = "6mo", interval: str = "1d") -> Optional[Dict[str, Any]]:
    """Fetch daily OHLCV history from Yahoo Finance v8 chart API using stdlib urllib."""
    encoded = urllib.parse.quote(symbol)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?interval={interval}&range={range_str}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        r0 = result[0]
        timestamps = r0.get("timestamp", [])
        quote = r0.get("indicators", {}).get("quote", [{}])[0]
        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        bars = []
        for i, ts in enumerate(timestamps):
            c = closes[i] if i < len(closes) else None
            if c is None:
                continue
            dt_str = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%d")
            bars.append(
                {
                    "date": dt_str,
                    "open": round(float(opens[i] if i < len(opens) and opens[i] is not None else c), 2),
                    "high": round(float(highs[i] if i < len(highs) and highs[i] is not None else c), 2),
                    "low": round(float(lows[i] if i < len(lows) and lows[i] is not None else c), 2),
                    "close": round(float(c), 2),
                    "volume": int(volumes[i] if i < len(volumes) and volumes[i] is not None else 0),
                }
            )
        dedup: Dict[str, Dict[str, Any]] = {}
        for b in bars:
            dedup[b["date"]] = b

        # If Yahoo hasn't finalized the 1d array for today yet, synthesize it from real-time meta
        meta = r0.get("meta", {})
        reg_time = meta.get("regularMarketTime")
        reg_price = meta.get("regularMarketPrice")
        if reg_time and reg_price is not None:
            reg_dt_str = dt.datetime.fromtimestamp(reg_time, tz=dt.timezone.utc).strftime("%Y-%m-%d")
            if reg_dt_str not in dedup:
                day_open = meta.get("regularMarketDayOpen") or reg_price
                day_high = meta.get("regularMarketDayHigh") or reg_price
                day_low = meta.get("regularMarketDayLow") or reg_price
                day_vol = meta.get("regularMarketVolume") or 0
                dedup[reg_dt_str] = {
                    "date": reg_dt_str,
                    "open": round(float(day_open), 2),
                    "high": round(float(day_high), 2),
                    "low": round(float(day_low), 2),
                    "close": round(float(reg_price), 2),
                    "volume": int(day_vol),
                }

        sorted_bars = [dedup[k] for k in sorted(dedup.keys())]
        return {"symbol": symbol, "meta": meta, "bars": sorted_bars}
    except Exception as exc:
        print(f"[warn] Could not fetch {symbol}: {exc}", file=sys.stderr)
        return None


def compute_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = deltas[-period:]
    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 85.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 1)


def compute_sma(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    subset = values[-window:] if len(values) >= window else values
    return sum(subset) / len(subset)


def compute_adaptive_weights(
    day_date: str,
    recommendations_log: List[Dict[str, Any]],
    current_prices: Dict[str, float],
) -> Dict[str, Any]:
    """
    Self-Improving Weight Engine:
    Evaluates all prior trades made before `day_date` against current real-world
    prices and dynamically adjusts the 4 factor weights + BUY threshold + ticker
    cooldowns based on what is actually winning or losing in the current market.
    """
    prior_recs = [r for r in recommendations_log if r["date"] < day_date and r["action"] == "BUY"]
    if not prior_recs:
        return {
            "trend_weight": 1.00,
            "dip_rsi_weight": 1.00,
            "rel_strength_weight": 1.00,
            "breakout_vol_weight": 1.00,
            "buy_threshold": 70.0,
            "cooldown_tickers": [],
            "learning_note": "Day 1 Baseline: All 4 quantitative factor weights initialized at 1.00x (Buy threshold: 70.0/100).",
        }

    setup_returns: Dict[str, List[float]] = {"trend": [], "dip": [], "breakout": []}
    cooldown_tickers = []
    wins = 0

    for r in prior_recs:
        t = r["ticker"]
        now_p = current_prices.get(t, r["rec_price"])
        ret_pct = ((now_p - r["rec_price"]) / r["rec_price"]) * 100.0 if r["rec_price"] else 0.0
        if ret_pct >= 0:
            wins += 1
        if ret_pct <= -2.2 and t not in ("SPY", "QQQ") and t not in cooldown_tickers:
            cooldown_tickers.append(t)

        tag = r.get("setup_tag", "")
        if "Dip" in tag or "Pullback" in tag or "Oversold" in tag:
            setup_returns["dip"].append(ret_pct)
        elif "Breakout" in tag:
            setup_returns["breakout"].append(ret_pct)
        else:
            setup_returns["trend"].append(ret_pct)

    win_rate = (wins / len(prior_recs)) * 100.0

    def adjust_mult(rets: List[float]) -> float:
        if not rets:
            return 1.00
        avg_r = statistics.mean(rets)
        mult = 1.00 + (avg_r * 0.08)
        return round(max(0.78, min(1.28, mult)), 2)

    trend_w = adjust_mult(setup_returns["trend"])
    dip_w = adjust_mult(setup_returns["dip"])
    breakout_w = adjust_mult(setup_returns["breakout"])
    rel_w = round(1.12 if win_rate < 55.0 else (1.05 if win_rate < 65.0 else 1.00), 2)

    if win_rate < 50.0:
        buy_thresh = 72.5
    elif win_rate < 60.0:
        buy_thresh = 71.0
    else:
        buy_thresh = 69.5

    notes = []
    if dip_w > 1.02:
        notes.append(f"boosted Dip/Pullback weight to {dip_w:.2f}x (winning setup)")
    elif dip_w < 0.98:
        notes.append(f"trimmed Dip weight to {dip_w:.2f}x")

    if trend_w > 1.02:
        notes.append(f"boosted Trend weight to {trend_w:.2f}x")
    elif trend_w < 0.98:
        notes.append(f"reduced Trend weight to {trend_w:.2f}x")

    if rel_w > 1.00:
        notes.append(f"raised Relative Strength weight to {rel_w:.2f}x to favor S&P 500 outperformers")

    if cooldown_tickers:
        notes.append(f"applied cooldown penalty (-7 pts) to lagging pick(s): {', '.join(cooldown_tickers)}")

    summary_str = (
        f"Self-Learning Update (Historical Win Rate {win_rate:.0f}% across {len(prior_recs)} calls): "
        + ("; ".join(notes) if notes else "weights stable near 1.00x")
        + f". Active BUY threshold set to {buy_thresh}/100."
    )

    return {
        "trend_weight": trend_w,
        "dip_rsi_weight": dip_w,
        "rel_strength_weight": rel_w,
        "breakout_vol_weight": breakout_w,
        "buy_threshold": buy_thresh,
        "cooldown_tickers": cooldown_tickers,
        "learning_note": summary_str,
    }


def score_stock_as_of(
    meta_info: Dict[str, Any],
    bars: List[Dict[str, Any]],
    spy_bars: List[Dict[str, Any]],
    adaptive_weights: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    w_trend = adaptive_weights["trend_weight"] if adaptive_weights else 1.0
    w_dip = adaptive_weights["dip_rsi_weight"] if adaptive_weights else 1.0
    w_rel = adaptive_weights["rel_strength_weight"] if adaptive_weights else 1.0
    w_brk = adaptive_weights["breakout_vol_weight"] if adaptive_weights else 1.0
    buy_thresh = adaptive_weights["buy_threshold"] if adaptive_weights else 70.0
    cooldowns = adaptive_weights["cooldown_tickers"] if adaptive_weights else []

    closes = [b["close"] for b in bars]
    volumes = [b["volume"] for b in bars]
    price = closes[-1]
    prev_close = closes[-2] if len(closes) >= 2 else price
    day_pct = ((price - prev_close) / prev_close * 100.0) if prev_close else 0.0

    sma20 = compute_sma(closes, 20)
    sma50 = compute_sma(closes, 50)
    rsi14 = compute_rsi(closes, 14)

    ret_5d = ((price - closes[-6]) / closes[-6] * 100.0) if len(closes) >= 6 else day_pct
    ret_20d = ((price - closes[-21]) / closes[-21] * 100.0) if len(closes) >= 21 else ret_5d

    spy_closes = [b["close"] for b in spy_bars]
    spy_ret_20d = (
        ((spy_closes[-1] - spy_closes[-21]) / spy_closes[-21] * 100.0)
        if len(spy_closes) >= 21
        else 0.0
    )
    rel_strength_20d = round(ret_20d - spy_ret_20d, 2)

    avg_vol_20 = compute_sma([float(v) for v in volumes[:-1]], 20) if len(volumes) > 2 else 1.0
    vol_ratio = round(volumes[-1] / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0

    recent_bars = bars[-12:]
    max_v = max([b["volume"] for b in recent_bars] or [1]) or 1
    spark_bars = []
    for idx, b in enumerate(recent_bars):
        prev_c = recent_bars[idx - 1]["close"] if idx > 0 else b["open"]
        spark_bars.append(
            {
                "h": max(15, int(round((b["volume"] / max_v) * 100))),
                "up": b["close"] >= prev_c,
                "close": b["close"],
            }
        )

    score = 50.0
    reasons = []
    setup_tag = "Trend Hold"

    if price > sma20 and sma20 >= sma50:
        score += 16.0 * w_trend
        reasons.append(f"Confirmed uptrend above 20d (${sma20:.2f}) & 50d (${sma50:.2f}) MAs")
    elif price > sma50 and price < sma20:
        score += 12.0 * w_dip
        setup_tag = "Pullback to Support"
        reasons.append(f"Constructive dip toward 50d support (${sma50:.2f})")
    elif price < sma50:
        score -= 12.0
        setup_tag = "Below 50d Trend"
        reasons.append(f"Trading below 50d moving average (${sma50:.2f})")

    if 33 <= rsi14 <= 48 and price >= sma50 * 0.97:
        score += 16.0 * w_dip
        setup_tag = "Tactical Dip Buy"
        reasons.append(f"Attractive swing entry on cooldown (RSI {rsi14})")
    elif 48 < rsi14 <= 64:
        score += 10.0 * w_trend
        if setup_tag == "Trend Hold":
            setup_tag = "Steady Momentum"
        reasons.append(f"Healthy momentum zone (RSI {rsi14})")
    elif rsi14 < 33:
        score += 8.0 * w_dip
        setup_tag = "Oversold Bounce Watch"
        reasons.append(f"Deeply oversold (RSI {rsi14})")
    elif rsi14 > 73:
        score -= 10.0
        setup_tag = "Overbought / Trim"
        reasons.append(f"Short-term extended (RSI {rsi14} > 73)")

    if rel_strength_20d > 2.0:
        score += min(14.0, rel_strength_20d * 1.8) * w_rel
        reasons.append(f"Outperforming S&P 500 by +{rel_strength_20d:.1f}% over 20d")
    elif rel_strength_20d < -3.0:
        score -= min(10.0, abs(rel_strength_20d) * 1.2)
        reasons.append(f"Lagging S&P 500 ({rel_strength_20d:.1f}% over 20d)")

    if -4.5 <= ret_5d <= -0.8 and price > sma50:
        score += 6.0 * w_dip
        reasons.append(f"5-day pullback ({ret_5d:+.1f}%) offers favorable reward/risk")
    elif ret_5d > 3.5 and vol_ratio >= 1.1:
        score += 7.0 * w_brk
        setup_tag = "Breakout Accumulation"
        reasons.append(f"Strong 5d breakout ({ret_5d:+.1f}%) on {vol_ratio:.1f}x volume")

    if meta_info["bucket"] == "Core":
        score += 5.0
    elif meta_info["bucket"] == "Defensive" and w_rel > 1.05:
        score += 6.5
        reasons.append("Self-learning defensive rotation boost (+6.5 pts)")

    if meta_info["ticker"] in cooldowns:
        score -= 7.0
        reasons.append("Self-learning cooldown (-7 pts due to recent drawdown)")

    score = round(max(15.0, min(96.0, score)), 1)

    if score >= buy_thresh and rsi14 <= 68 and price >= sma50 * 0.98:
        signal = "BUY"
    elif rsi14 > 73 or score < 46:
        signal = "TRIM"
    elif score >= 56:
        signal = "HOLD"
    else:
        signal = "WATCH"

    return {
        "ticker": meta_info["ticker"],
        "name": meta_info["name"],
        "sector": meta_info["sector"],
        "bucket": meta_info["bucket"],
        "max_weight": meta_info["max_weight"],
        "price": round(price, 2),
        "day_open": bars[-1]["open"],
        "day_high": bars[-1]["high"],
        "day_low": bars[-1]["low"],
        "day_pct": round(day_pct, 2),
        "ret_5d": round(ret_5d, 2),
        "ret_20d": round(ret_20d, 2),
        "rel_strength_20d": rel_strength_20d,
        "rsi14": rsi14,
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "vol_ratio": vol_ratio,
        "score": score,
        "signal": signal,
        "setup_tag": setup_tag,
        "reason": " • ".join(reasons[:2]) if reasons else "Balanced risk/reward profile",
        "target_price": round(price * 1.08, 2),
        "stop_price": round(price * 0.94, 2),
        "sparkline": spark_bars,
    }


def build_initial_benchmarks(day1_prices: Dict[str, float], start_date: str) -> Dict[str, Any]:
    spy_p = day1_prices.get("SPY", 500.0)
    agg_p = day1_prices.get("AGG", 100.0)
    qqq_p = day1_prices.get("QQQ", 400.0)
    schd_p = day1_prices.get("SCHD", 28.0)
    gld_p = day1_prices.get("GLD", 250.0)

    return {
        "start_date": start_date,
        "spy_shares": round(STARTING_CAPITAL / spy_p, 6),
        "qqq_shares": round(STARTING_CAPITAL / qqq_p, 6),
        "schd_shares": round(STARTING_CAPITAL / schd_p, 6),
        "gld_shares": round(STARTING_CAPITAL / gld_p, 6),
        "bal60_spy_shares": round((STARTING_CAPITAL * 0.60) / spy_p, 6),
        "bal40_agg_shares": round((STARTING_CAPITAL * 0.40) / agg_p, 6),
    }


def evaluate_alternative_options(
    day_date: str,
    model_total_value: float,
    benchmarks_init: Dict[str, Any],
    day_prices: Dict[str, float],
    model_v2_total_value: Optional[float] = None,
) -> List[Dict[str, Any]]:
    start_dt = dt.datetime.strptime(benchmarks_init["start_date"], "%Y-%m-%d")
    curr_dt = dt.datetime.strptime(day_date, "%Y-%m-%d")
    elapsed_days = max(0, (curr_dt - start_dt).days)

    spy_p = day_prices.get("SPY", 500.0)
    agg_p = day_prices.get("AGG", 100.0)
    qqq_p = day_prices.get("QQQ", 400.0)
    schd_p = day_prices.get("SCHD", 28.0)
    gld_p = day_prices.get("GLD", 250.0)

    spy_val = round(benchmarks_init["spy_shares"] * spy_p, 2)
    bal_val = round(
        benchmarks_init["bal60_spy_shares"] * spy_p
        + benchmarks_init["bal40_agg_shares"] * agg_p,
        2,
    )
    qqq_val = round(benchmarks_init["qqq_shares"] * qqq_p, 2)
    schd_val = round(benchmarks_init["schd_shares"] * schd_p, 2)
    gld_val = round(benchmarks_init["gld_shares"] * gld_p, 2)
    hysa_val = round(STARTING_CAPITAL * ((1.0 + HYSA_APY / 365.0) ** elapsed_days), 2)

    raw_options = [
        {
            "id": "model",
            "name": "Market Insight Model v1.0 (Baseline)",
            "subtitle": "19 Static Equities • Partial Trims • 97% Invested",
            "category": "Legacy AI Baseline",
            "value": model_total_value,
            "is_model": True,
            "is_model_v1": True,
        },
        {
            "id": "spy_mutual",
            "name": "S&P 500 Index Mutual Fund",
            "subtitle": "VFIAX / SPY (500 Largest US Companies)",
            "category": "Stock Mutual Fund",
            "value": spy_val,
            "is_model": False,
        },
        {
            "id": "balanced_60_40",
            "name": "60/40 Balanced Mutual Fund",
            "subtitle": "VBIAX Style (60% Stocks / 40% US Bonds)",
            "category": "Retirement Mutual Fund",
            "value": bal_val,
            "is_model": False,
        },
        {
            "id": "qqq_growth",
            "name": "Tech Growth Mutual Fund / ETF",
            "subtitle": "QQQ / VITAX (100 Largest Nasdaq Leaders)",
            "category": "Aggressive Growth Fund",
            "value": qqq_val,
            "is_model": False,
        },
        {
            "id": "schd_dividend",
            "name": "Dividend Value Mutual Fund",
            "subtitle": "SCHD / VDADX (High-Dividend Blue Chips)",
            "category": "Income Mutual Fund",
            "value": schd_val,
            "is_model": False,
        },
        {
            "id": "hysa_cash",
            "name": "High-Yield Savings (4.50% APY)",
            "subtitle": "FDIC Insured Savings (~+$0.86/day guaranteed)",
            "category": "Risk-Free Bank Savings",
            "value": hysa_val,
            "is_model": False,
        },
        {
            "id": "gold_fund",
            "name": "Gold Bullion Fund (GLD)",
            "subtitle": "Physical Gold Inflation Hedge",
            "category": "Precious Metals",
            "value": gld_val,
            "is_model": False,
        },
        {
            "id": "uninvested_cash",
            "name": "Standard Checking (Do Nothing)",
            "subtitle": "0.00% Interest Uninvested Cash",
            "category": "Uninvested Baseline",
            "value": STARTING_CAPITAL,
            "is_model": False,
        },
    ]

    if model_v2_total_value is not None:
        raw_options.append(
            {
                "id": "model_v2",
                "name": "Market Insight Model v2.0 (Enhanced)",
                "subtitle": "Stop-Losses (-7%) • Cash Shield • Gemini AI Veto",
                "category": "Active AI Model v2.0",
                "value": model_v2_total_value,
                "is_model": False,
                "is_model_v2": True,
            }
        )

    ref_val = model_v2_total_value if model_v2_total_value is not None else model_total_value
    raw_options.sort(key=lambda x: x["value"], reverse=True)
    for rank_idx, opt in enumerate(raw_options):
        val = opt["value"]
        gain_d = round(val - STARTING_CAPITAL, 2)
        gain_p = round((gain_d / STARTING_CAPITAL) * 100.0, 2)
        diff_vs_model = round(ref_val - val, 2)
        opt["rank"] = rank_idx + 1
        opt["gain_dollars"] = gain_d
        opt["gain_pct"] = gain_p
        opt["model_edge_dollars"] = diff_vs_model
        opt["model_edge_pct"] = round(((ref_val - val) / STARTING_CAPITAL) * 100.0, 2)

    return raw_options


def build_initial_portfolio(
    day_date: str,
    scored_stocks: List[Dict[str, Any]],
    spy_price: float,
) -> Tuple[Dict[str, Dict[str, Any]], float, float, List[Dict[str, Any]]]:
    ranked = sorted(scored_stocks, key=lambda x: x["score"], reverse=True)
    selected: List[Tuple[Dict[str, Any], float]] = []

    by_ticker = {s["ticker"]: s for s in scored_stocks}
    if "SPY" in by_ticker:
        selected.append((by_ticker["SPY"], 0.20))
    if "QQQ" in by_ticker:
        selected.append((by_ticker["QQQ"], 0.14))

    used = {s[0]["ticker"] for s in selected}

    core_candidates = [
        s for s in ranked
        if s["bucket"] == "Core" and s["ticker"] not in used and s["rsi14"] <= 68
    ]
    for c in core_candidates[:2]:
        selected.append((c, 0.12))
        used.add(c["ticker"])

    swing_candidates = [
        s for s in ranked
        if s["bucket"] == "Swing" and s["ticker"] not in used and s["rsi14"] <= 66
    ]
    for s in swing_candidates[:2]:
        selected.append((s, 0.11))
        used.add(s["ticker"])

    def_candidates = [
        s for s in ranked
        if s["bucket"] == "Defensive" and s["ticker"] not in used and s["rsi14"] <= 68
    ]
    if def_candidates:
        selected.append((def_candidates[0], 0.08))
        used.add(def_candidates[0]["ticker"])

    positions: Dict[str, Dict[str, Any]] = {}
    cash = STARTING_CAPITAL
    recommendations_log: List[Dict[str, Any]] = []

    for stock, weight in selected:
        alloc_dollars = round(STARTING_CAPITAL * weight, 2)
        if alloc_dollars > cash:
            alloc_dollars = cash
        shares = round(alloc_dollars / stock["price"], 6)
        cost_basis = round(shares * stock["price"], 2)
        cash = round(cash - cost_basis, 2)

        positions[stock["ticker"]] = {
            "ticker": stock["ticker"],
            "name": stock["name"],
            "sector": stock["sector"],
            "bucket": stock["bucket"],
            "shares": shares,
            "avg_entry_price": stock["price"],
            "cost_basis": cost_basis,
            "entry_date": day_date,
            "entry_reason": stock["reason"],
            "setup_tag": stock["setup_tag"],
        }
        recommendations_log.append(
            {
                "id": f"{day_date}-BUY-{stock['ticker']}",
                "date": day_date,
                "action": "BUY",
                "ticker": stock["ticker"],
                "name": stock["name"],
                "sector": stock["sector"],
                "setup_tag": stock["setup_tag"],
                "rec_price": stock["price"],
                "rec_dollars": cost_basis,
                "shares": shares,
                "score": stock["score"],
                "reason": stock["reason"],
            }
        )

    spy_benchmark_shares = round(STARTING_CAPITAL / spy_price, 6)
    return positions, cash, spy_benchmark_shares, recommendations_log



def build_initial_portfolio_v2(
    day_date: str,
    scored_stocks: List[Dict[str, Any]],
    spy_price: float,
    spy_sma50: float,
) -> Tuple[Dict[str, Dict[str, Any]], float, List[Dict[str, Any]], Dict[str, int]]:
    """
    Initial portfolio allocation for Model v2.0.
    Respects market regime: if SPY is below its 50d SMA, initiates with 40% cash shield
    and allocates 60% across top healthy momentum candidates.
    """
    is_defensive = spy_price < spy_sma50
    alloc_capital = STARTING_CAPITAL * (0.60 if is_defensive else 0.90)
    cash = round(STARTING_CAPITAL - alloc_capital, 2)

    ranked = sorted(scored_stocks, key=lambda x: x["score"], reverse=True)
    candidates = [s for s in ranked if s["score"] >= 72.0 and s["rsi14"] <= 68]
    selected = candidates[:5] if candidates else ranked[:4]
    weight_per_stock = (alloc_capital / len(selected)) if selected else 0.0

    positions: Dict[str, Dict[str, Any]] = {}
    recommendations_log: List[Dict[str, Any]] = []
    last_exit_dates: Dict[str, int] = {}

    for s in selected:
        t = s["ticker"]
        p = s["price"]
        sh = round(weight_per_stock / p, 6)
        cost = round(sh * p, 2)
        positions[t] = {
            "ticker": t,
            "name": s["name"],
            "sector": s["sector"],
            "bucket": s["bucket"],
            "shares": sh,
            "avg_entry_price": p,
            "cost_basis": cost,
            "entry_date": day_date,
            "high_watermark": p,
        }
        rec = {
            "id": f"{day_date}-BUY-{t}",
            "date": day_date,
            "action": "BUY",
            "ticker": t,
            "name": s["name"],
            "sector": s["sector"],
            "setup_tag": s["setup_tag"],
            "rec_price": p,
            "rec_dollars": cost,
            "shares": sh,
            "score": s["score"],
            "reason": f"Initial v2 selection ({s['setup_tag']}) — Score {s['score']:.1f}",
        }
        recommendations_log.append(rec)

    return positions, cash, recommendations_log, last_exit_dates


def apply_daily_tactical_rebalance(
    day_date: str,
    positions: Dict[str, Dict[str, Any]],
    cash: float,
    scored_stocks: List[Dict[str, Any]],
    recommendations_log: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], float, List[Dict[str, Any]]]:
    by_ticker = {s["ticker"]: s for s in scored_stocks}
    ranked = sorted(scored_stocks, key=lambda x: x["score"], reverse=True)
    todays_actions: List[Dict[str, Any]] = []

    for ticker, pos in list(positions.items()):
        if ticker in ("SPY", "QQQ"):
            continue
        st = by_ticker.get(ticker)
        if not st:
            continue
        gain_pct = ((st["price"] - pos["avg_entry_price"]) / pos["avg_entry_price"]) * 100.0
        if st["score"] < 46 or (st["rsi14"] > 76 and gain_pct > 4.0):
            trim_shares = round(pos["shares"] * 0.40, 6)
            proceeds = round(trim_shares * st["price"], 2)
            if proceeds >= 120:
                pos["shares"] = round(pos["shares"] - trim_shares, 6)
                pos["cost_basis"] = round(pos["shares"] * pos["avg_entry_price"], 2)
                cash = round(cash + proceeds, 2)
                rec = {
                    "id": f"{day_date}-TRIM-{ticker}",
                    "date": day_date,
                    "action": "TRIM",
                    "ticker": ticker,
                    "name": st["name"],
                    "sector": st["sector"],
                    "setup_tag": st["setup_tag"],
                    "rec_price": st["price"],
                    "rec_dollars": proceeds,
                    "shares": trim_shares,
                    "score": st["score"],
                    "reason": f"Trimmed 40% (${proceeds:.0f}) at ${st['price']:.2f} ({gain_pct:+.1f}% vs entry) — {st['reason']}",
                }
                todays_actions.append(rec)
                recommendations_log.append(rec)

    total_port_val = cash + sum(
        pos["shares"] * by_ticker.get(t, {"price": pos["avg_entry_price"]})["price"]
        for t, pos in positions.items()
    )

    if cash >= 260.0:
        for candidate in ranked:
            t = candidate["ticker"]
            cur_val = positions[t]["shares"] * candidate["price"] if t in positions else 0.0
            cur_weight = cur_val / total_port_val if total_port_val > 0 else 0.0
            if candidate["signal"] == "BUY" and cur_weight < (candidate["max_weight"] - 0.03):
                buy_dollars = min(round(cash * 0.45, 2), round(total_port_val * 0.08, 2))
                if buy_dollars < 180.0 and cash >= 220.0:
                    buy_dollars = 220.0
                if buy_dollars <= cash and buy_dollars >= 150.0:
                    add_shares = round(buy_dollars / candidate["price"], 6)
                    actual_cost = round(add_shares * candidate["price"], 2)
                    cash = round(cash - actual_cost, 2)
                    if t in positions:
                        old = positions[t]
                        new_shares = round(old["shares"] + add_shares, 6)
                        new_cost = round(old["cost_basis"] + actual_cost, 2)
                        old["shares"] = new_shares
                        old["cost_basis"] = new_cost
                        old["avg_entry_price"] = round(new_cost / new_shares, 2)
                    else:
                        positions[t] = {
                            "ticker": t,
                            "name": candidate["name"],
                            "sector": candidate["sector"],
                            "bucket": candidate["bucket"],
                            "shares": add_shares,
                            "avg_entry_price": candidate["price"],
                            "cost_basis": actual_cost,
                            "entry_date": day_date,
                            "entry_reason": candidate["reason"],
                            "setup_tag": candidate["setup_tag"],
                        }
                    rec = {
                        "id": f"{day_date}-BUY-{t}",
                        "date": day_date,
                        "action": "BUY",
                        "ticker": t,
                        "name": candidate["name"],
                        "sector": candidate["sector"],
                        "setup_tag": candidate["setup_tag"],
                        "rec_price": candidate["price"],
                        "rec_dollars": actual_cost,
                        "shares": add_shares,
                        "score": candidate["score"],
                        "reason": f"Tactical Add (${actual_cost:.0f} @ ${candidate['price']:.2f}): {candidate['reason']}",
                    }
                    todays_actions.append(rec)
                    recommendations_log.append(rec)
                    break

    if not todays_actions and ranked:
        top = ranked[0]
        rec = {
            "id": f"{day_date}-HOLD-{top['ticker']}",
            "date": day_date,
            "action": "HOLD / CONVICTION",
            "ticker": top["ticker"],
            "name": top["name"],
            "sector": top["sector"],
            "setup_tag": top["setup_tag"],
            "rec_price": top["price"],
            "rec_dollars": round(positions[top["ticker"]]["shares"] * top["price"], 2)
            if top["ticker"] in positions
            else 500.0,
            "shares": positions[top["ticker"]]["shares"]
            if top["ticker"] in positions
            else round(500.0 / top["price"], 4),
            "score": top["score"],
            "reason": f"Maintain disciplined allocation; #1 model score ({top['score']}/100): {top['reason']}",
        }
        todays_actions.append(rec)
        recommendations_log.append(rec)

    return positions, cash, todays_actions



def apply_daily_tactical_rebalance_v2(
    day_date: str,
    day_index: int,
    positions: Dict[str, Dict[str, Any]],
    cash: float,
    scored_stocks: List[Dict[str, Any]],
    recommendations_log: List[Dict[str, Any]],
    spy_price: float,
    spy_sma50: float,
    last_exit_dates: Dict[str, int],
    veto_ticker: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], float, List[Dict[str, Any]], Dict[str, int]]:
    """
    Model v2.0 Tactical Rebalancing Engine:
    1. Full Stop-Loss (-7.0% or score < 42 with loss): Liquidates 100% of broken positions.
    2. Trailing Stop: Protects gains if pullbacks exceed 5% from a >5% profit peak.
    3. Profit Trim: Trims 33% once if gain >= 8.0% and RSI >= 76.
    4. Anti-Whipsaw Cooldown: Blocks re-buying a ticker within 4 trading days of exit.
    5. Market Regime Cash Shield: Holds minimum 35% cash if SPY < 50d SMA (earning daily interest).
    6. No SPY/QQQ immunity.
    """
    by_ticker = {s["ticker"]: s for s in scored_stocks}
    todays_actions: List[Dict[str, Any]] = []

    # Daily compounding interest on uninvested cash (4.5% APY / 252 days)
    cash = round(cash * (1.0 + (0.045 / 252.0)), 2)

    is_defensive = spy_price < spy_sma50
    min_cash_pct = 0.35 if is_defensive else 0.10

    # Step 1: Manage open positions (Exits, Stop-Losses, Profit Takes)
    for t, pos in list(positions.items()):
        st = by_ticker.get(t)
        if not st:
            continue
        curr_p = st["price"]
        hw = pos.get("high_watermark", pos.get("avg_entry_price", curr_p))
        if curr_p > hw:
            hw = curr_p
        pos["high_watermark"] = hw

        gain_pct = ((curr_p - pos["avg_entry_price"]) / pos["avg_entry_price"]) * 100.0
        drawdown_from_peak = ((curr_p - pos["high_watermark"]) / pos["high_watermark"]) * 100.0

        is_stop_loss = gain_pct <= -7.0
        is_breakdown = st["score"] < 42.0 and gain_pct < -2.0
        is_trailing_stop = (pos["high_watermark"] > pos["avg_entry_price"] * 1.05 and drawdown_from_peak <= -5.0)

        if is_stop_loss or is_breakdown or is_trailing_stop:
            proceeds = round(pos["shares"] * curr_p, 2)
            cash = round(cash + proceeds, 2)
            action_type = "STOP_LOSS" if is_stop_loss else ("TRAILING_STOP" if is_trailing_stop else "EXIT")
            rec = {
                "id": f"{day_date}-{action_type}-{t}",
                "date": day_date,
                "action": action_type,
                "ticker": t,
                "name": pos["name"],
                "sector": pos["sector"],
                "setup_tag": st["setup_tag"],
                "rec_price": curr_p,
                "rec_dollars": proceeds,
                "shares": pos["shares"],
                "score": st["score"],
                "reason": f"Liquidated 100% (${proceeds:.2f}) at ${curr_p:.2f} (PnL: {gain_pct:+.1f}%) — {action_type}",
            }
            todays_actions.append(rec)
            recommendations_log.append(rec)
            last_exit_dates[t] = day_index
            del positions[t]
        elif gain_pct >= 8.0 and st["rsi14"] >= 76:
            trim_sh = round(pos["shares"] * 0.33, 6)
            proceeds = round(trim_sh * curr_p, 2)
            if proceeds >= 50.0:
                pos["shares"] = round(pos["shares"] - trim_sh, 6)
                pos["cost_basis"] = round(pos["shares"] * pos["avg_entry_price"], 2)
                cash = round(cash + proceeds, 2)
                rec = {
                    "id": f"{day_date}-PROFIT_TRIM-{t}",
                    "date": day_date,
                    "action": "PROFIT_TRIM",
                    "ticker": t,
                    "name": pos["name"],
                    "sector": pos["sector"],
                    "setup_tag": st["setup_tag"],
                    "rec_price": curr_p,
                    "rec_dollars": proceeds,
                    "shares": trim_sh,
                    "score": st["score"],
                    "reason": f"Took profit on 33% (${proceeds:.2f}) at ${curr_p:.2f} (+{gain_pct:.1f}% gain)",
                }
                todays_actions.append(rec)
                recommendations_log.append(rec)

    # Step 2: Deploy cash if available above minimum regime reserve
    total_port_val = cash + sum(
        pos["shares"] * by_ticker.get(t, {"price": pos["avg_entry_price"]})["price"]
        for t, pos in positions.items()
    )
    cash_floor = total_port_val * min_cash_pct
    deployable = max(0.0, cash - cash_floor)

    if deployable >= 180.0:
        ranked = sorted(scored_stocks, key=lambda x: x["score"], reverse=True)
        for cand in ranked:
            t = cand["ticker"]
            if veto_ticker and t == veto_ticker:
                continue
            if t in last_exit_dates and (day_index - last_exit_dates[t]) < 4:
                continue
            if cand["signal"] == "BUY" and cand["score"] >= 74.0 and cand["rsi14"] <= 65:
                if t not in positions:
                    buy_amt = min(deployable, 300.0)
                    if buy_amt >= 150.0:
                        p = cand["price"]
                        sh = round(buy_amt / p, 6)
                        actual_cost = round(sh * p, 2)
                        cash = round(cash - actual_cost, 2)
                        deployable = round(deployable - actual_cost, 2)
                        positions[t] = {
                            "ticker": t,
                            "name": cand["name"],
                            "sector": cand["sector"],
                            "bucket": cand["bucket"],
                            "shares": sh,
                            "avg_entry_price": p,
                            "cost_basis": actual_cost,
                            "entry_date": day_date,
                            "high_watermark": p,
                        }
                        rec = {
                            "id": f"{day_date}-BUY-{t}",
                            "date": day_date,
                            "action": "BUY",
                            "ticker": t,
                            "name": cand["name"],
                            "sector": cand["sector"],
                            "setup_tag": cand["setup_tag"],
                            "rec_price": p,
                            "rec_dollars": actual_cost,
                            "shares": sh,
                            "score": cand["score"],
                            "reason": f"Bought entry (${actual_cost:.2f}) at ${p:.2f} — {cand['setup_tag']}",
                        }
                        todays_actions.append(rec)
                        recommendations_log.append(rec)
                        if deployable < 180.0:
                            break

    return positions, cash, todays_actions, last_exit_dates


def build_day_snapshot(
    day_index: int,
    day_date: str,
    tab_label: str,
    is_backtest: bool,
    positions: Dict[str, Dict[str, Any]],
    cash: float,
    spy_benchmark_shares: float,
    spy_price: float,
    benchmarks_init: Dict[str, Any],
    day_prices: Dict[str, float],
    adaptive_weights: Dict[str, Any],
    scored_stocks: List[Dict[str, Any]],
    todays_actions: List[Dict[str, Any]],
    recommendations_log: List[Dict[str, Any]],
    prev_total_value: float,
    positions_v2: Optional[Dict[str, Dict[str, Any]]] = None,
    cash_v2: Optional[float] = None,
    todays_actions_v2: Optional[List[Dict[str, Any]]] = None,
    recommendations_log_v2: Optional[List[Dict[str, Any]]] = None,
    prev_total_value_v2: Optional[float] = None,
    news_map: Optional[Dict[str, List[Dict[str, str]]]] = None,
    memory: Optional[Dict[str, Any]] = None,
    call_gemini_live: bool = False,
    custom_note: Optional[str] = None,
) -> Dict[str, Any]:
    by_ticker = {s["ticker"]: s for s in scored_stocks}

    # Model v1 calculations
    holdings_list = []
    invested_value = 0.0
    day_open_val = cash
    day_high_val = cash
    day_low_val = cash

    for ticker, pos in positions.items():
        st = by_ticker.get(ticker)
        cur_price = st["price"] if st else pos["avg_entry_price"]
        d_open = st["day_open"] if st else cur_price
        d_high = st["day_high"] if st else cur_price
        d_low = st["day_low"] if st else cur_price

        mkt_val = round(pos["shares"] * cur_price, 2)
        invested_value += mkt_val
        day_open_val += pos["shares"] * d_open
        day_high_val += pos["shares"] * d_high
        day_low_val += pos["shares"] * d_low

        pnl_dollars = round(mkt_val - pos["cost_basis"], 2)
        pnl_pct = round((pnl_dollars / pos["cost_basis"] * 100.0), 2) if pos["cost_basis"] > 0 else 0.0

        holdings_list.append(
            {
                "ticker": ticker,
                "name": pos["name"],
                "sector": pos["sector"],
                "bucket": pos["bucket"],
                "shares": round(pos["shares"], 4),
                "avg_entry": pos["avg_entry_price"],
                "current_price": cur_price,
                "day_pct": st["day_pct"] if st else 0.0,
                "cost_basis": pos["cost_basis"],
                "market_value": mkt_val,
                "pnl_dollars": pnl_dollars,
                "pnl_pct": pnl_pct,
                "score": st["score"] if st else 65.0,
                "signal": st["signal"] if st else "HOLD",
                "setup_tag": st["setup_tag"] if st else pos.get("setup_tag", "Core"),
                "reason": st["reason"] if st else pos.get("entry_reason", ""),
                "sparkline": st["sparkline"] if st else [],
            }
        )

    holdings_list.sort(key=lambda x: x["market_value"], reverse=True)
    invested_value = round(invested_value, 2)
    total_value = round(invested_value + cash, 2)

    total_gain_dollars = round(total_value - STARTING_CAPITAL, 2)
    total_gain_pct = round((total_gain_dollars / STARTING_CAPITAL) * 100.0, 2)

    day_change_dollars = round(total_value - prev_total_value, 2)
    day_change_pct = round((day_change_dollars / prev_total_value) * 100.0, 2) if prev_total_value > 0 else 0.0

    spy_benchmark_value = round(spy_benchmark_shares * spy_price, 2)
    spy_gain_dollars = round(spy_benchmark_value - STARTING_CAPITAL, 2)
    spy_gain_pct = round((spy_gain_dollars / STARTING_CAPITAL) * 100.0, 2)
    alpha_dollars = round(total_value - spy_benchmark_value, 2)
    alpha_pct = round(total_gain_pct - spy_gain_pct, 2)

    # Model v2 calculations (if provided)
    holdings_v2 = []
    invested_v2 = 0.0
    c_v2 = cash_v2 if cash_v2 is not None else cash
    prev_v2 = prev_total_value_v2 if prev_total_value_v2 is not None else prev_total_value
    pos_v2_map = positions_v2 if positions_v2 is not None else positions

    for ticker, pos in pos_v2_map.items():
        st = by_ticker.get(ticker)
        cur_price = st["price"] if st else pos["avg_entry_price"]
        mkt_val = round(pos["shares"] * cur_price, 2)
        invested_v2 += mkt_val
        pnl_dollars = round(mkt_val - pos["cost_basis"], 2)
        pnl_pct = round((pnl_dollars / pos["cost_basis"] * 100.0), 2) if pos["cost_basis"] > 0 else 0.0

        holdings_v2.append(
            {
                "ticker": ticker,
                "name": pos["name"],
                "sector": pos["sector"],
                "bucket": pos["bucket"],
                "shares": round(pos["shares"], 4),
                "avg_entry": pos["avg_entry_price"],
                "current_price": cur_price,
                "day_pct": st["day_pct"] if st else 0.0,
                "cost_basis": pos["cost_basis"],
                "market_value": mkt_val,
                "pnl_dollars": pnl_dollars,
                "pnl_pct": pnl_pct,
                "score": st["score"] if st else 65.0,
                "signal": st["signal"] if st else "HOLD",
                "setup_tag": st["setup_tag"] if st else pos.get("setup_tag", "Core"),
                "reason": st["reason"] if st else pos.get("entry_reason", ""),
                "sparkline": st["sparkline"] if st else [],
            }
        )
    holdings_v2.sort(key=lambda x: x["market_value"], reverse=True)
    invested_v2 = round(invested_v2, 2)
    total_value_v2 = round(invested_v2 + c_v2, 2)
    gain_dollars_v2 = round(total_value_v2 - STARTING_CAPITAL, 2)
    gain_pct_v2 = round((gain_dollars_v2 / STARTING_CAPITAL) * 100.0, 2)
    day_chg_d_v2 = round(total_value_v2 - prev_v2, 2)
    day_chg_p_v2 = round((day_chg_d_v2 / prev_v2) * 100.0, 2) if prev_v2 > 0 else 0.0

    # Evaluate 9 alternative options
    alternatives = evaluate_alternative_options(
        day_date=day_date,
        model_total_value=total_value,
        benchmarks_init=benchmarks_init,
        day_prices=day_prices,
        model_v2_total_value=total_value_v2,
    )
    bal_opt = next((o for o in alternatives if o["id"] == "balanced_60_40"), None)
    hysa_opt = next((o for o in alternatives if o["id"] == "hysa_cash"), None)

    # Audit v1 recommendations
    audited_recommendations = []
    wins = 0
    total_rec_evaluated = 0
    for rec in recommendations_log:
        if rec["date"] > day_date:
            continue
        st = by_ticker.get(rec["ticker"])
        curr_p = st["price"] if st else rec["rec_price"]
        chg_pct = round(((curr_p - rec["rec_price"]) / rec["rec_price"] * 100.0), 2) if rec["rec_price"] else 0.0
        is_win = chg_pct >= 0.0
        if rec["action"] == "TRIM":
            is_win = True
        if is_win:
            wins += 1
        total_rec_evaluated += 1
        audited_recommendations.append(
            {
                "id": rec["id"],
                "date": rec["date"],
                "action": rec["action"],
                "ticker": rec["ticker"],
                "name": rec.get("name", rec["ticker"]),
                "rec_price": rec["rec_price"],
                "current_price": curr_p,
                "change_pct": chg_pct,
                "is_win": is_win,
                "rec_dollars": rec.get("rec_dollars", 0.0),
                "setup_tag": rec.get("setup_tag", "Setup"),
                "reason": rec.get("reason", ""),
            }
        )

    win_rate = round((wins / total_rec_evaluated * 100.0), 1) if total_rec_evaluated > 0 else 50.0

    # Audit v2 recommendations
    audited_v2 = []
    wins_v2 = 0
    tot_v2 = 0
    recs_v2_source = recommendations_log_v2 if recommendations_log_v2 is not None else recommendations_log
    for rec in recs_v2_source:
        if rec["date"] > day_date:
            continue
        st = by_ticker.get(rec["ticker"])
        curr_p = st["price"] if st else rec["rec_price"]
        chg_pct = round(((curr_p - rec["rec_price"]) / rec["rec_price"] * 100.0), 2) if rec["rec_price"] else 0.0
        is_win = chg_pct >= 0.0
        if rec["action"] in ("PROFIT_TRIM", "TRIM"):
            is_win = True
        if is_win:
            wins_v2 += 1
        tot_v2 += 1
        audited_v2.append(
            {
                "id": rec["id"],
                "date": rec["date"],
                "action": rec["action"],
                "ticker": rec["ticker"],
                "name": rec.get("name", rec["ticker"]),
                "rec_price": rec["rec_price"],
                "current_price": curr_p,
                "change_pct": chg_pct,
                "is_win": is_win,
                "rec_dollars": rec.get("rec_dollars", 0.0),
                "setup_tag": rec.get("setup_tag", "Setup"),
                "reason": rec.get("reason", ""),
            }
        )
    win_rate_v2 = round((wins_v2 / tot_v2 * 100.0), 1) if tot_v2 > 0 else 50.0

    # Sector allocations
    allocations = []
    for h in holdings_list:
        allocations.append(
            {
                "label": f"{h['ticker']} — {h['name']}",
                "ticker": h["ticker"],
                "value": h["market_value"],
                "pct": round((h["market_value"] / total_value) * 100.0, 2),
                "pnl_pct": h["pnl_pct"],
            }
        )
    if cash > 0:
        allocations.append(
            {
                "label": "CASH — Tactical Dry Powder",
                "ticker": "CASH",
                "value": cash,
                "pct": round((cash / total_value) * 100.0, 2),
                "pnl_pct": 0.0,
            }
        )

    portfolio_obj = {
        "starting_capital": STARTING_CAPITAL,
        "total_value": total_value,
        "invested_value": invested_value,
        "cash_reserve": cash,
        "day_open": round(day_open_val, 2),
        "day_high": round(day_high_val, 2),
        "day_low": round(day_low_val, 2),
        "day_change_dollars": day_change_dollars,
        "day_change_pct": day_change_pct,
        "total_gain_dollars": total_gain_dollars,
        "total_gain_pct": total_gain_pct,
        "uninvested_cash_value": STARTING_CAPITAL,
        "spy_benchmark_value": spy_benchmark_value,
        "spy_gain_dollars": spy_gain_dollars,
        "spy_gain_pct": spy_gain_pct,
        "balanced_60_40_value": bal_opt["value"] if bal_opt else STARTING_CAPITAL,
        "hysa_value": hysa_opt["value"] if hysa_opt else STARTING_CAPITAL,
        "alpha_dollars": alpha_dollars,
        "alpha_pct": alpha_pct,
        "win_rate": win_rate,
    }

    portfolio_obj_v2 = {
        "starting_capital": STARTING_CAPITAL,
        "total_value": total_value_v2,
        "invested_value": invested_v2,
        "cash_reserve": c_v2,
        "day_change_dollars": day_chg_d_v2,
        "day_change_pct": day_chg_p_v2,
        "total_gain_dollars": gain_dollars_v2,
        "total_gain_pct": gain_pct_v2,
        "spy_benchmark_value": spy_benchmark_value,
        "spy_gain_dollars": spy_gain_dollars,
        "spy_gain_pct": spy_gain_pct,
        "balanced_60_40_value": bal_opt["value"] if bal_opt else STARTING_CAPITAL,
        "hysa_value": hysa_opt["value"] if hysa_opt else STARTING_CAPITAL,
        "alpha_dollars": round(total_value_v2 - spy_benchmark_value, 2),
        "alpha_pct": round(gain_pct_v2 - spy_gain_pct, 2),
        "win_rate": win_rate_v2,
    }

    ranked_candidates = sorted(scored_stocks, key=lambda x: x["score"], reverse=True)[:4]
    actions_v2_report = todays_actions_v2 if todays_actions_v2 is not None else todays_actions

    if call_gemini_live:
        gemini_insight = call_gemini_advisor(
            day_date=day_date,
            portfolio_summary=portfolio_obj_v2,
            alternatives=alternatives,
            top_candidates=ranked_candidates,
            todays_actions=actions_v2_report,
            news_map=news_map or {},
            memory=memory or {},
            adaptive_weights=adaptive_weights,
        )
        if memory is not None:
            existing_dates = {l.get("date") for l in memory.get("lessons_learned", [])}
            if day_date not in existing_dates:
                memory.setdefault("lessons_learned", []).append(
                    {"date": day_date, "lesson": gemini_insight["lesson_learned"]}
                )
            if gemini_insight.get("veto_trade") and gemini_insight.get("vetoed_ticker"):
                memory.setdefault("veto_history", []).append(
                    {
                        "date": day_date,
                        "ticker": gemini_insight["vetoed_ticker"],
                        "reason": gemini_insight.get("veto_note", "Risk vetoed"),
                    }
                )
            save_model_memory(memory)
    else:
        top_winner = max(holdings_v2, key=lambda x: x["pnl_dollars"]) if holdings_v2 else None
        action_headlines = [
            f"{a['action']} {a['ticker']} (${a['rec_dollars']:,.0f} @ ${a['rec_price']:,.2f})"
            for a in actions_v2_report
        ]
        bal_val_str = f"${bal_opt['value']:,.2f}" if bal_opt else "$7,000.00"
        hysa_val_str = f"${hysa_opt['value']:,.2f}" if hysa_opt else "$7,000.00"
        brief = (
            f"On {day_date}, Model v2.0 closed at ${total_value_v2:,.2f} "
            f"({gain_dollars_v2:+,.2f} / {gain_pct_v2:+.2f}%) vs. Baseline v1.0 at ${total_value:,.2f} ({total_gain_pct:+.2f}%), "
            f"SPY Mutual Fund at ${spy_benchmark_value:,.2f}, and 4.5% HYSA at ${hysa_val_str}. "
            + (
                f"Top holding: {top_winner['ticker']} ({top_winner['pnl_dollars']:+,.2f}, {top_winner['pnl_pct']:+.2f}%). "
                if top_winner
                else ""
            )
            + f"Today's v2 action: {', '.join(action_headlines) if action_headlines else 'HOLD (Regime Cash Shield Active)'}."
        )
        gemini_insight = {
            "engine": "Gemini Hybrid Catalyst & Quant Engine",
            "catalyst_headline": f"Evaluated 19 assets on {day_date}; Model v2.0 held {round((c_v2/total_value_v2)*100, 1)}% cash defense.",
            "veto_note": (
                f"Cooldown active on {', '.join(adaptive_weights['cooldown_tickers'])}"
                if adaptive_weights.get("cooldown_tickers")
                else "Blocked overbought entries with RSI > 68."
            ),
            "lesson_learned": adaptive_weights["learning_note"],
            "headlines_scanned": [],
            "ai_briefing": brief,
        }

    strength_meter = {
        "bullish_count": len([s for s in scored_stocks if s["signal"] == "BUY"]),
        "hold_count": len([s for s in scored_stocks if s["signal"] == "HOLD"]),
        "bearish_count": len([s for s in scored_stocks if s["signal"] == "TRIM"]),
        "avg_score": round(statistics.mean([s["score"] for s in scored_stocks]), 1) if scored_stocks else 50.0,
        "market_regime": "DEFENSIVE (Cash Shield Active)" if spy_price < adaptive_weights.get("spy_sma50", spy_price) else "CONFIRMED UPTREND",
    }

    return {
        "day_index": day_index,
        "date": day_date,
        "tab_label": tab_label,
        "is_backtest": is_backtest,
        "updated_at": dt.datetime.now().strftime("%Y-%m-%d %I:%M %p ET"),
        "portfolio": portfolio_obj,
        "portfolio_v2": portfolio_obj_v2,
        "alternative_options": alternatives,
        "adaptive_learning": adaptive_weights,
        "strength_meter": strength_meter,
        "gemini_insight": gemini_insight,
        "ai_briefing": gemini_insight.get("ai_briefing", ""),
        "todays_actions": todays_actions,
        "todays_actions_v2": actions_v2_report,
        "holdings": holdings_list,
        "holdings_v2": holdings_v2,
        "audited_recommendations": audited_recommendations,
        "audited_recommendations_v2": audited_v2,
        "watchlist_scores": scored_stocks,
        "allocations": allocations,
    }


def fetch_top_bar_indices() -> List[Dict[str, Any]]:
    ribbon = []
    for item in MARKET_INDICES:
        chart = fetch_yahoo_chart(item["symbol"], range_str="1mo", interval="1d")
        if not chart or len(chart["bars"]) < 2:
            chart = fetch_yahoo_chart(item["fallback"], range_str="1mo", interval="1d")
        if chart and len(chart["bars"]) >= 2:
            bars = chart["bars"]
            last = bars[-1]["close"]
            prev = bars[-2]["close"]
            chg = last - prev
            pct = (chg / prev * 100.0) if prev else 0.0
            spark = [b["close"] for b in bars[-14:]]
            ribbon.append(
                {
                    "label": item["label"],
                    "symbol": item["symbol"],
                    "price": round(last, 2),
                    "change": round(chg, 2),
                    "pct": round(pct, 2),
                    "up": chg >= 0,
                    "spark": spark,
                }
            )
    return ribbon


def get_day_prices_map(raw_histories: Dict[str, List[Dict[str, Any]]], d_str: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sym, bars in raw_histories.items():
        sliced = [b for b in bars if b["date"] <= d_str]
        if sliced:
            out[sym] = sliced[-1]["close"]
    return out


def run_update(init_history: bool = False, custom_note: Optional[str] = None) -> Dict[str, Any]:
    load_env_key()
    memory = load_model_memory()
    print("Fetching live market, mutual fund benchmarks & Yahoo Finance news headlines...")
    ribbon = fetch_top_bar_indices()

    raw_histories: Dict[str, List[Dict[str, Any]]] = {}
    for item in MODEL_UNIVERSE:
        t = item["ticker"]
        res = fetch_yahoo_chart(t, range_str="6mo", interval="1d")
        if res and len(res["bars"]) >= 30:
            raw_histories[t] = res["bars"]

    for extra_sym in BENCHMARK_EXTRAS:
        if extra_sym not in raw_histories:
            res = fetch_yahoo_chart(extra_sym, range_str="6mo", interval="1d")
            if res and len(res["bars"]) >= 10:
                raw_histories[extra_sym] = res["bars"]

    if "SPY" not in raw_histories:
        raise RuntimeError("Could not fetch SPY benchmark data from Yahoo Finance.")

    spy_bars_all = raw_histories["SPY"]
    available_dates = [b["date"] for b in spy_bars_all]

    state: Optional[Dict[str, Any]] = None
    if STATE_FILE.exists() and not init_history:
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if "benchmarks_init" not in state.get("internal_ledger", {}):
                state = None
        except Exception:
            state = None

    # Load existing day dates if present
    existing_day_tabs = state.get("day_tabs", []) if state else []
    sim_dates = [t["date"] for t in existing_day_tabs] if existing_day_tabs else (available_dates[-15:] if len(available_dates) >= 15 else available_dates)

    # Always ensure available_dates includes today if fresh
    last_known = sim_dates[-1] if sim_dates else available_dates[-1]
    for d in available_dates:
        if d > last_known and d not in sim_dates:
            sim_dates.append(d)

    # Simulate Dual Models across sim_dates
    positions_v1: Dict[str, Dict[str, Any]] = {}
    cash_v1 = STARTING_CAPITAL
    recs_v1: List[Dict[str, Any]] = []

    positions_v2: Dict[str, Dict[str, Any]] = {}
    cash_v2 = STARTING_CAPITAL
    recs_v2: List[Dict[str, Any]] = []
    last_exit_dates_v2: Dict[str, int] = {}

    day_tabs: List[Dict[str, Any]] = []
    prev_val_v1 = STARTING_CAPITAL
    prev_val_v2 = STARTING_CAPITAL

    day1_prices = get_day_prices_map(raw_histories, sim_dates[0])
    benchmarks_init = build_initial_benchmarks(day1_prices, sim_dates[0])
    spy_benchmark_shares = benchmarks_init["spy_shares"]

    for idx, d_str in enumerate(sim_dates):
        spy_slice = [b for b in spy_bars_all if b["date"] <= d_str]
        spy_price = spy_slice[-1]["close"]
        spy_sma50 = compute_sma([b["close"] for b in spy_slice], 50)
        day_prices = get_day_prices_map(raw_histories, d_str)

        adaptive_weights = compute_adaptive_weights(d_str, recs_v1, day_prices)
        adaptive_weights["spy_sma50"] = spy_sma50

        scored_today = []
        for item in MODEL_UNIVERSE:
            t = item["ticker"]
            if t not in raw_histories:
                continue
            b_slice = [b for b in raw_histories[t] if b["date"] <= d_str]
            if len(b_slice) >= 15:
                scored_today.append(score_stock_as_of(item, b_slice, spy_slice, adaptive_weights))

        is_last = (idx == len(sim_dates) - 1)
        dt_obj = dt.datetime.strptime(d_str, "%Y-%m-%d")
        short_date = dt_obj.strftime("%b %d")

        FORK_DATE = "2026-11-06"
        is_v2_active = (d_str >= FORK_DATE)

        if idx == 0:
            positions_v1, cash_v1, spy_benchmark_shares, recs_v1 = build_initial_portfolio(
                d_str, scored_today, spy_price
            )
            todays_actions_v1 = list(recs_v1)
            tab_label = f"Sim Start • {short_date}"
            is_bt = True
            todays_actions_v2 = []
        else:
            positions_v1, cash_v1, todays_actions_v1 = apply_daily_tactical_rebalance(
                d_str, positions_v1, cash_v1, scored_today, recs_v1
            )
            if is_last:
                tab_label = f"Today • {short_date} (Live)"
                is_bt = False
            else:
                tab_label = f"Day {idx + 1} • {short_date}"
                is_bt = True

            # Model v2 handling
            if d_str == FORK_DATE:
                # Fork launch at exact current pricepoint of Model v1
                positions_v2 = {k: dict(v) for k, v in positions_v1.items()}
                for p_dict in positions_v2.values():
                    p_dict["high_watermark"] = p_dict.get("high_watermark", p_dict["avg_entry_price"])
                cash_v2 = cash_v1
                fork_val = round(cash_v1 + sum(pos["shares"] * day_prices.get(t, pos["avg_entry_price"]) for t, pos in positions_v1.items()), 2)
                todays_actions_v2 = [{
                    "id": f"{d_str}-FORK-LAUNCH-v2",
                    "date": d_str,
                    "action": "MODEL_FORK",
                    "ticker": "BOOK",
                    "name": "Model v2.0 Live Fork Launch",
                    "sector": "Algorithmic Upgrade",
                    "setup_tag": "Forward Test Start",
                    "rec_price": fork_val,
                    "rec_dollars": fork_val,
                    "shares": 1.0,
                    "score": 100.0,
                    "reason": f"Model v2.0 officially launched from Model v1's exact current pricepoint (${fork_val:,.2f}) with stop-losses and cash shield active for Monday's open."
                }]
                recs_v2 = list(todays_actions_v2)
                last_exit_dates_v2 = {}
            elif d_str > FORK_DATE:
                # Model v2 enhanced rebalancing for future days
                positions_v2, cash_v2, todays_actions_v2, last_exit_dates_v2 = apply_daily_tactical_rebalance_v2(
                    day_date=d_str,
                    day_index=idx,
                    positions=positions_v2,
                    cash=cash_v2,
                    scored_stocks=scored_today,
                    recommendations_log=recs_v2,
                    spy_price=spy_price,
                    spy_sma50=spy_sma50,
                    last_exit_dates=last_exit_dates_v2,
                )
            else:
                todays_actions_v2 = []

        news_map = {}
        if is_last:
            top_tickers = [a["ticker"] for a in todays_actions_v2] + [
                s["ticker"] for s in sorted(scored_today, key=lambda x: x["score"], reverse=True)[:4]
            ]
            news_map = fetch_ticker_news(list(dict.fromkeys(top_tickers)))

        snap = build_day_snapshot(
            day_index=idx + 1,
            day_date=d_str,
            tab_label=tab_label,
            is_backtest=is_bt,
            positions=positions_v1,
            cash=cash_v1,
            spy_benchmark_shares=spy_benchmark_shares,
            spy_price=spy_price,
            benchmarks_init=benchmarks_init,
            day_prices=day_prices,
            adaptive_weights=adaptive_weights,
            scored_stocks=scored_today,
            todays_actions=todays_actions_v1,
            recommendations_log=recs_v1,
            prev_total_value=prev_val_v1,
            positions_v2=positions_v2 if is_v2_active else None,
            cash_v2=cash_v2 if is_v2_active else None,
            todays_actions_v2=todays_actions_v2 if is_v2_active else None,
            recommendations_log_v2=recs_v2 if is_v2_active else None,
            prev_total_value_v2=prev_val_v2 if is_v2_active else None,
            news_map=news_map,
            memory=memory,
            call_gemini_live=is_last,
            custom_note=custom_note if is_last else None,
        )
        prev_val_v1 = snap["portfolio"]["total_value"]
        prev_val_v2 = snap["portfolio_v2"]["total_value"]
        day_tabs.append(snap)

    state = {
        "model_name": "Market Insight Dual Model Simulation (v1 Baseline vs. v2 Enhanced)",
        "starting_capital": STARTING_CAPITAL,
        "created_at": sim_dates[0],
        "last_updated": dt.datetime.now().strftime("%Y-%m-%d %I:%M %p ET"),
        "market_ribbon": ribbon,
        "model_memory": memory,
        "internal_ledger": {
            "positions": positions_v1,
            "cash": cash_v1,
            "spy_benchmark_shares": spy_benchmark_shares,
            "benchmarks_init": benchmarks_init,
            "recommendations_log": recs_v1,
        },
        "internal_ledger_v1": {
            "positions": positions_v1,
            "cash": cash_v1,
            "spy_benchmark_shares": spy_benchmark_shares,
            "benchmarks_init": benchmarks_init,
            "recommendations_log": recs_v1,
        },
        "internal_ledger_v2": {
            "positions": positions_v2,
            "cash": cash_v2,
            "recommendations_log": recs_v2,
            "last_exit_dates": last_exit_dates_v2,
        },
        "day_tabs": day_tabs,
    }

    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    JS_DATA_FILE.write_text(
        f"window.MARKET_INSIGHT_DATA = {json.dumps(state, indent=2)};\n",
        encoding="utf-8",
    )
    sync_to_github_if_configured()
    return state


def sync_to_github_if_configured() -> None:
    """If .github_token exists locally, push updated data files in a single atomic commit to GitHub Pages."""
    token_path = WORKSPACE_DIR / ".github_token"
    if not token_path.exists():
        return
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        return
    try:
        import base64

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "MarketInsightUpdater/1.0",
        }
        # 1. Get HEAD commit of main
        url_ref = "https://api.github.com/repos/PDub1101/market-insight/git/refs/heads/main"
        req_ref = urllib.request.Request(url_ref, headers=headers)
        with urllib.request.urlopen(req_ref, timeout=10) as r:
            commit_sha = json.loads(r.read().decode("utf-8"))["object"]["sha"]

        # 2. Get tree SHA of HEAD commit
        url_commit = f"https://api.github.com/repos/PDub1101/market-insight/git/commits/{commit_sha}"
        req_commit = urllib.request.Request(url_commit, headers=headers)
        with urllib.request.urlopen(req_commit, timeout=10) as r:
            tree_sha = json.loads(r.read().decode("utf-8"))["tree"]["sha"]

        # 3. Build tree with all updated data files at once in a single commit
        tree_items = []
        for fname in ("portfolio_data.js", "portfolio_state.json", "model_memory.json"):
            fpath = WORKSPACE_DIR / fname
            if not fpath.exists():
                continue
            tree_items.append({
                "path": fname,
                "mode": "100644",
                "type": "blob",
                "content": fpath.read_text(encoding="utf-8"),
            })

        if not tree_items:
            return

        url_tree = "https://api.github.com/repos/PDub1101/market-insight/git/trees"
        tree_payload = {"base_tree": tree_sha, "tree": tree_items}
        req_tree = urllib.request.Request(
            url_tree,
            data=json.dumps(tree_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req_tree, timeout=12) as r:
            new_tree_sha = json.loads(r.read().decode("utf-8"))["sha"]

        # 4. Create one single commit
        url_new_commit = "https://api.github.com/repos/PDub1101/market-insight/git/commits"
        commit_payload = {
            "message": f"Automated daily model sync ({dt.datetime.now().strftime('%Y-%m-%d')})",
            "tree": new_tree_sha,
            "parents": [commit_sha],
        }
        req_new_commit = urllib.request.Request(
            url_new_commit,
            data=json.dumps(commit_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req_new_commit, timeout=12) as r:
            new_commit_sha = json.loads(r.read().decode("utf-8"))["sha"]

        # 5. Point main branch to new commit
        req_patch = urllib.request.Request(
            url_ref,
            data=json.dumps({"sha": new_commit_sha}).encode("utf-8"),
            headers=headers,
            method="PATCH",
        )
        with urllib.request.urlopen(req_patch, timeout=12):
            pass

        print(f"Synced updated Day Tab (single commit {new_commit_sha[:8]}) to https://pdub1101.github.io/market-insight/")
    except Exception as exc:
        print(f"[warn] GitHub sync skipped: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"[warn] GitHub sync skipped: {exc}", file=sys.stderr)


def print_concise_summary(state: Dict[str, Any]) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    latest = state["day_tabs"][-1]
    p = latest["portfolio"]
    al = latest["adaptive_learning"]
    gem = latest.get("gemini_insight", {})
    print("=" * 74)
    print(f"MARKET INSIGHT $7,000 GEMINI MODEL UPDATE - {latest['date']} ({latest['tab_label']})")
    print("=" * 74)
    print(f"AI Model Portfolio ($7K)  : ${p['total_value']:,.2f} ({p['total_gain_dollars']:+,.2f} / {p['total_gain_pct']:+.2f}%)")
    print(f"S&P 500 Mutual Fund (SPY) : ${p['spy_benchmark_value']:,.2f} ({p['spy_gain_dollars']:+,.2f} / {p['spy_gain_pct']:+.2f}%)")
    print(f"60/40 Balanced Fund       : ${p['balanced_60_40_value']:,.2f}")
    print(f"4.50% High-Yield Savings  : ${p['hysa_value']:,.2f}")
    print("-" * 74)
    print(f"Gemini Catalyst Analysis  : {gem.get('catalyst_headline', '')}")
    print(f"Gemini Risk Veto Check    : {gem.get('veto_note', '')}")
    print(f"New Lesson Saved to Memory: {gem.get('lesson_learned', '')}")
    print(f"Adaptive Factor Weights   : {al['learning_note']}")
    print("=" * 74)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update $7,000 stock model and webpage data.")
    parser.add_argument("--init-history", action="store_true", help="Initialize 5-day backtest preview + Today tab")
    parser.add_argument("--note", type=str, default=None, help="Custom AI briefing note for today's tab")
    args = parser.parse_args()

    updated_state = run_update(init_history=args.init_history, custom_note=args.note)
    print_concise_summary(updated_state)
