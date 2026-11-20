"""Gamma-exposure (GEX) profile by strike for a single ticker, for the CIX chart's side panel.

Open interest from the yfinance option chain; Black-Scholes gamma priced off a smooth per-expiry
IV smile (yfinance's own IVs are unreliable, with 1e-5 placeholders and stale pre-market prints);
classic dealer convention: dealers long calls, short puts. That convention is an assumption, not
a measurement, and flow-based models (e.g. VolSignals) can sign strikes differently.

It is a snapshot of today's open interest, so it only makes sense drawn against today's price.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
CACHE_DIR = Path(__file__).parent / "options_cache"
# Yahoo has no option chain on the index ticker people usually type for these
OPTION_ROOT = {"^GSPC": "^SPX", "^NDX": "^NDX", "^RUT": "^RUT"}


def _gex(S, df):
    d1 = (np.log(S / df.strike) + 0.5 * df.iv ** 2 * df["T"]) / (df.iv * np.sqrt(df["T"]))
    gamma = norm.pdf(d1) / (S * df.iv * np.sqrt(df["T"]))
    sign = np.where(df.type == "C", 1.0, -1.0)
    return sign * df.openInterest * 100 * gamma * S * S * 0.01          # $ per 1% move


def _fetch_chain(root, max_days):
    t = yf.Ticker(root)
    hist = t.history(period="3mo").Close
    spot = float(hist.iloc[-1])
    rv30 = float(np.log(hist).diff().tail(30).std() * np.sqrt(252))
    today = pd.Timestamp.now(tz=NY).normalize().tz_localize(None)
    exps = [e for e in t.options if pd.Timestamp(e) <= today + pd.Timedelta(days=max_days)]
    rows = []
    for e in exps:
        c = t.option_chain(e)
        for typ, df in (("C", c.calls), ("P", c.puts)):
            df = df[["strike", "openInterest", "impliedVolatility", "lastPrice", "bid", "ask"]].copy()
            df["type"], df["exp"] = typ, e
            rows.append(df)
        time.sleep(0.4)
    chain = pd.concat(rows) if rows else pd.DataFrame()
    chain.attrs["rv30"] = rv30
    return chain, spot


def _is_placeholder(v):
    """yfinance fills missing IVs with 1e-5 or powers of two (0.25, 0.125, 0.0625, ...) pre-market."""
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        l2 = np.log2(v)
    return (v < 0.01) | ((v <= 0.5) & (np.abs(l2 - np.round(l2)) < 0.003))


def _implied_vol(price, S, K, T, cp):
    if not (price > 0 and T > 0):
        return np.nan
    def f(s):
        d1 = (np.log(S / K) + 0.5 * s * s * T) / (s * np.sqrt(T)); d2 = d1 - s * np.sqrt(T)
        model = S * norm.cdf(d1) - K * norm.cdf(d2) if cp == "C" else K * norm.cdf(-d2) - S * norm.cdf(-d1)
        return model - price
    try:
        return brentq(f, 0.02, 5.0) if f(0.02) < 0 < f(5.0) else np.nan
    except ValueError:
        return np.nan


@st.cache_data(ttl=900, show_spinner=False)
def gex_profile(ticker, max_days=21, n_buckets=40, band=0.15):
    """Returns dict(profile=DataFrame[strike, gex_bn, call_oi, put_oi], spot, flip, total_bn,
    asof, expiries, source, note) or dict(error=...)."""
    root = OPTION_ROOT.get(ticker, ticker)
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"gex_chain_{root.replace('^', '')}_{max_days}d.pkl"
    source = "live"
    try:
        chain, spot = _fetch_chain(root, max_days)
        if chain.empty:
            return {"error": f"No listed options found for {root}."}
        chain.attrs.update(spot=spot, asof=str(pd.Timestamp.now(tz=NY)))
        chain.to_pickle(cache)
    except Exception as err:  # mostly yfinance rate limits
        if not cache.exists():
            return {"error": f"Option chain unavailable for {root} ({type(err).__name__})."}
        chain = pd.read_pickle(cache); spot = chain.attrs["spot"]; source = "cached"

    now = pd.Timestamp.now(tz=NY)
    o = chain.dropna(subset=["openInterest"])
    o = o[o.openInterest > 0].reset_index(drop=True)
    exp_ts = pd.to_datetime(o.exp).dt.tz_localize(NY) + pd.Timedelta(hours=16)
    o["T"] = ((exp_ts - now).dt.total_seconds() / (365 * 86400)).clip(lower=1 / (365 * 24 * 4))
    # drop today's expiry: with hours left its ATM gamma explodes and swamps the real positioning
    o = o[exp_ts.dt.normalize() > now.normalize()]
    if o.empty:
        return {"error": "Only same-day expiries in range."}
    o["k"] = np.log(o.strike / spot)
    # clean IVs: drop yfinance placeholders; where an expiry has too few, back IV out of the option's
    # mid (or last) price, which pairs with the last close (that is also what `spot` is pre-market)
    o["ivq"] = o.impliedVolatility.where(~_is_placeholder(o.impliedVolatility) & o.impliedVolatility.between(0.03, 5))
    last_close = now.normalize() + pd.Timedelta(hours=16)
    while last_close > now or last_close.weekday() >= 5:
        last_close -= pd.Timedelta(days=1)
    otm = ((o.type == "P") & (o.k < 0.01)) | ((o.type == "C") & (o.k > -0.01))
    if "lastPrice" in o.columns:
        for e, g in o[otm & (o.k.abs() < 0.3)].groupby("exp"):
            if g.ivq.notna().sum() >= 8:
                continue
            Tc = (pd.Timestamp(e).tz_localize(NY) + pd.Timedelta(hours=16) - last_close).total_seconds() / (365 * 86400)
            mid = ((g.bid + g.ask) / 2).where((g.bid > 0) & (g.ask > 0), g.lastPrice)
            o.loc[g.index, "ivq"] = [
                _implied_vol(px, spot, K, Tc, cp) for px, K, cp in zip(mid, g.strike, g.type)]

    def smile(df):
        q = df[((df.type == "P") & (df.k < 0.01)) | ((df.type == "C") & (df.k > -0.01))]
        q = q[q.ivq.between(0.06, 2.5) & (q.k.abs() < 0.3)]
        return np.polyfit(q.k, q.ivq, 2, w=np.sqrt(q.openInterest + 1)) if len(q) >= 6 else None

    fits = {e: smile(g) for e, g in o.groupby("exp")}
    good = [f for f in fits.values() if f is not None]
    rv30 = chain.attrs.get("rv30", np.nan)
    if good:
        fallback = good[len(good) // 2]
    elif np.isfinite(rv30):
        fallback = np.array([0.0, 0.0, 1.1 * rv30])                 # flat vol = 1.1x 30d realised
    else:
        return {"error": "Not enough clean IV quotes to fit a smile."}
    o["iv"] = [np.polyval(fits[e] if fits.get(e) is not None else fallback, k) for e, k in zip(o.exp, o.k)]
    o["iv"] = o.iv.clip(0.06, 2.5)
    o["gex"] = _gex(spot, o)

    # bucket strikes to a round step that gives ~n_buckets rows inside +-band of spot
    # smallest round step >= the target that leaves most buckets populated (listed strikes are often
    # $1 near spot for weeklies but $5 further out, which otherwise draws as spikes with gaps)
    atm = float(np.clip(np.polyval(fallback, 0.0), 0.06, 2.5))
    band = max(band, min(0.35, 1.5 * atm * np.sqrt(max_days / 365)))
    raw = 2 * band * spot / n_buckets
    inband = o[(o.strike >= spot * (1 - band)) & (o.strike <= spot * (1 + band))]
    for step in [s for s in [0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500] if s >= raw]:
        filled = (inband.strike / step).round().nunique()
        if filled >= 0.6 * (2 * band * spot / step):
            break
    o["b"] = (o.strike / step).round() * step
    w = o[(o.b >= spot * (1 - band)) & (o.b <= spot * (1 + band))]
    prof = pd.DataFrame({
        "gex_bn": w.groupby("b").gex.sum() / 1e9,
        "call_oi": w[w.type == "C"].groupby("b").openInterest.sum(),
        "put_oi": w[w.type == "P"].groupby("b").openInterest.sum(),
    }).fillna(0).rename_axis("strike").reset_index()

    grid = np.linspace(spot * (1 - band), spot * (1 + band), 241)
    tot = np.array([_gex(s, o).sum() for s in grid])
    flip = next((grid[i] + (grid[i + 1] - grid[i]) * (-tot[i]) / (tot[i + 1] - tot[i])
                 for i in range(len(grid) - 1) if np.sign(tot[i]) != np.sign(tot[i + 1])), np.nan)
    return {"profile": prof, "spot": spot, "flip": flip, "total_bn": o.gex.sum() / 1e9, "step": step,
            "asof": chain.attrs.get("asof", ""), "atm_iv": atm, "band": band, "expiries": sorted(o.exp.unique()), "source": source,
            "root": root}
