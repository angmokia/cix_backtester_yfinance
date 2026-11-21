"""Point-in-time market regimes for the CIX backtester.

Every label only uses data that was public on that day: FRED daily rates as of their own date,
CPI shifted ~45 days and payrolls/unemployment ~35 days for their release lags, equity/vol data
from Yahoo as of the close. Nothing is fitted on the full sample (quantile buckets use a trailing
5-year rank), so filtering a backtest by these regimes adds no lookahead.

Data: FRED keyless CSV (fredgraph.csv) + yfinance; both work on Streamlit Cloud.
"""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
WINDOWS = {"1M": 21, "3M": 63, "6M": 126}

# Display order + colours for each family's states (green = benign for risk assets, red = hostile)
GREEN, LGREEN, GREY, AMBER, RED, BLUE, PURPLE = "#2ecc71", "#a3e48d", "#8c8c8c", "#f5b041", "#e74c3c", "#5dade2", "#af7ac5"
FAMILIES = {
    "Fed policy cycle": {
        "Hiking": RED, "Pause after hikes": AMBER, "Cutting": GREEN, "Pause after cuts": BLUE},
    "Fed pricing (2y vs funds)": {
        "Hikes priced": RED, "Neutral": GREY, "Cuts priced": GREEN},
    "Front-end repricing (Δ2y)": {
        "Hawkish repricing": RED, "Stable": GREY, "Dovish repricing": GREEN},
    "Curve shape (2s10s)": {
        "Inverted": RED, "Flat (0–50bp)": AMBER, "Normal (50–150bp)": GREY, "Steep (>150bp)": BLUE},
    "Curve dynamics (2s10s)": {
        "Bull steepener": GREEN, "Bull flattener": BLUE, "Bear steepener": AMBER, "Bear flattener": RED,
        "Parallel / quiet": GREY},
    "10y yield trend": {
        "Sharp sell-off": RED, "Yields rising": AMBER, "Yields falling": LGREEN, "Sharp rally": GREEN},
    "Equity drawdown (SPX)": {
        "Bull (<5%)": GREEN, "Pullback (5–10%)": LGREEN, "Correction (10–20%)": AMBER, "Bear (>20%)": RED},
    "Equity trend (SPX 200d)": {
        "Above & rising": GREEN, "Above & falling": AMBER, "Below & rising": BLUE, "Below & falling": RED},
    "VIX level": {
        "<15": GREEN, "15–20": LGREEN, "20–30": AMBER, ">30": RED},
    "VIX momentum": {
        "Falling / crush": GREEN, "Steady": GREY, "Rising": AMBER, "Spiking": RED},
    "Inflation level (CPI YoY)": {
        "Below 2%": BLUE, "2–3%": GREEN, "3–5%": AMBER, "Above 5%": RED},
    "Inflation momentum": {
        "Decelerating": GREEN, "Stable": GREY, "Accelerating": RED},
    "Labour market (Sahm)": {
        "Healthy (<0.3)": GREEN, "Softening (0.3–0.5)": AMBER, "Sahm triggered (≥0.5)": RED},
    "Growth × inflation quadrant": {
        "Goldilocks": GREEN, "Reflation": AMBER, "Disinflationary slowdown": BLUE, "Stagflation": RED},
    "Dollar (DXY)": {
        "Weakening": GREEN, "Range": GREY, "Strengthening": RED},
}
DESCRIPTIONS = {
    "Fed policy cycle": "Direction of the last Fed funds target change; 'pause' once 6 months pass without a move.",
    "Fed pricing (2y vs funds)": "2y Treasury minus the funds target (mid of range): above +25bp = hikes priced, below −50bp = cuts priced.",
    "Front-end repricing (Δ2y)": "Change in the 2y yield over the window: the market moving its Fed path before the Fed does.",
    "Curve shape (2s10s)": "Level of 10y minus 2y.",
    "Curve dynamics (2s10s)": "Over the window: steepener/flattener = change in 2s10s; bull/bear = average yield level falling/rising. "
                              "Moves under the threshold are 'parallel / quiet'.",
    "10y yield trend": "Change in the 10y yield over the window.",
    "Equity drawdown (SPX)": "S&P 500 distance below its 52-week closing high.",
    "Equity trend (SPX 200d)": "S&P 500 above/below its 200-day average, and whether that average is rising over 20 days.",
    "VIX level": "VIX close.",
    "VIX momentum": "VIX vs its own 20-day average: >1.25× spiking, 1.05–1.25× rising, 0.9–1.05× steady, <0.9× falling.",
    "Inflation level (CPI YoY)": "Headline CPI YoY, as published (~45 days after the reference month).",
    "Inflation momentum": "3-month change in CPI YoY: beyond ±0.25pp = accelerating / decelerating.",
    "Labour market (Sahm)": "Real-time Sahm gap: 3m avg unemployment minus its prior-12m low, as published (~35 days lag).",
    "Growth × inflation quadrant": "Growth = 3m avg unemployment falling (up) or rising (down) over 3 months; inflation = CPI YoY momentum.",
    "Dollar (DXY)": "DXY % change over the window (±3% on a 3M window, scaled for others).",
}
CUSTOM = "Custom driver"
DRIVERS = {  # name -> (column in drivers frame, unit for display)
    "Fed funds target": ("ff", "%"), "2y yield": ("us2", "%"), "10y yield": ("us10", "%"),
    "2s10s spread": ("s2s10", "bp"), "10y − 3m spread": ("s3m10", "bp"), "VIX": ("vix", ""),
    "CPI YoY": ("cpi_yoy", "%"), "SPX drawdown": ("dd", "%"), "DXY": ("dxy", ""), "Sahm gap": ("sahm", "pp"),
}
CUSTOM_BUCKETS = {
    "Rising / Falling": {"Falling": GREEN, "Rising": RED},
    "Terciles (trailing 5y rank)": {"Low tercile": BLUE, "Mid tercile": GREY, "High tercile": AMBER},
    "Quintiles (trailing 5y rank)": {"Q1 (low)": BLUE, "Q2": LGREEN, "Q3": GREY, "Q4": AMBER, "Q5 (high)": RED},
}


def states_for(family, custom=None):
    if family == CUSTOM:
        return CUSTOM_BUCKETS[(custom or {}).get("bucket", "Terciles (trailing 5y rank)")]
    return FAMILIES[family]


# ---------------------------------------------------------------- data
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _fred(series_id):
    df = pd.read_csv(FRED_CSV.format(series_id), na_values=".")
    return pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                     index=pd.to_datetime(df.iloc[:, 0]), name=series_id).dropna()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _yahoo(tickers, start):
    raw = yf.download(list(tickers), start=start, progress=False, auto_adjust=False,
                      group_by="ticker", threads=False)
    out = {}
    for t in tickers:
        try:
            out[t] = raw[t]["Close"]
        except (KeyError, TypeError):
            pass
    return pd.DataFrame(out).dropna(how="all")


def _fed_target():
    """Daily Fed funds target: single target to 2008-12-15, midpoint of the range after."""
    old = _fred("DFEDTAR")
    new = _fred("DFEDTARU") - 0.125
    return pd.concat([old[old.index < new.index[0]], new])


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def drivers(start):
    """Numeric regime drivers on S&P trading days from `start` (with 2y warm-up), point-in-time."""
    warm = (pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    yh = _yahoo(("^GSPC", "^VIX", "DX-Y.NYB"), warm)
    idx = yh["^GSPC"].dropna().index
    ff = _fed_target()
    us2, us10, us3m = _fred("DGS2"), _fred("DGS10"), _fred("DTB3")
    cpi = _fred("CPIAUCSL")
    cpi_yoy = (cpi.pct_change(12) * 100)
    cpi_yoy.index = cpi_yoy.index + pd.Timedelta(days=45)              # published mid next month
    ur = _fred("UNRATE")
    ur3 = ur.rolling(3).mean()
    sahm = ur3 - ur3.shift(1).rolling(12).min()
    ur3_chg = ur3.diff(3)
    for s in (sahm, ur3_chg):
        s.index = s.index + pd.Timedelta(days=35)                     # jobs report ~first Friday

    def on_idx(s):
        return s.reindex(s.index.union(idx)).ffill().reindex(idx)

    d = pd.DataFrame({
        "spx": yh["^GSPC"].reindex(idx), "vix": on_idx(yh["^VIX"].dropna()) if "^VIX" in yh else np.nan,
        "dxy": on_idx(yh["DX-Y.NYB"].dropna()) if "DX-Y.NYB" in yh else np.nan,
        "ff": on_idx(ff), "us2": on_idx(us2), "us10": on_idx(us10), "us3m": on_idx(us3m),
        "cpi_yoy": on_idx(cpi_yoy.dropna()), "sahm": on_idx(sahm.dropna()), "ur3_chg": on_idx(ur3_chg.dropna()),
    }, index=idx)
    d["s2s10"] = (d.us10 - d.us2) * 100
    d["s3m10"] = (d.us10 - d.us3m) * 100
    d["dd"] = (d.spx / d.spx.rolling(252, min_periods=1).max() - 1) * 100
    # Fed cycle: direction of the last target change and days since it, on the daily FRED calendar
    chg = ff.diff().fillna(0)
    moves = chg[chg != 0]
    last_dir = np.sign(moves).reindex(ff.index).ffill()
    last_date = pd.Series(moves.index, index=moves.index).reindex(ff.index).ffill()
    d["fed_dir"] = on_idx(last_dir)
    d["fed_days"] = on_idx((pd.Series(ff.index, index=ff.index) - last_date).dt.days)
    return d


# ---------------------------------------------------------------- labels
def _cut(x, edges, names):
    return pd.Series(pd.cut(x, edges, labels=names, right=False).astype(object), index=x.index).where(x.notna())


def labels(d, window=63, custom=None):
    """DataFrame of regime labels (one column per family) on the drivers' index."""
    w = int(window)
    sc = np.sqrt(w / 63)                    # thresholds are set for a 3M window, scaled like vol
    L = pd.DataFrame(index=d.index)

    hiking = d.fed_dir > 0
    recent = d.fed_days <= 182
    L["Fed policy cycle"] = np.select(
        [recent & hiking, recent & ~hiking, ~recent & hiking, ~recent & ~hiking],
        ["Hiking", "Cutting", "Pause after hikes", "Pause after cuts"], None)
    L.loc[d.fed_dir.isna(), "Fed policy cycle"] = None

    gap = (d.us2 - d.ff) * 100
    L["Fed pricing (2y vs funds)"] = _cut(gap, [-np.inf, -50, 25, np.inf], ["Cuts priced", "Neutral", "Hikes priced"])

    d2 = d.us2.diff(w) * 100
    t = 25 * sc
    L["Front-end repricing (Δ2y)"] = _cut(d2, [-np.inf, -t, t, np.inf], ["Dovish repricing", "Stable", "Hawkish repricing"])

    L["Curve shape (2s10s)"] = _cut(d.s2s10, [-np.inf, 0, 50, 150, np.inf],
                                    ["Inverted", "Flat (0–50bp)", "Normal (50–150bp)", "Steep (>150bp)"])

    dS = d.s2s10.diff(w)
    dL = ((d.us2 + d.us10) / 2).diff(w) * 100
    tc = 10 * sc
    L["Curve dynamics (2s10s)"] = np.select(
        [dS.abs() < tc, (dS > 0) & (dL < 0), (dS > 0) & (dL >= 0), (dS < 0) & (dL < 0), (dS < 0) & (dL >= 0)],
        ["Parallel / quiet", "Bull steepener", "Bear steepener", "Bull flattener", "Bear flattener"], None)
    L.loc[dS.isna() | dL.isna(), "Curve dynamics (2s10s)"] = None

    d10 = d.us10.diff(w) * 100
    t = 50 * sc
    L["10y yield trend"] = _cut(d10, [-np.inf, -t, 0, t, np.inf], ["Sharp rally", "Yields falling", "Yields rising", "Sharp sell-off"])

    L["Equity drawdown (SPX)"] = _cut(-d.dd, [0, 5, 10, 20, np.inf],
                                      ["Bull (<5%)", "Pullback (5–10%)", "Correction (10–20%)", "Bear (>20%)"])

    ma = d.spx.rolling(200).mean()
    up, rising = d.spx > ma, ma.diff(20) > 0
    L["Equity trend (SPX 200d)"] = np.select(
        [up & rising, up & ~rising, ~up & rising, ~up & ~rising],
        ["Above & rising", "Above & falling", "Below & rising", "Below & falling"], None)
    L.loc[ma.isna(), "Equity trend (SPX 200d)"] = None

    L["VIX level"] = _cut(d.vix, [0, 15, 20, 30, np.inf], ["<15", "15–20", "20–30", ">30"])
    L["VIX momentum"] = _cut(d.vix / d.vix.rolling(20).mean(), [0, 0.9, 1.05, 1.25, np.inf],
                             ["Falling / crush", "Steady", "Rising", "Spiking"])

    L["Inflation level (CPI YoY)"] = _cut(d.cpi_yoy, [-np.inf, 2, 3, 5, np.inf], ["Below 2%", "2–3%", "3–5%", "Above 5%"])
    infl_mom = d.cpi_yoy - d.cpi_yoy.shift(63)
    L["Inflation momentum"] = _cut(infl_mom, [-np.inf, -0.25, 0.25, np.inf], ["Decelerating", "Stable", "Accelerating"])

    L["Labour market (Sahm)"] = _cut(d.sahm, [-np.inf, 0.3, 0.5, np.inf],
                                     ["Healthy (<0.3)", "Softening (0.3–0.5)", "Sahm triggered (≥0.5)"])

    g_up = d.ur3_chg <= 0
    i_up = infl_mom > 0
    L["Growth × inflation quadrant"] = np.select(
        [g_up & ~i_up, g_up & i_up, ~g_up & ~i_up, ~g_up & i_up],
        ["Goldilocks", "Reflation", "Disinflationary slowdown", "Stagflation"], None)
    L.loc[d.ur3_chg.isna() | infl_mom.isna(), "Growth × inflation quadrant"] = None

    dx = d.dxy.pct_change(w) * 100
    t = 3 * sc
    L["Dollar (DXY)"] = _cut(dx, [-np.inf, -t, t, np.inf], ["Weakening", "Range", "Strengthening"])

    if custom:
        L[CUSTOM] = custom_labels(d, custom, w)
    return L


def custom_driver_series(d, custom, window):
    col = DRIVERS[custom["driver"]][0]
    x = d[col]
    if custom.get("transform") == "Change over window":
        x = x.diff(int(window))
    return x


def custom_labels(d, custom, window):
    x = custom_driver_series(d, custom, window)
    b = custom.get("bucket", "Terciles (trailing 5y rank)")
    if b == "Rising / Falling":
        if custom.get("transform") != "Change over window":
            x = x.diff(int(window))
        return pd.Series(np.where(x > 0, "Rising", "Falling"), index=x.index).where(x.notna() & (x != 0))
    # trailing 5y percentile rank: today's value vs only its own past, so no lookahead
    r = x.rolling(1260, min_periods=252).rank(pct=True)
    names = list(CUSTOM_BUCKETS[b])
    return _cut(r, np.linspace(0, 1, len(names) + 1)[:-1].tolist() + [1.0001], names)


def custom_label_name(custom, window_label):
    tr = f"{window_label} change in " if custom.get("transform") == "Change over window" else ""
    return f"{tr}{custom['driver']} — {custom['bucket'].split(' (')[0].lower()}"


# ---------------------------------------------------------------- analysis
def breakdown(lab, fwd_all, signal_dates, horizons, suffix, expected_direction, family_states):
    """Per regime state: share of days, signal count, and for each horizon the signal's avg/win
    vs the baseline (every day in that regime) and the edge = signal − baseline."""
    inc = expected_direction == "Increase"
    win = (lambda s: (s > 0).mean() * 100) if inc else (lambda s: (s < 0).mean() * 100)
    sig_lab = lab.reindex(signal_dates)
    rows = []
    for state in list(family_states) + ["All regimes"]:
        in_state = lab.notna() if state == "All regimes" else (lab == state)
        if state != "All regimes" and not in_state.any():
            continue
        sig = signal_dates[(sig_lab.notna() if state == "All regimes" else (sig_lab == state)).values]
        row = {"Regime": state, "% of days": in_state.mean() * 100, "Signals": len(sig)}
        for h in horizons:
            s = fwd_all[f"Forward_{h}D_{suffix}"]
            base = s[in_state.reindex(s.index, fill_value=False)].dropna()
            sg = s.reindex(sig).dropna()
            row[f"{h}D signal avg"] = sg.mean() if len(sg) else np.nan
            row[f"{h}D signal win %"] = win(sg) if len(sg) else np.nan
            row[f"{h}D base avg"] = base.mean() if len(base) else np.nan
            row[f"{h}D base win %"] = win(base) if len(base) else np.nan
            row[f"{h}D edge"] = row[f"{h}D signal avg"] - row[f"{h}D base avg"]
        rows.append(row)
    return pd.DataFrame(rows)


def segments(lab):
    """Contiguous runs of the same label -> list of (start, end, state)."""
    lab = lab.dropna()
    if lab.empty:
        return []
    brk = lab.ne(lab.shift()).cumsum()
    out = []
    for _, g in lab.groupby(brk):
        out.append((g.index[0], g.index[-1], g.iloc[0]))
    return out


def current(d, L, window_label):
    """Latest state of every family with the reading behind it."""
    last = d.index[-1]
    dl = d.loc[last]
    w = WINDOWS[window_label]
    readings = {
        "Fed policy cycle": f"target {dl.ff:.2f}%, last move {int(dl.fed_days)}d ago",
        "Fed pricing (2y vs funds)": f"2y − funds {(dl.us2 - dl.ff) * 100:+.0f}bp",
        "Front-end repricing (Δ2y)": f"Δ2y {window_label} {(d.us2.diff(w).iloc[-1]) * 100:+.0f}bp",
        "Curve shape (2s10s)": f"2s10s {dl.s2s10:+.0f}bp",
        "Curve dynamics (2s10s)": f"Δ2s10s {d.s2s10.diff(w).iloc[-1]:+.0f}bp, Δlevel {((d.us2 + d.us10) / 2).diff(w).iloc[-1] * 100:+.0f}bp",
        "10y yield trend": f"Δ10y {window_label} {d.us10.diff(w).iloc[-1] * 100:+.0f}bp",
        "Equity drawdown (SPX)": f"{dl.dd:.1f}% from 52w high",
        "Equity trend (SPX 200d)": f"SPX {dl.spx / d.spx.rolling(200).mean().iloc[-1] * 100 - 100:+.1f}% vs 200d",
        "VIX level": f"VIX {dl.vix:.1f}",
        "VIX momentum": f"{dl.vix / d.vix.rolling(20).mean().iloc[-1]:.2f}× 20d avg",
        "Inflation level (CPI YoY)": f"CPI {dl.cpi_yoy:.1f}% YoY",
        "Inflation momentum": f"Δ3m {dl.cpi_yoy - d.cpi_yoy.shift(63).iloc[-1]:+.2f}pp",
        "Labour market (Sahm)": f"Sahm gap {dl.sahm:.2f}pp",
        "Growth × inflation quadrant": f"Δ3m unemp {dl.ur3_chg:+.2f}pp",
        "Dollar (DXY)": f"DXY {window_label} {d.dxy.pct_change(w).iloc[-1] * 100:+.1f}%",
    }
    return pd.DataFrame([{"Regime family": f, "Today": L[f].iloc[-1] if f in L else None, "Reading": readings.get(f, "")}
                         for f in FAMILIES])
