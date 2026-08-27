import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

pio.templates.default = "plotly_dark"

st.set_page_config(page_title="CIX Backtest", layout="wide")
st.title("CIX Backtest")

@st.cache_data
def fetch_yahoo_data(tickers, start_date, end_date):
    """
    Fetch daily close prices from Yahoo Finance for one or more tickers.
    Returns a DataFrame indexed by date, with one column per ticker.
    """
    try:
        import yfinance as yf

        if isinstance(tickers, str):
            tickers = [tickers]
        tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order

        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )

        if raw.empty:
            return pd.DataFrame()

        data = pd.DataFrame(index=raw.index)

        if len(tickers) == 1:
            # yfinance returns a flat column index when only one ticker is requested
            ticker = tickers[0]
            if isinstance(raw.columns, pd.MultiIndex):
                data[ticker] = raw[ticker]['Close']
            else:
                data[ticker] = raw['Close']
        else:
            for ticker in tickers:
                try:
                    data[ticker] = raw[ticker]['Close']
                except (KeyError, TypeError):
                    # Ticker failed to download (delisted/invalid/no data)
                    st.warning(f"No data returned for '{ticker}' - check the ticker symbol")

        data.index.name = 'date'
        return data.dropna(how='all')

    except Exception as e:
        st.error(f"Yahoo Finance API error: {str(e)}")
        return pd.DataFrame()

def process_economic_data(data, economic_tickers):
    """Forward fill economic data to handle missing values on non-release days"""
    processed_data = data.copy()
    
    for ticker in economic_tickers:
        if ticker in processed_data.columns:
            # Forward fill missing values - economic data stays constant until next release
            processed_data[ticker] = processed_data[ticker].ffill()
            
    return processed_data

def calculate_dependent_variable(data, ticker_weights):
    active_tickers = {k: v for k, v in ticker_weights.items() if k and v != 0}
    if not active_tickers:
        return pd.Series(dtype=float), pd.DataFrame()
    
    # Requires every active ticker to have data - the combined series starts from whichever
    # component starts LATEST, since a partial combination (e.g. only one leg of a spread)
    # isn't a meaningful value for the dependent variable.
    ticker_data = data[list(active_tickers.keys())].dropna()
    if len(ticker_data) == 0:
        return pd.Series(dtype=float), pd.DataFrame()

    weighted_components = pd.DataFrame(index=ticker_data.index)
    for ticker, weight in active_tickers.items():
        weighted_components[f"{weight:+.1f}×{ticker}"] = weight * ticker_data[ticker]

    dependent_var = weighted_components.sum(axis=1)
    result_data = ticker_data.copy()
    for col in weighted_components.columns:
        result_data[col] = weighted_components[col]
    result_data['Dependent_Variable'] = dependent_var
    
    return dependent_var, result_data

@st.cache_data
def fetch_yahoo_ohlc(tickers, start_date, end_date):
    """
    Fetch Open/High/Low/Close (not just Close) for the dependent-variable candlestick chart.
    Returns {'Open': df, 'High': df, 'Low': df, 'Close': df}, each a date-indexed DataFrame
    with one column per ticker - same shape/fetch pattern as fetch_yahoo_data.
    """
    try:
        import yfinance as yf

        if isinstance(tickers, str):
            tickers = [tickers]
        tickers = list(dict.fromkeys(tickers))

        raw = yf.download(
            tickers, start=start_date, end=end_date,
            progress=False, auto_adjust=False, group_by="ticker", threads=True,
        )
        if raw.empty:
            return {}

        result = {}
        for field in ['Open', 'High', 'Low', 'Close']:
            data = pd.DataFrame(index=raw.index)
            if len(tickers) == 1:
                ticker = tickers[0]
                data[ticker] = raw[ticker][field] if isinstance(raw.columns, pd.MultiIndex) else raw[field]
            else:
                for ticker in tickers:
                    try:
                        data[ticker] = raw[ticker][field]
                    except (KeyError, TypeError):
                        pass
            data.index.name = 'date'
            result[field] = data.dropna(how='all')
        return result

    except Exception as e:
        st.error(f"Yahoo Finance API error: {str(e)}")
        return {}

def calculate_dependent_variable_ohlc(ohlc_data, ticker_weights):
    """
    Weighted OHLC for the dependent variable. Open/Close are the same weighted sum used to
    build the (Close-based) dependent variable, applied to that field. High/Low need care:
    for a NEGATIVE weight, that leg's contribution to the combination's daily High actually
    comes from its own Low (multiplying by a negative flips which extreme pushes the sum up),
    and vice versa - so this isn't just summing each ticker's own High/Low. Still an
    approximation of the combined path's true intraday extreme (would need intraday data for
    that), but a correctly-signed one - the same construction index providers use for a
    synthetic instrument's OHLC.
    """
    active_tickers = {k: v for k, v in ticker_weights.items() if k and v != 0}
    if not active_tickers or not ohlc_data:
        return pd.DataFrame()

    tickers = list(active_tickers.keys())
    per_field = {}
    for field in ['Open', 'High', 'Low', 'Close']:
        field_data = ohlc_data.get(field, pd.DataFrame())
        if any(t not in field_data.columns for t in tickers):
            return pd.DataFrame()
        per_field[field] = field_data[tickers]

    idx = per_field['Open'].dropna().index
    for field in ['High', 'Low', 'Close']:
        idx = idx.intersection(per_field[field].dropna().index)
    if len(idx) == 0:
        return pd.DataFrame()
    open_d, high_d, low_d, close_d = (per_field[f].loc[idx] for f in ['Open', 'High', 'Low', 'Close'])

    weighted_open = sum(w * open_d[t] for t, w in active_tickers.items())
    weighted_close = sum(w * close_d[t] for t, w in active_tickers.items())
    weighted_high = sum((w * high_d[t] if w > 0 else w * low_d[t]) for t, w in active_tickers.items())
    weighted_low = sum((w * low_d[t] if w > 0 else w * high_d[t]) for t, w in active_tickers.items())

    result = pd.DataFrame({'Open': weighted_open, 'High': weighted_high, 'Low': weighted_low, 'Close': weighted_close})
    # guard against float edge cases so Open/Close always sit within [Low, High]
    result['High'] = result[['Open', 'High', 'Low', 'Close']].max(axis=1)
    result['Low'] = result[['Open', 'High', 'Low', 'Close']].min(axis=1)
    return result.dropna()

def evaluate_indicator_conditions(data, indicators):
    individual_conditions = {}
    rolling_return_columns = {}
    cumulative_sum_columns = {}
    condition_results = {}
    overall_mask = pd.Series(True, index=data.index)
    
    if not indicators:
        return overall_mask, condition_results, individual_conditions, rolling_return_columns, cumulative_sum_columns
    
    for indicator in indicators:
        ticker = indicator['ticker']
        if not ticker or ticker not in data.columns:
            continue
        
        if indicator['type'] == 'level':
            threshold = indicator['threshold']
            above = indicator['above']
            if above:
                condition_mask = data[ticker] > threshold
                condition_name = f"{ticker} > {threshold}"
                column_name = f"{ticker}_Above_{threshold}".replace('.', '_').replace(' ', '_')
            else:
                condition_mask = data[ticker] < threshold
                condition_name = f"{ticker} < {threshold}"
                column_name = f"{ticker}_Below_{threshold}".replace('.', '_').replace(' ', '_')
        
        elif indicator['type'] == 'rolling_return':
            return_pct = indicator['return_pct']
            days = indicator['days']
            above = indicator['above']
            
            # Calculate rolling return
            rolling_return = data[ticker].pct_change(days) * 100
            
            # Store the rolling return values for display
            rolling_return_col_name = f"{ticker}_{days}D_Rolling_Return_pct".replace('.', '_').replace(' ', '_').replace('-', 'neg')
            rolling_return_columns[rolling_return_col_name] = rolling_return
            
            if above:
                condition_mask = rolling_return > return_pct
                condition_name = f"{ticker} {days}D return > {return_pct}%"
                column_name = f"{ticker}_{days}D_Return_Above_{return_pct}pct".replace('.', '_').replace(' ', '_').replace('-', 'neg')
            else:
                condition_mask = rolling_return < return_pct
                condition_name = f"{ticker} {days}D return < {return_pct}%"
                column_name = f"{ticker}_{days}D_Return_Below_{return_pct}pct".replace('.', '_').replace(' ', '_').replace('-', 'neg')
        
        elif indicator['type'] == 'cumulative_sum':
            threshold = indicator['threshold']
            days = indicator['days']
            above = indicator['above']
            
            # Calculate rolling sum of raw values (not percentage returns)
            rolling_sum = data[ticker].rolling(window=days, min_periods=1).sum()
            
            # Store the rolling sum values for display
            cumsum_col_name = f"{ticker}_{days}D_Cumulative_Sum".replace('.', '_').replace(' ', '_').replace('-', 'neg')
            cumulative_sum_columns[cumsum_col_name] = rolling_sum
            
            if above:
                condition_mask = rolling_sum > threshold
                condition_name = f"{ticker} {days}D cumsum > {threshold}"
                column_name = f"{ticker}_{days}D_CumSum_Above_{threshold}".replace('.', '_').replace(' ', '_').replace('-', 'neg')
            else:
                condition_mask = rolling_sum < threshold
                condition_name = f"{ticker} {days}D cumsum < {threshold}"
                column_name = f"{ticker}_{days}D_CumSum_Below_{threshold}".replace('.', '_').replace(' ', '_').replace('-', 'neg')
        
        # Store results
        condition_results[condition_name] = condition_mask
        individual_conditions[column_name] = condition_mask.fillna(False)
        overall_mask = overall_mask & condition_mask.fillna(False)
    
    return overall_mask, condition_results, individual_conditions, rolling_return_columns, cumulative_sum_columns

def apply_cluster_free_filter(matching_mask, cluster_free_days):
    """
    Apply cluster-free zone filter to remove signals within X days of previous signal
    
    Args:
        matching_mask: Boolean series indicating where conditions are met
        cluster_free_days: Number of days to wait after a signal before allowing another
    
    Returns:
        filtered_mask: Boolean series with cluster-free filter applied
        removed_signals: Boolean series showing which signals were removed due to clustering
    """
    if cluster_free_days == 0:
        # No filtering - return original mask
        return matching_mask, pd.Series(False, index=matching_mask.index)
    
    filtered_mask = pd.Series(False, index=matching_mask.index)
    removed_signals = pd.Series(False, index=matching_mask.index)
    
    # Get dates where original conditions are met
    signal_dates = matching_mask.index[matching_mask]
    
    if len(signal_dates) == 0:
        return filtered_mask, removed_signals
    
    # Track last accepted signal date
    last_signal_date = None
    
    for signal_date in signal_dates:
        if last_signal_date is None:
            # First signal is always accepted
            filtered_mask.loc[signal_date] = True
            last_signal_date = signal_date
        else:
            # Calculate days since last accepted signal
            days_since_last = (signal_date - last_signal_date).days
            
            if days_since_last >= cluster_free_days:
                # Enough time has passed - accept this signal
                filtered_mask.loc[signal_date] = True
                last_signal_date = signal_date
            else:
                # Too soon - reject this signal
                removed_signals.loc[signal_date] = True
    
    return filtered_mask, removed_signals

def calculate_forward_returns_all_dates(dependent_var, horizons, change_type='nominal'):
    """Calculate forward returns for ALL dates, not just matching dates. change_type='pct' uses
    % return instead of nominal change - meaningless/explosive if the series crosses or sits
    near zero (same caveat as compute_seasonality). Returns (dict, column_suffix)."""
    forward_returns_all = {}
    suffix = 'Pct' if change_type == 'pct' else 'Nominal'

    for horizon in horizons:
        col = f'Forward_{horizon}D_{suffix}'
        forward_returns_all[col] = pd.Series(index=dependent_var.index, dtype=float)

        for i in range(len(dependent_var)):
            forward_idx = i + horizon

            if forward_idx < len(dependent_var):
                initial_value = dependent_var.iloc[i]
                forward_value = dependent_var.iloc[forward_idx]

                if change_type == 'pct':
                    change = (forward_value - initial_value) / initial_value * 100 if initial_value != 0 else np.nan
                else:
                    change = forward_value - initial_value

                forward_returns_all[col].iloc[i] = change

    return forward_returns_all, suffix

def calculate_forward_returns_matching_only(dependent_var, matching_dates, horizons, expected_direction, change_type='nominal'):
    """Calculate forward returns for matching dates only (for dashboard analysis) with Win Rate
    calculation. Works for both the dependent variable and a benchmark series - just pass
    whichever series and matching dates. change_type='pct' uses % return (same zero-crossing
    caveat as calculate_forward_returns_all_dates)."""
    forward_returns = {}
    for horizon in horizons:
        horizon_data = []
        for match_date in matching_dates:
            try:
                match_idx = dependent_var.index.get_loc(match_date)
                forward_idx = match_idx + horizon

                if forward_idx < len(dependent_var):
                    initial_value = dependent_var.iloc[match_idx]
                    forward_value = dependent_var.iloc[forward_idx]
                    nominal_change = forward_value - initial_value

                    if change_type == 'pct':
                        change = (nominal_change / initial_value * 100) if initial_value != 0 else np.nan
                    else:
                        change = nominal_change

                    # Calculate hit based on expected direction
                    if pd.isna(change):
                        continue
                    if expected_direction == "Increase":
                        hit = change > 0
                    else:  # "Decrease"
                        hit = change < 0

                    horizon_data.append({
                        'Match_Date': match_date,
                        'Change': change,
                        'Hit': hit
                    })
            except (KeyError, IndexError):
                continue

        forward_returns[f'{horizon}D'] = pd.DataFrame(horizon_data) if horizon_data else pd.DataFrame()
    return forward_returns

def calculate_forward_range_matching_only(dep_ohlc, matching_dates, horizons, change_type='nominal', method='true_range'):
    """Average day-to-day range over each horizon's forward window (the `horizon` trading days
    immediately after the signal date) - same signal sample as calculate_forward_returns_matching_only,
    so it's directly comparable to those return stats. dep_ohlc must already be reindexed to the same
    index as the dependent variable used to derive matching_dates, so positions line up.

    method='true_range' uses max(High-Low, |High-PrevClose|, |Low-PrevClose|) - the standard ATR
    building block, which also captures gap risk between sessions (relevant here since the dependent
    variable is a weighted combination of tickers, not a single spot price - a gap in one leg can
    show up as a gap in the combination even on a day the combination's own High-Low looks tame).
    method='simple' uses plain High-Low if that's not a concern for your configuration.

    change_type='pct' expresses each day's range as a % of that day's prior close, same normalisation
    spirit as the pct forward returns elsewhere."""
    forward_ranges = {}
    high, low, close = dep_ohlc['High'], dep_ohlc['Low'], dep_ohlc['Close']
    prev_close = close.shift(1)
    if method == 'true_range':
        daily_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    else:
        daily_range = high - low
    if change_type == 'pct':
        daily_range = daily_range / prev_close * 100

    for horizon in horizons:
        event_avgs = []
        for match_date in matching_dates:
            try:
                match_idx = dep_ohlc.index.get_loc(match_date)
            except KeyError:
                continue
            window = daily_range.iloc[match_idx + 1: match_idx + 1 + horizon]
            if len(window) == horizon and window.notna().all():
                event_avgs.append(window.mean())
        forward_ranges[f'{horizon}D'] = pd.Series(event_avgs, dtype=float)
    return forward_ranges

def _aligned_daily_changes(dependent_var, benchmark_var, change_type='nominal'):
    """Daily changes of the dependent variable and a benchmark, aligned over their overlapping
    date range. change_type='nominal' uses daily diffs (safe for spread-type series that cross
    zero); 'pct' uses daily % returns (the conventional definition for price-like series, but
    unreliable if either series crosses zero - same caveat as everywhere else % change is used
    in this dashboard). Shared by calculate_beta and get_beta_regression_data so both use
    identical alignment logic."""
    aligned = pd.concat([dependent_var, benchmark_var], axis=1, keys=['Dep', 'Bench']).dropna()
    if len(aligned) < 3:
        return pd.DataFrame(columns=['Dep', 'Bench'])
    if change_type == 'pct':
        dep_chg = aligned['Dep'].pct_change() * 100
        bench_chg = aligned['Bench'].pct_change() * 100
    else:
        dep_chg = aligned['Dep'].diff()
        bench_chg = aligned['Bench'].diff()
    return pd.concat([dep_chg, bench_chg], axis=1, keys=['Dep', 'Bench']).replace([np.inf, -np.inf], np.nan).dropna()

Z_SCORE_WINDOWS = {"1M": 21, "3M": 63, "1Y": 252}  # trading days

def calculate_zscores(dependent_var, windows=Z_SCORE_WINDOWS):
    """Rolling z-score of the dependent variable against its own trailing history - works the
    same whether dependent_var is an outright single ticker or a weighted/constructed spread,
    since it only ever operates on the final computed series."""
    zscores = pd.DataFrame(index=dependent_var.index)
    for label, window in windows.items():
        roll_mean = dependent_var.rolling(window).mean()
        roll_std = dependent_var.rolling(window).std()
        zscores[f"{label} Z-Score"] = (dependent_var - roll_mean) / roll_std
    return zscores

def calculate_beta(dependent_var, benchmark_var, change_type='nominal'):
    """Single-figure beta of the dependent variable vs a benchmark: slope of dependent-variable
    daily change regressed on benchmark daily change (via cov/var, equivalent to OLS slope)."""
    chg = _aligned_daily_changes(dependent_var, benchmark_var, change_type)
    if len(chg) < 3 or chg['Bench'].var() == 0:
        return None
    return chg['Dep'].cov(chg['Bench']) / chg['Bench'].var()

def get_beta_regression_data(dependent_var, benchmark_var, change_type='nominal'):
    """Same beta as calculate_beta, plus the aligned scatter data, the regression intercept, and
    R^2 - everything needed to draw the beta-vs-benchmark scatter + regression line chart."""
    chg = _aligned_daily_changes(dependent_var, benchmark_var, change_type)
    if len(chg) < 3 or chg['Bench'].var() == 0:
        return chg, None, None, None
    beta, intercept = np.polyfit(chg['Bench'], chg['Dep'], 1)
    corr = chg['Bench'].corr(chg['Dep'])
    r_squared = corr ** 2 if pd.notna(corr) else None
    return chg, beta, intercept, r_squared

def create_comprehensive_dataframe(price_data, ticker_weights, indicators, dependent_var, matching_mask, individual_conditions, rolling_return_columns, cumulative_sum_columns, forward_returns_all, horizons, forward_return_suffix='Nominal', benchmark_var=None):
    # Start with the full dependent variable date range
    df = pd.DataFrame(index=dependent_var.index)

    # Add individual tickers used in dependent variable
    active_dep_tickers = [t for t, w in ticker_weights.items() if t and w != 0]
    for ticker in active_dep_tickers:
        if ticker in price_data.columns:
            df[ticker] = price_data[ticker].reindex(dependent_var.index)

    # Add dependent variable
    df['Dependent_Variable'] = dependent_var
    if benchmark_var is not None and len(benchmark_var) > 0:
        df['Benchmark'] = benchmark_var.reindex(dependent_var.index)

    # Add indicator tickers (that aren't already included)
    indicator_tickers = [ind['ticker'] for ind in indicators if ind['ticker']]
    for ticker in indicator_tickers:
        if ticker in price_data.columns and ticker not in df.columns:
            df[ticker] = price_data[ticker].reindex(dependent_var.index)
    
    # Add rolling return columns for sanity check
    for rolling_col_name, rolling_values in rolling_return_columns.items():
        aligned_rolling = rolling_values.reindex(dependent_var.index)
        df[rolling_col_name] = aligned_rolling
    
    # Add cumulative sum columns for sanity check
    for cumsum_col_name, cumsum_values in cumulative_sum_columns.items():
        aligned_cumsum = cumsum_values.reindex(dependent_var.index)
        df[cumsum_col_name] = aligned_cumsum
    
    # Add individual condition columns
    for column_name, condition_mask in individual_conditions.items():
        aligned_condition = condition_mask.reindex(dependent_var.index, fill_value=False)
        df[column_name] = aligned_condition
    
    # Add overall condition column
    aligned_matching = matching_mask.reindex(dependent_var.index, fill_value=False)
    df['Independent_Variable_Condition'] = aligned_matching
    
    # Add forward return columns for ALL dates
    for horizon in horizons:
        col = f'Forward_{horizon}D_{forward_return_suffix}'
        df[col] = forward_returns_all[col]
    
    return df

def compute_seasonality(series, freq='M', change_type='nominal'):
    """Resample to month-end/quarter-end and compute period-over-period change, grouped by
    calendar month or quarter. change_type='nominal' uses absolute change (matches the
    Nominal_Change convention used elsewhere - safe even when dependent_var is a spread that
    crosses zero); change_type='pct' uses % return, which is more intuitive for price-like
    series but meaningless/explosive if the series crosses or sits near zero."""
    # pandas deprecated the bare 'M'/'Q' resample aliases in favor of 'ME'/'QE' (removed entirely
    # in newer pandas) - map our simple 'M'/'Q' param to the modern alias pandas expects.
    resample_freq = 'ME' if freq == 'M' else 'QE'
    resampled = series.resample(resample_freq).last().dropna()
    if change_type == 'pct':
        changes = resampled.pct_change().dropna() * 100
    else:
        changes = resampled.diff().dropna()
    period_num = changes.index.month if freq == 'M' else changes.index.quarter
    period_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] if freq == 'M' else ['Q1','Q2','Q3','Q4']
    returns_df = pd.DataFrame({'Change': changes.values, 'Period': period_num, 'Year': changes.index.year})
    return returns_df, period_labels

# Sidebar Configuration
st.sidebar.header("Configuration")

# Date Range
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime.now())

# Dependent Variable
with st.sidebar.expander("Dependent Variable Components", expanded=True):
    num_components = st.number_input("Number of components", min_value=1, max_value=10, value=2)
    
    ticker_weights = {}
    for i in range(num_components):
        col1, col2 = st.columns([2.5, 1])
        with col1:
            default_tickers = ["^TNX", "^IRX"]
            ticker = st.text_input("Ticker", value=default_tickers[i] if i < len(default_tickers) else "", key=f"ticker_{i}", help="Use Yahoo Finance ticker symbols, e.g. AAPL, ^VIX, EURUSD=X, ^TNX")
        with col2:
            default_weights = [1.0, -1.0]
            weight = st.number_input("Weight", value=default_weights[i] if i < len(default_weights) else 0.0, step=0.1, key=f"weight_{i}")
        
        if ticker.strip():
            ticker_weights[ticker.strip()] = weight

# Benchmark (optional) - built from its own weighted Yahoo Finance components, same pattern as
# the dependent variable, so it can be a single index or a custom multi-leg combination too.
with st.sidebar.expander("Benchmark (Optional)", expanded=False):
    enable_benchmark = st.checkbox("Enable custom benchmark", value=False, key="enable_benchmark")

    benchmark_weights = {}
    if enable_benchmark:
        num_benchmark_components = st.number_input("Number of benchmark components", min_value=1, max_value=10, value=1, key="num_benchmark_components")
        for i in range(num_benchmark_components):
            col1, col2 = st.columns([2.5, 1])
            with col1:
                b_ticker = st.text_input("Ticker", value="", key=f"bench_ticker_{i}", help="Use Yahoo Finance ticker symbols, e.g. ^GSPC, SPY, ^TNX")
            with col2:
                b_weight = st.number_input("Weight", value=1.0, step=0.1, key=f"bench_weight_{i}")

            if b_ticker.strip():
                benchmark_weights[b_ticker.strip()] = b_weight

# Independent Indicators
with st.sidebar.expander("Independent Indicator Conditions", expanded=True):
    num_indicators = st.number_input("Number of indicators", min_value=0, max_value=10, value=1)
    
    indicators = []
    economic_tickers = []  # Track which tickers are economic data
    
    for i in range(num_indicators):
        st.markdown(f"**Indicator {i+1}:**")
        
        # Ticker input with data type selection
        col1, col2 = st.columns([2, 1])
        with col1:
            indicator_ticker = st.text_input("Yahoo Finance Ticker", value="", key=f"ind_ticker_{i}", placeholder="e.g., ^VIX, AAPL, ^TNX")
        with col2:
            data_type = st.selectbox("Data Type", ["Market Data", "Economic Data"], key=f"data_type_{i}", 
                                   help="Economic Data: Forward fills missing values for non-release days (source this from elsewhere if not on Yahoo Finance)")
        
        # Track economic data tickers
        if indicator_ticker.strip() and data_type == "Economic Data":
            economic_tickers.append(indicator_ticker.strip())
        
        if indicator_ticker.strip():
            condition_type = st.selectbox("Condition Type", ["Level", "Rolling Return", "Cumulative Sum"], key=f"ind_type_{i}")
            
            if condition_type == "Level":
                col1, col2 = st.columns(2)
                with col1:
                    threshold = st.number_input("Threshold", value=18.0, key=f"ind_threshold_{i}")
                with col2:
                    above_below = st.selectbox("Above/Below", ["Above", "Below"], key=f"ind_above_{i}")
                
                indicators.append({
                    'ticker': indicator_ticker.strip(),
                    'type': 'level',
                    'threshold': threshold,
                    'above': above_below == "Above",
                    'data_type': data_type
                })
            
            elif condition_type == "Rolling Return":
                col1, col2 = st.columns(2)
                with col1:
                    return_pct = st.number_input("Return %", value=-2.0, key=f"ind_return_{i}")
                with col2:
                    days = st.number_input("Days", min_value=1, max_value=252, value=3, key=f"ind_days_{i}")
                
                above_below_ret = st.selectbox("Above/Below", ["Above", "Below"], key=f"ind_above_ret_{i}")
                
                indicators.append({
                    'ticker': indicator_ticker.strip(),
                    'type': 'rolling_return',
                    'return_pct': return_pct,
                    'days': days,
                    'above': above_below_ret == "Above",
                    'data_type': data_type
                })
            
            else:  # Cumulative Sum
                col1, col2 = st.columns(2)
                with col1:
                    threshold = st.number_input("Threshold", value=1000.0, key=f"ind_cumsum_threshold_{i}")
                with col2:
                    days = st.number_input("Days", min_value=1, max_value=252, value=30, key=f"ind_cumsum_days_{i}")
                
                above_below_cumsum = st.selectbox("Above/Below", ["Above", "Below"], key=f"ind_above_cumsum_{i}")
                
                indicators.append({
                    'ticker': indicator_ticker.strip(),
                    'type': 'cumulative_sum',
                    'threshold': threshold,
                    'days': days,
                    'above': above_below_cumsum == "Above",
                    'data_type': data_type
                })

# Forward Return Horizons (Original Style)
with st.sidebar.expander("Forward Return Horizons", expanded=True):
    num_horizons = st.number_input("Number of horizons", min_value=1, max_value=10, value=3)
    
    horizons = []
    for i in range(num_horizons):
        default_horizons = [5, 10, 30]
        horizon = st.number_input(
            f"Horizon {i+1} (days)", 
            min_value=1, 
            max_value=252, 
            value=default_horizons[i] if i < len(default_horizons) else 1, 
            key=f"horizon_{i}"
        )
        horizons.append(horizon)

    forward_change_type = st.radio("Change Type", ["Nominal", "Percentage"], horizontal=True, key="forward_change_type",
                                    help="Percentage is unreliable/explosive if the dependent variable (or benchmark) crosses zero - same caveat as the Seasonality toggle.")
    forward_change_type_code = 'pct' if forward_change_type == "Percentage" else 'nominal'

# Expected Direction for Win Rate Analysis
with st.sidebar.expander("Win Rate Analysis", expanded=True):
    expected_direction = st.selectbox(
        "Expected Direction After Conditions Trigger",
        ["Increase", "Decrease"],
        index=0,
        help="Choose whether you expect the dependent variable to increase or decrease after matching conditions are met"
    )
    
    st.markdown(f"**Current Setting:** Expecting dependent variable to **{expected_direction.lower()}** after conditions trigger")

# Cluster-Free Zone Configuration
with st.sidebar.expander("Cluster-Free Zone", expanded=True):
    cluster_free_days = st.number_input(
        "Cluster-Free Zone (Days)", 
        min_value=0, 
        max_value=252, 
        value=0,
        help="Number of days to wait after a signal before allowing another signal. 0 = no clustering filter (current behavior)"
    )
    
    st.markdown(f"**Current Setting:** {cluster_free_days} day{'s' if cluster_free_days != 1 else ''} cooldown period")
    if cluster_free_days == 0:
        st.info("⚠️ No clustering filter applied - all matching dates included")
    else:
        st.info(f"🔒 {cluster_free_days}-day cooldown after each signal")

# Calculate Button
calculate_button = st.sidebar.button("Calculate & Plot", type="primary", use_container_width=True)

# st.button() only returns True on the exact run it's clicked - any later widget interaction
# (e.g. the Monthly/Quarterly seasonality toggle) reruns the script with it back to False, which
# would otherwise drop back to the welcome screen. Persist the "calculated" state instead.
if calculate_button:
    st.session_state['calculated'] = True

# Main Content
if st.session_state.get('calculated', False):
    if not ticker_weights or not any(w != 0 for w in ticker_weights.values()):
        st.error("Please add at least one ticker with non-zero weight")
    else:
        with st.spinner("Fetching Yahoo Finance data and calculating..."):
            try:
                # Get all tickers
                active_tickers = [t for t, w in ticker_weights.items() if t and w != 0]
                indicator_tickers = [ind['ticker'] for ind in indicators if ind['ticker']]
                active_benchmark_tickers = [t for t, w in benchmark_weights.items() if t and w != 0] if enable_benchmark else []
                all_tickers = list(dict.fromkeys(active_tickers + indicator_tickers + active_benchmark_tickers))

                # Fetch data
                st.info(f"Fetching data for {len(all_tickers)} tickers from {start_date} to {end_date}")
                price_data = fetch_yahoo_data(all_tickers, start_date, end_date)

                if not price_data.empty:
                    st.success(f"Retrieved {len(price_data)} trading days of data")

                    # Process economic data with forward fill
                    if economic_tickers:
                        st.info(f"Forward filling economic data for: {', '.join(economic_tickers)}")
                        price_data = process_economic_data(price_data, economic_tickers)

                    # Calculate dependent variable
                    dependent_var, result_data = calculate_dependent_variable(price_data, ticker_weights)

                    # Calculate benchmark (optional) - same weighted-combination logic as the
                    # dependent variable, just against benchmark_weights instead.
                    benchmark_var = pd.Series(dtype=float)
                    if enable_benchmark and active_benchmark_tickers:
                        benchmark_var, _ = calculate_dependent_variable(price_data, benchmark_weights)
                        if benchmark_var.empty:
                            st.warning("⚠️ Benchmark enabled but no data could be computed for it - check the benchmark ticker(s).")

                    # Flag it when components don't all start on the same date, since the combined
                    # series can only start from whichever component starts LATEST.
                    active_dep_tickers = {t: w for t, w in ticker_weights.items() if t and w != 0}
                    ticker_start_dates = {t: price_data[t].dropna().index.min() for t in active_dep_tickers if t in price_data.columns and price_data[t].notna().any()}
                    if len(set(ticker_start_dates.values())) > 1:
                        availability_str = " | ".join(f"{t}: from {d.date()}" for t, d in sorted(ticker_start_dates.items(), key=lambda kv: kv[1]))
                        latest_start = max(ticker_start_dates.values())
                        st.warning(f"⚠️ Component tickers have different data start dates — {availability_str}. "
                                   f"The dependent variable requires all components, so it only starts from {latest_start.date()}.")

                    if len(dependent_var) > 0:
                        # Evaluate conditions (now returns cumulative sum columns too)
                        matching_mask, condition_results, individual_conditions, rolling_return_columns, cumulative_sum_columns = evaluate_indicator_conditions(price_data, indicators)
                        
                        # Apply cluster-free filter
                        original_matching_mask = matching_mask.reindex(dependent_var.index, fill_value=False)
                        filtered_matching_mask, removed_signals = apply_cluster_free_filter(original_matching_mask, cluster_free_days)
                        
                        # Get both sets of matching dates
                        all_matching_dates = dependent_var.index[original_matching_mask]  # All original signals
                        cluster_free_dates = dependent_var.index[filtered_matching_mask]  # Filtered signals

                        # Track clustering statistics
                        original_signal_count = original_matching_mask.sum()
                        filtered_signal_count = filtered_matching_mask.sum()
                        removed_signal_count = removed_signals.sum()
                        
                        # Forward-return change type (Nominal/Percentage) - configured in the
                        # sidebar since it's read here, before any of this is rendered.
                        if forward_change_type_code == 'pct' and (dependent_var <= 0).any():
                            st.warning("⚠️ The dependent variable crosses zero (or goes negative) over this range - "
                                       "% change is unreliable/explosive here (division by a near-zero base) for the "
                                       "Forward Return Horizons. Nominal change is safer for spread-type dependent variables.")
                        if forward_change_type_code == 'pct' and not benchmark_var.empty and (benchmark_var <= 0).any():
                            st.warning("⚠️ The benchmark crosses zero (or goes negative) over this range - % change is "
                                       "unreliable/explosive for its forward returns and beta too.")

                        change_metric_name = "Nominal" if forward_change_type_code == 'nominal' else "Pct"
                        change_value_suffix = '' if forward_change_type_code == 'nominal' else '%'
                        change_display_precision = 3 if forward_change_type_code == 'pct' else 4

                        # Calculate forward returns for ALL dates (for CSV)
                        forward_returns_all, forward_return_suffix = calculate_forward_returns_all_dates(dependent_var, horizons, forward_change_type_code)

                        # Calculate forward returns for BOTH approaches
                        forward_returns_cluster_free = calculate_forward_returns_matching_only(dependent_var, cluster_free_dates, horizons, expected_direction, forward_change_type_code)
                        forward_returns_all_signals = calculate_forward_returns_matching_only(dependent_var, all_matching_dates, horizons, expected_direction, forward_change_type_code)

                        # Weighted OHLC dependent variable - needed for the Avg Range stat below,
                        # and reused later for the candlestick chart (cached, so no duplicate fetch).
                        with st.spinner("Fetching OHLC data for range calculations..."):
                            ohlc_data = fetch_yahoo_ohlc(active_tickers, start_date, end_date)
                        dep_ohlc = calculate_dependent_variable_ohlc(ohlc_data, ticker_weights)
                        dep_ohlc = dep_ohlc.reindex(dependent_var.index)

                        forward_ranges_cluster_free = calculate_forward_range_matching_only(dep_ohlc, cluster_free_dates, horizons, forward_change_type_code)
                        forward_ranges_all_signals = calculate_forward_range_matching_only(dep_ohlc, all_matching_dates, horizons, forward_change_type_code)

                        # Same forward-return calculation, applied to the benchmark instead, at
                        # the same signal dates/horizons - lets the benchmark's forward returns
                        # be shown alongside the dependent variable's.
                        benchmark_forward_returns_cluster_free = {}
                        benchmark_forward_returns_all_signals = {}
                        beta, beta_intercept, beta_r_squared = None, None, None
                        beta_chg = pd.DataFrame()
                        if not benchmark_var.empty:
                            benchmark_forward_returns_cluster_free = calculate_forward_returns_matching_only(benchmark_var, cluster_free_dates, horizons, expected_direction, forward_change_type_code)
                            benchmark_forward_returns_all_signals = calculate_forward_returns_matching_only(benchmark_var, all_matching_dates, horizons, expected_direction, forward_change_type_code)
                            beta_chg, beta, beta_intercept, beta_r_squared = get_beta_regression_data(dependent_var, benchmark_var, forward_change_type_code)

                        # Create comprehensive dataset (using filtered matching mask)
                        comprehensive_df = create_comprehensive_dataframe(
                            price_data, ticker_weights, indicators,
                            dependent_var, filtered_matching_mask, individual_conditions,
                            rolling_return_columns, cumulative_sum_columns, forward_returns_all, horizons,
                            forward_return_suffix, benchmark_var
                        )
                                                # Enhanced Metrics with Clustering Info
                        zscores = calculate_zscores(dependent_var)
                        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
                        with col1:
                            st.metric("Total Data Points", f"{len(dependent_var):,}")
                        with col2:
                            if cluster_free_days > 0:
                                st.metric("Original Signals", f"{original_signal_count:,}",
                                         help="Signals before cluster-free filter")
                            else:
                                st.metric("Matching Dates", f"{len(all_matching_dates):,}")
                        with col3:
                            if cluster_free_days > 0:
                                st.metric("Filtered Signals", f"{filtered_signal_count:,}",
                                         delta=f"-{removed_signal_count}" if removed_signal_count > 0 else None,
                                         help=f"Signals after {cluster_free_days}-day cluster-free filter")
                            else:
                                match_rate = len(all_matching_dates) / len(dependent_var) * 100 if len(dependent_var) > 0 else 0
                                st.metric("Hit Rate", f"{match_rate:.1f}%")
                        with col4:
                            if cluster_free_days > 0:
                                filter_rate = (removed_signal_count / original_signal_count * 100) if original_signal_count > 0 else 0
                                st.metric("Filtered Out", f"{filter_rate:.1f}%",
                                         help="Percentage of original signals removed by clustering filter")
                            # else: nothing clustering-related to show here when the filter is off -
                            # col5's "Current Value" already covers the dependent variable's latest reading.
                        with col5:
                            st.metric("Current Value", f"{dependent_var.iloc[-1]:.2f}")
                        for col, label in zip([col6, col7, col8], ["1M Z-Score", "3M Z-Score", "1Y Z-Score"]):
                            with col:
                                latest_z = zscores[label].iloc[-1]
                                st.metric(label, f"{latest_z:.2f}" if pd.notna(latest_z) else "N/A",
                                         help=f"({label.split()[0]} lookback) - not enough history yet if N/A")
                        
                        # Show clustering filter information
                        if cluster_free_days > 0:
                            st.info(f"🔒 **Cluster-Free Filter Applied:** {cluster_free_days} days | "
                                    f"Removed {removed_signal_count:,} signals ({(removed_signal_count/original_signal_count*100):.1f}% of original) | "
                                    f"Using {filtered_signal_count:,} signals for cluster-free analysis")
                            
                            if removed_signal_count > 0:
                                with st.expander("View Removed Signals", expanded=False):
                                    removed_dates = dependent_var.index[removed_signals]
                                    if len(removed_dates) > 0:
                                        removed_df = pd.DataFrame({
                                            'Removed_Date': removed_dates,
                                            'Dependent_Variable_Value': dependent_var.loc[removed_dates].values
                                        })
                                        st.dataframe(removed_df, use_container_width=True)
                        else:
                            st.info("ℹ️ **No Cluster-Free Filter:** All matching signals included in analysis")
                        
                        # Show economic data processing info
                        if economic_tickers:
                            st.info(f"📊 Economic data tickers processed with forward fill: {', '.join(economic_tickers)}")

                        # Benchmark: overall beta (single figure, whole selected date range)
                        if not benchmark_var.empty:
                            st.subheader("Benchmark Analysis")
                            beta_col, info_col = st.columns([1, 3])
                            with beta_col:
                                st.metric("Beta vs Benchmark", f"{beta:.3f}" if beta is not None else "N/A")
                            with info_col:
                                basis = "daily nominal changes" if forward_change_type_code == 'nominal' else "daily % returns"
                                st.caption(f"Slope of the dependent variable's {basis} regressed on the benchmark's, over the "
                                           f"full overlapping date range ({change_metric_name} basis, matching the Change Type "
                                           f"toggle above). Forward returns for the benchmark are shown alongside the "
                                           f"dependent variable's below.")

                            if not beta_chg.empty and beta is not None:
                                fig_beta_reg = go.Figure()
                                fig_beta_reg.add_trace(go.Scatter(
                                    x=beta_chg['Bench'], y=beta_chg['Dep'], mode='markers', name='Daily Changes',
                                    marker=dict(color='#636EFA', size=5, opacity=0.5),
                                    hovertemplate=(f"Benchmark: %{{x:.{change_display_precision}f}}{change_value_suffix}<br>"
                                                   f"Dependent: %{{y:.{change_display_precision}f}}{change_value_suffix}<extra></extra>"),
                                ))
                                x_range = np.linspace(beta_chg['Bench'].min(), beta_chg['Bench'].max(), 50)
                                y_fit = beta * x_range + beta_intercept
                                fig_beta_reg.add_trace(go.Scatter(
                                    x=x_range, y=y_fit, mode='lines', name=f'Fit (β={beta:.3f})',
                                    line=dict(color='#ef5350', width=2),
                                ))
                                r2_text = f", R²={beta_r_squared:.3f}" if beta_r_squared is not None else ""
                                fig_beta_reg.update_layout(
                                    title=dict(text=f"Dependent Variable vs Benchmark — Daily {change_metric_name} Changes (β={beta:.3f}{r2_text})",
                                               x=0.5, xanchor="center"),
                                    template="plotly_dark", height=450,
                                    xaxis_title=f"Benchmark Daily {change_metric_name} Change{change_value_suffix}",
                                    yaxis_title=f"Dependent Variable Daily {change_metric_name} Change{change_value_suffix}",
                                )
                                st.plotly_chart(fig_beta_reg, use_container_width=True)

                        # Analysis 1: Cluster-Free Forward Return Analysis
                        if forward_returns_cluster_free and any(not df.empty for df in forward_returns_cluster_free.values()):
                            st.subheader("Cluster-Free Forward Return Analysis")
                            st.markdown(f"**Method:** {cluster_free_days}-day cooldown after each signal | **Expected Direction:** {expected_direction} | **Change Type:** {forward_change_type}")

                            avg_col, median_col = f'Avg {change_metric_name}', f'Median {change_metric_name}'
                            range_col = f'Avg Range ({change_metric_name})'

                            # Summary statistics with Win Rate for Cluster-Free
                            summary_data_cf = []
                            for horizon in horizons:
                                horizon_key = f'{horizon}D'
                                if horizon_key in forward_returns_cluster_free and not forward_returns_cluster_free[horizon_key].empty:
                                    df_fwd = forward_returns_cluster_free[horizon_key]

                                    # Calculate Win Rate and standard deviation
                                    win_rate = df_fwd['Hit'].mean() * 100 if len(df_fwd) > 0 else 0
                                    std_dev = df_fwd['Change'].std()
                                    range_series = forward_ranges_cluster_free.get(horizon_key, pd.Series(dtype=float))

                                    summary_data_cf.append({
                                        'Horizon': f'{horizon}D',
                                        'Sample Size': len(df_fwd),
                                        avg_col: df_fwd['Change'].mean(),
                                        median_col: df_fwd['Change'].median(),
                                        'Std Dev': std_dev,
                                        'Win Rate': win_rate,
                                        range_col: range_series.mean() if len(range_series) else np.nan
                                    })

                            if summary_data_cf:
                                # Display metrics with Win Rate for Cluster-Free
                                horizon_cols = st.columns(len(summary_data_cf))
                                for i, row in enumerate(summary_data_cf):
                                    with horizon_cols[i]:
                                        st.metric(f"{row['Horizon']} Sample", f"{int(row['Sample Size']):,}")
                                        st.metric(avg_col, f"{row[avg_col]:.{change_display_precision}f}{change_value_suffix}")
                                        st.metric(median_col, f"{row[median_col]:.{change_display_precision}f}{change_value_suffix}")
                                        st.metric("Win Rate", f"{row['Win Rate']:.1f}%",
                                                help=f"% of times dependent variable moved in expected direction ({expected_direction.lower()})")
                                        st.metric(range_col, f"{row[range_col]:.{change_display_precision}f}{change_value_suffix}" if pd.notna(row[range_col]) else "N/A",
                                                help="Average True Range (max of High-Low, |High-PrevClose|, |Low-PrevClose|) over the horizon's forward trading days, averaged across matching signals.")

                                # Summary table for Cluster-Free
                                st.markdown("**Cluster-Free Summary Statistics:**")
                                summary_df_cf = pd.DataFrame(summary_data_cf)
                                st.dataframe(summary_df_cf.round({avg_col: change_display_precision, median_col: change_display_precision, 'Win Rate': 1, range_col: change_display_precision}),
                                           use_container_width=True, hide_index=True)

                                # Distribution plots for Cluster-Free
                                fig_dist_cf = make_subplots(
                                    rows=1, cols=len(summary_data_cf),
                                    subplot_titles=[f'{row["Horizon"]} Cluster-Free (Win Rate: {row["Win Rate"]:.1f}%)' for row in summary_data_cf]
                                )

                                colors = ['#ff6692', '#ab63fa', '#ffa15a', '#19d3f3', '#ff97ff', '#fecb52']

                                col_idx = 1
                                for i, horizon in enumerate(horizons):
                                    horizon_key = f'{horizon}D'
                                    if horizon_key in forward_returns_cluster_free and not forward_returns_cluster_free[horizon_key].empty:
                                        df_fwd = forward_returns_cluster_free[horizon_key]
                                        color = colors[i % len(colors)]

                                        # Add histogram - manually binned (rather than go.Histogram's auto-binning) so the
                                        # hover can show an explicit "X to Y<unit>" range instead of Plotly's default
                                        # unlabeled "(15 - 20, 4)" tuple, which doesn't say whether that's nominal or %.
                                        counts, bin_edges = np.histogram(df_fwd['Change'].dropna(), bins=20)
                                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                        bin_widths = bin_edges[1:] - bin_edges[:-1]
                                        fig_dist_cf.add_trace(go.Bar(
                                            x=bin_centers, y=counts, width=bin_widths, marker_color=color, opacity=0.7,
                                            customdata=np.stack([bin_edges[:-1], bin_edges[1:]], axis=-1),
                                            hovertemplate=(f"Range: %{{customdata[0]:.{change_display_precision}f}}{change_value_suffix} to "
                                                           f"%{{customdata[1]:.{change_display_precision}f}}{change_value_suffix}<br>"
                                                           f"Count: %{{y}}<extra></extra>"),
                                        ), row=1, col=col_idx)

                                        # Calculate statistics
                                        median_val = df_fwd['Change'].median()
                                        std_val = df_fwd['Change'].std()

                                        # Add median line
                                        fig_dist_cf.add_vline(x=median_val, line_dash="dash", line_color="blue", line_width=2, row=1, col=col_idx)

                                        # Add +1 std deviation line
                                        fig_dist_cf.add_vline(x=median_val + std_val, line_dash="dot", line_color="red", line_width=2, row=1, col=col_idx)

                                        # Add -1 std deviation line
                                        fig_dist_cf.add_vline(x=median_val - std_val, line_dash="dot", line_color="red", line_width=2, row=1, col=col_idx)

                                        col_idx += 1

                                fig_dist_cf.update_layout(title=dict(text="Cluster-Free Forward Return Distributions", x=0.5, xanchor="center"), template="plotly_dark", height=400, showlegend=False)
                                st.plotly_chart(fig_dist_cf, use_container_width=True)

                            # Benchmark forward returns, same signal dates/horizons
                            if benchmark_forward_returns_cluster_free and any(not df.empty for df in benchmark_forward_returns_cluster_free.values()):
                                st.markdown("**Benchmark Forward Returns (Cluster-Free):**")
                                bench_summary_cf = []
                                for horizon in horizons:
                                    horizon_key = f'{horizon}D'
                                    if horizon_key in benchmark_forward_returns_cluster_free and not benchmark_forward_returns_cluster_free[horizon_key].empty:
                                        df_b = benchmark_forward_returns_cluster_free[horizon_key]
                                        bench_summary_cf.append({
                                            'Horizon': f'{horizon}D', 'Sample Size': len(df_b),
                                            avg_col: df_b['Change'].mean(), median_col: df_b['Change'].median(),
                                            'Std Dev': df_b['Change'].std(), 'Win Rate': df_b['Hit'].mean() * 100 if len(df_b) > 0 else 0,
                                        })
                                if bench_summary_cf:
                                    bench_df_cf = pd.DataFrame(bench_summary_cf)
                                    st.dataframe(bench_df_cf.round({avg_col: change_display_precision, median_col: change_display_precision, 'Win Rate': 1}),
                                                 use_container_width=True, hide_index=True)

                                    fig_bench_cf = go.Figure()
                                    dep_avgs = [round(row[avg_col], change_display_precision) for row in summary_data_cf]
                                    bench_avgs = [round(row[avg_col], change_display_precision) for row in bench_summary_cf]
                                    fig_bench_cf.add_trace(go.Bar(x=[row['Horizon'] for row in summary_data_cf], y=dep_avgs, name='Dependent Variable', marker_color='#636EFA',
                                                                   text=dep_avgs, texttemplate=f'%{{text:.{change_display_precision}f}}{change_value_suffix}'))
                                    fig_bench_cf.add_trace(go.Bar(x=[row['Horizon'] for row in bench_summary_cf], y=bench_avgs, name='Benchmark', marker_color='#FFA500',
                                                                   text=bench_avgs, texttemplate=f'%{{text:.{change_display_precision}f}}{change_value_suffix}'))
                                    fig_bench_cf.update_layout(title=dict(text=f"{avg_col} Forward Return — Dependent vs Benchmark (Cluster-Free)", x=0.5, xanchor="center"),
                                                                template="plotly_dark", height=350, barmode='group', yaxis_title=avg_col)
                                    st.plotly_chart(fig_bench_cf, use_container_width=True)

                        # Analysis 2: All Signals Forward Return Analysis
                        if forward_returns_all_signals and any(not df.empty for df in forward_returns_all_signals.values()):
                            st.subheader("All Signals Forward Return Analysis")
                            st.markdown(f"**Method:** All original signals (no clustering filter) | **Expected Direction:** {expected_direction} | **Change Type:** {forward_change_type}")

                            avg_col, median_col = f'Avg {change_metric_name}', f'Median {change_metric_name}'
                            range_col = f'Avg Range ({change_metric_name})'

                            # Summary statistics with Win Rate for All Signals
                            summary_data_all = []
                            for horizon in horizons:
                                horizon_key = f'{horizon}D'
                                if horizon_key in forward_returns_all_signals and not forward_returns_all_signals[horizon_key].empty:
                                    df_fwd = forward_returns_all_signals[horizon_key]

                                    # Calculate Win Rate and standard deviation
                                    win_rate = df_fwd['Hit'].mean() * 100 if len(df_fwd) > 0 else 0
                                    std_dev = df_fwd['Change'].std()
                                    range_series = forward_ranges_all_signals.get(horizon_key, pd.Series(dtype=float))

                                    summary_data_all.append({
                                        'Horizon': f'{horizon}D',
                                        'Sample Size': len(df_fwd),
                                        avg_col: df_fwd['Change'].mean(),
                                        median_col: df_fwd['Change'].median(),
                                        'Std Dev': std_dev,
                                        'Win Rate': win_rate,
                                        range_col: range_series.mean() if len(range_series) else np.nan
                                    })

                            if summary_data_all:
                                # Display metrics with Win Rate for All Signals
                                horizon_cols = st.columns(len(summary_data_all))
                                for i, row in enumerate(summary_data_all):
                                    with horizon_cols[i]:
                                        st.metric(f"{row['Horizon']} Sample", f"{int(row['Sample Size']):,}")
                                        st.metric(avg_col, f"{row[avg_col]:.{change_display_precision}f}{change_value_suffix}")
                                        st.metric(median_col, f"{row[median_col]:.{change_display_precision}f}{change_value_suffix}")
                                        st.metric("Win Rate", f"{row['Win Rate']:.1f}%",
                                                help=f"% of times dependent variable moved in expected direction ({expected_direction.lower()})")
                                        st.metric(range_col, f"{row[range_col]:.{change_display_precision}f}{change_value_suffix}" if pd.notna(row[range_col]) else "N/A",
                                                help="Average True Range (max of High-Low, |High-PrevClose|, |Low-PrevClose|) over the horizon's forward trading days, averaged across matching signals.")

                                # Summary table for All Signals
                                st.markdown("**All Signals Summary Statistics:**")
                                summary_df_all = pd.DataFrame(summary_data_all)
                                st.dataframe(summary_df_all.round({avg_col: change_display_precision, median_col: change_display_precision, 'Win Rate': 1, range_col: change_display_precision}),
                                           use_container_width=True, hide_index=True)

                                # Distribution plots for All Signals
                                fig_dist_all = make_subplots(
                                    rows=1, cols=len(summary_data_all),
                                    subplot_titles=[f'{row["Horizon"]} All Signals (Win Rate: {row["Win Rate"]:.1f}%)' for row in summary_data_all]
                                )

                                colors = ['#ff6692', '#ab63fa', '#ffa15a', '#19d3f3', '#ff97ff', '#fecb52']

                                col_idx = 1
                                for i, horizon in enumerate(horizons):
                                    horizon_key = f'{horizon}D'
                                    if horizon_key in forward_returns_all_signals and not forward_returns_all_signals[horizon_key].empty:
                                        df_fwd = forward_returns_all_signals[horizon_key]
                                        color = colors[i % len(colors)]

                                        # Add histogram - manually binned, see Cluster-Free section above for why.
                                        counts, bin_edges = np.histogram(df_fwd['Change'].dropna(), bins=20)
                                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                        bin_widths = bin_edges[1:] - bin_edges[:-1]
                                        fig_dist_all.add_trace(go.Bar(
                                            x=bin_centers, y=counts, width=bin_widths, marker_color=color, opacity=0.7,
                                            customdata=np.stack([bin_edges[:-1], bin_edges[1:]], axis=-1),
                                            hovertemplate=(f"Range: %{{customdata[0]:.{change_display_precision}f}}{change_value_suffix} to "
                                                           f"%{{customdata[1]:.{change_display_precision}f}}{change_value_suffix}<br>"
                                                           f"Count: %{{y}}<extra></extra>"),
                                        ), row=1, col=col_idx)

                                        # Calculate statistics
                                        median_val = df_fwd['Change'].median()
                                        std_val = df_fwd['Change'].std()

                                        # Add median line
                                        fig_dist_all.add_vline(x=median_val, line_dash="dash", line_color="blue", line_width=2, row=1, col=col_idx)

                                        # Add +1 std deviation line
                                        fig_dist_all.add_vline(x=median_val + std_val, line_dash="dot", line_color="red", line_width=2, row=1, col=col_idx)

                                        # Add -1 std deviation line
                                        fig_dist_all.add_vline(x=median_val - std_val, line_dash="dot", line_color="red", line_width=2, row=1, col=col_idx)

                                        col_idx += 1

                                fig_dist_all.update_layout(title=dict(text="All Signals Forward Return Distributions", x=0.5, xanchor="center"), template="plotly_dark", height=400, showlegend=False)
                                st.plotly_chart(fig_dist_all, use_container_width=True)

                            # Benchmark forward returns, same signal dates/horizons
                            if benchmark_forward_returns_all_signals and any(not df.empty for df in benchmark_forward_returns_all_signals.values()):
                                st.markdown("**Benchmark Forward Returns (All Signals):**")
                                bench_summary_all = []
                                for horizon in horizons:
                                    horizon_key = f'{horizon}D'
                                    if horizon_key in benchmark_forward_returns_all_signals and not benchmark_forward_returns_all_signals[horizon_key].empty:
                                        df_b = benchmark_forward_returns_all_signals[horizon_key]
                                        bench_summary_all.append({
                                            'Horizon': f'{horizon}D', 'Sample Size': len(df_b),
                                            avg_col: df_b['Change'].mean(), median_col: df_b['Change'].median(),
                                            'Std Dev': df_b['Change'].std(), 'Win Rate': df_b['Hit'].mean() * 100 if len(df_b) > 0 else 0,
                                        })
                                if bench_summary_all:
                                    bench_df_all = pd.DataFrame(bench_summary_all)
                                    st.dataframe(bench_df_all.round({avg_col: change_display_precision, median_col: change_display_precision, 'Win Rate': 1}),
                                                 use_container_width=True, hide_index=True)

                                    fig_bench_all = go.Figure()
                                    dep_avgs = [round(row[avg_col], change_display_precision) for row in summary_data_all]
                                    bench_avgs = [round(row[avg_col], change_display_precision) for row in bench_summary_all]
                                    fig_bench_all.add_trace(go.Bar(x=[row['Horizon'] for row in summary_data_all], y=dep_avgs, name='Dependent Variable', marker_color='#636EFA',
                                                                    text=dep_avgs, texttemplate=f'%{{text:.{change_display_precision}f}}{change_value_suffix}'))
                                    fig_bench_all.add_trace(go.Bar(x=[row['Horizon'] for row in bench_summary_all], y=bench_avgs, name='Benchmark', marker_color='#FFA500',
                                                                    text=bench_avgs, texttemplate=f'%{{text:.{change_display_precision}f}}{change_value_suffix}'))
                                    fig_bench_all.update_layout(title=dict(text=f"{avg_col} Forward Return — Dependent vs Benchmark (All Signals)", x=0.5, xanchor="center"),
                                                                 template="plotly_dark", height=350, barmode='group', yaxis_title=avg_col)
                                    st.plotly_chart(fig_bench_all, use_container_width=True)

                        # Seasonality
                        st.subheader("Seasonality")
                        season_col1, season_col2 = st.columns(2)
                        with season_col1:
                            seasonality_freq = st.radio("Timeframe", ["Monthly", "Quarterly"], horizontal=True, key="seasonality_freq")
                        with season_col2:
                            seasonality_change_type = st.radio("Change Type", ["Nominal", "Percentage"], horizontal=True, key="seasonality_change_type")
                        freq_code = 'M' if seasonality_freq == "Monthly" else 'Q'
                        change_type_code = 'pct' if seasonality_change_type == "Percentage" else 'nominal'
                        period_axis_title = "Month" if freq_code == 'M' else "Quarter"
                        value_suffix = '%' if change_type_code == 'pct' else ''
                        value_label = "% Change" if change_type_code == 'pct' else "Nominal Change"

                        if change_type_code == 'pct' and (dependent_var <= 0).any():
                            st.warning("⚠️ The dependent variable crosses zero (or goes negative) over this range - "
                                       "% change is unreliable/explosive here (division by a near-zero base). "
                                       "Nominal change is safer for spread-type dependent variables.")

                        returns_df, period_labels = compute_seasonality(dependent_var, freq_code, change_type_code)

                        if not returns_df.empty:
                            avg_change = returns_df.groupby('Period')['Change'].mean().reindex(range(1, len(period_labels) + 1))
                            std_change = returns_df.groupby('Period')['Change'].std().reindex(range(1, len(period_labels) + 1))

                            fig_season_bar = go.Figure(go.Bar(
                                x=period_labels, y=avg_change.values,
                                error_y=dict(type='data', array=std_change.values, visible=True),
                                marker_color=['#26a69a' if v >= 0 else '#ef5350' for v in avg_change.fillna(0).values],
                                text=avg_change.round(4), texttemplate='%{text}' + value_suffix, textposition='outside'
                            ))
                            fig_season_bar.update_layout(title=dict(text=f"Average {seasonality_freq} {value_label} (± 1 Std Dev)", x=0.5, xanchor="center"), template="plotly_dark",
                                                          height=400, xaxis_title=period_axis_title, yaxis_title=f"Average {value_label}")
                            fig_season_bar.update_yaxes(ticksuffix=value_suffix)
                            st.plotly_chart(fig_season_bar, use_container_width=True)

                            pivot = returns_df.pivot_table(index='Year', columns='Period', values='Change', aggfunc='mean')
                            pivot.columns = [period_labels[c - 1] for c in pivot.columns]
                            avg_row = pd.DataFrame(pivot.mean(axis=0)).T
                            avg_row.index = ['Average']
                            pivot_display = pd.concat([avg_row, pivot.sort_index(ascending=False)])

                            fig_season_heat = go.Figure(go.Heatmap(
                                z=pivot_display.values, x=pivot_display.columns, y=pivot_display.index.astype(str),
                                text=np.round(pivot_display.values, 4), texttemplate="%{text}" + value_suffix,
                                colorscale='RdYlGn', zmid=0, colorbar=dict(title=value_label)
                            ))
                            fig_season_heat.update_layout(title=dict(text=f"{seasonality_freq} {value_label} Heatmap by Year", x=0.5, xanchor="center"), template="plotly_dark",
                                                           height=600, xaxis_title=period_axis_title, yaxis_title="Year", xaxis_side='top')
                            fig_season_heat.update_yaxes(autorange='reversed')
                            st.plotly_chart(fig_season_heat, use_container_width=True)
                        else:
                            st.info(f"Not enough history in the selected date range to compute {seasonality_freq.lower()} seasonality.")

                        # Time series plot
                        st.subheader("Dependent Variable with Signal Analysis")
                        fig = go.Figure()

                        # dep_ohlc was already fetched/computed above (for the Avg Range stat) -
                        # just drop the NaN rows here for a clean candlestick.
                        dep_ohlc_chart = dep_ohlc.dropna()

                        if not dep_ohlc_chart.empty:
                            fig.add_trace(go.Candlestick(
                                x=dep_ohlc_chart.index, open=dep_ohlc_chart['Open'], high=dep_ohlc_chart['High'],
                                low=dep_ohlc_chart['Low'], close=dep_ohlc_chart['Close'], name='Dependent Variable',
                                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                            ))
                            fig.update_layout(xaxis_rangeslider_visible=False)
                        else:
                            st.info("Candlestick unavailable for this configuration (missing OHLC data) - showing as a line instead.")
                            fig.add_trace(go.Scatter(x=dependent_var.index, y=dependent_var.values, mode='lines', name='Dependent Variable', line=dict(color='#636EFA', width=1.5)))

                        if len(cluster_free_dates) > 0:
                            cluster_free_values = dependent_var.loc[cluster_free_dates]
                            fig.add_trace(go.Scatter(x=cluster_free_values.index, y=cluster_free_values.values, mode='markers', name='Cluster-Free Signals', marker=dict(color='#FF6B6B', size=8)))
                        
                        if len(all_matching_dates) > 0:
                            all_matching_values = dependent_var.loc[all_matching_dates]
                            fig.add_trace(go.Scatter(x=all_matching_values.index, y=all_matching_values.values, mode='markers', name='All Original Signals', marker=dict(color='#00CC96', size=6, symbol='diamond')))
                        
                        # Add removed signals if clustering is active
                        if cluster_free_days > 0 and removed_signal_count > 0:
                            removed_dates = dependent_var.index[removed_signals]
                            removed_values = dependent_var.loc[removed_dates]
                            fig.add_trace(go.Scatter(x=removed_values.index, y=removed_values.values, mode='markers', name='Removed by Clustering', marker=dict(color='#FFA500', size=6, symbol='x')))
                        
                        fig.update_layout(title=dict(text="Dependent Variable Time Series with Dual Analysis", x=0.5, xanchor="center"), template="plotly_dark", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Dependent Variable Breakdown
                        st.subheader("Dependent Variable Breakdown")

                        fig_components = go.Figure()

                        component_cols = [col for col in result_data.columns if '×' in col]

                        if not component_cols:
                            # Fallback for single ticker case
                            component_cols = [col for col in result_data.columns if col != 'Dependent_Variable' and col in ticker_weights.keys()]

                        colors = ['#ff6692', '#ab63fa', '#ffa15a', '#19d3f3', '#ff97ff', '#fecb52']

                        # Add component lines
                        for i, col in enumerate(component_cols):
                            fig_components.add_trace(go.Scatter(
                                x=result_data.index, 
                                y=result_data[col], 
                                mode='lines', 
                                name=col, 
                                line=dict(color=colors[i % len(colors)], width=2)
                            ))

                        # Only add total line if it's different from components (i.e., multiple components)
                        if len(component_cols) > 1:
                            fig_components.add_trace(go.Scatter(
                                x=dependent_var.index, 
                                y=dependent_var.values, 
                                mode='lines', 
                                name='Total', 
                                line=dict(color='white', width=3)
                            ))
                        else:
                            # For single component, just show a note
                            st.info("Single component - the component line represents the total dependent variable")

                        fig_components.update_layout(title=dict(text="Weighted Components and Total", x=0.5, xanchor="center"), template="plotly_dark", height=400, showlegend=True)
                        st.plotly_chart(fig_components, use_container_width=True)
                        
                        # Complete Dataset Display
                        st.subheader("Complete Dataset")
                        st.markdown(f"**Dataset contains {len(comprehensive_df):,} rows with {len(comprehensive_df.columns)} columns**")
                        
                        # Show column summary
                        st.markdown("**Column Summary:**")
                        col_info = []
                        for col in comprehensive_df.columns:
                            if 'Forward_' in col:
                                non_null = comprehensive_df[col].notna().sum()
                                col_info.append(f"- {col}: {non_null:,} non-null values (ALL dates)")
                            elif 'Rolling_Return_pct' in col:
                                non_null = comprehensive_df[col].notna().sum()
                                avg_val = comprehensive_df[col].mean()
                                col_info.append(f"- {col}: {non_null:,} values (avg: {avg_val:.2f}%)")
                            elif 'Cumulative_Sum' in col:
                                non_null = comprehensive_df[col].notna().sum()
                                avg_val = comprehensive_df[col].mean()
                                col_info.append(f"- {col}: {non_null:,} values (avg: {avg_val:.2f})")
                            elif any(cond_col in col for cond_col in ['Above_', 'Below_', 'Return_', 'CumSum_']):
                                true_count = comprehensive_df[col].sum()
                                col_info.append(f"- {col}: {true_count:,} TRUE values")
                            else:
                                non_null = comprehensive_df[col].notna().sum()
                                col_info.append(f"- {col}: {non_null:,} non-null values")
                        
                        for info in col_info[:12]:
                            st.markdown(info)
                        if len(col_info) > 12:
                            st.markdown(f"... and {len(col_info)-12} more columns")
                        
                        # Display sample of data
                        st.markdown("**Data (Last 20 rows):**")
                        st.dataframe(comprehensive_df.tail(20).round(4), use_container_width=True, height=400)
                        
                        # Download Complete Dataset
                        st.markdown("### Download Complete Dataset")
                        csv_data = comprehensive_df.to_csv()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label=f"Download Complete Dataset ({len(comprehensive_df):,} rows)",
                                data=csv_data,
                                file_name=f"market_analysis_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        with col2:
                            st.metric("File Size", f"{len(csv_data)/1024/1024:.1f} MB")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

else:
    st.markdown("""
    ### Welcome to CIX Backtest Tool
    
    This tool allows you to:
    - **Create dependent variables** from weighted Yahoo Finance ticker combinations
    - **Set independent conditions** using level, rolling return, or cumulative sum indicators
    - **Apply cluster-free filtering** to prevent signal clustering bias
    - **Analyze forward returns** with dual analysis approach, in Nominal or Percentage terms
    - **Compare against an optional custom benchmark** - overall beta, plus the benchmark's own forward returns alongside the dependent variable's
    - **Download comprehensive datasets** for further analysis
    
    **Data Source: YFinance API**
    - Use YFinance tickers (e.g. `AAPL`, `^VIX`, `^TNX`, `EURUSD=X`, `CL=F`)
    - Daily close prices are pulled via `yfinance`
    - ECO series (CPI, GDP, etc.) unavailable through YFinance - the
      "Economic Data" forward-fill option is best used for gappy market series
    
    **Dual Forward Return Analysis**
    - **Cluster-Free Analysis**: X-day cooldown period after each signal
    - **All Signals Analysis**: Every original signal (no clustering filter)
    - Compare filtered vs unfiltered approaches side-by-side
    - 0 days = both analyses show same results (no filtering)
    - X days = see impact of clustering prevention
    
    **Original Horizons Interface**
    - Specify number of horizons (1-10)
    - Individual input fields for each horizon
    - Default: 5, 10, 30 days
    
    Configure your analysis in the sidebar and click **"Calculate & Plot"** to begin.
    """)
