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

def calculate_forward_returns_all_dates(dependent_var, horizons):
    """Calculate forward returns for ALL dates, not just matching dates"""
    forward_returns_all = {}
    
    for horizon in horizons:
        forward_returns_all[f'Forward_{horizon}D_Nominal'] = pd.Series(index=dependent_var.index, dtype=float)
        
        for i in range(len(dependent_var)):
            current_date = dependent_var.index[i]
            forward_idx = i + horizon
            
            if forward_idx < len(dependent_var):
                initial_value = dependent_var.iloc[i]
                forward_value = dependent_var.iloc[forward_idx]
                
                nominal_change = forward_value - initial_value
                
                forward_returns_all[f'Forward_{horizon}D_Nominal'].iloc[i] = nominal_change
    
    return forward_returns_all

def calculate_forward_returns_matching_only(dependent_var, matching_dates, horizons, expected_direction):
    """Calculate forward returns for matching dates only (for dashboard analysis) with Win Rate calculation"""
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
                    
                    # Calculate hit based on expected direction
                    if expected_direction == "Increase":
                        hit = nominal_change > 0
                    else:  # "Decrease"
                        hit = nominal_change < 0
                    
                    horizon_data.append({
                        'Match_Date': match_date,
                        'Nominal_Change': nominal_change,
                        'Hit': hit
                    })
            except (KeyError, IndexError):
                continue
        
        forward_returns[f'{horizon}D'] = pd.DataFrame(horizon_data) if horizon_data else pd.DataFrame()
    return forward_returns

def create_comprehensive_dataframe(price_data, ticker_weights, indicators, dependent_var, matching_mask, individual_conditions, rolling_return_columns, cumulative_sum_columns, forward_returns_all, horizons):
    # Start with the full dependent variable date range
    df = pd.DataFrame(index=dependent_var.index)
    
    # Add individual tickers used in dependent variable
    active_dep_tickers = [t for t, w in ticker_weights.items() if t and w != 0]
    for ticker in active_dep_tickers:
        if ticker in price_data.columns:
            df[ticker] = price_data[ticker].reindex(dependent_var.index)
    
    # Add dependent variable
    df['Dependent_Variable'] = dependent_var
    
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
    
    # Add forward return columns for ALL dates (only nominal now)
    for horizon in horizons:
        df[f'Forward_{horizon}D_Nominal'] = forward_returns_all[f'Forward_{horizon}D_Nominal']
    
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
                all_tickers = list(dict.fromkeys(active_tickers + indicator_tickers))
                
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
                        
                        # Calculate forward returns for ALL dates (for CSV)
                        forward_returns_all = calculate_forward_returns_all_dates(dependent_var, horizons)
                        
                        # Calculate forward returns for BOTH approaches
                        forward_returns_cluster_free = calculate_forward_returns_matching_only(dependent_var, cluster_free_dates, horizons, expected_direction)
                        forward_returns_all_signals = calculate_forward_returns_matching_only(dependent_var, all_matching_dates, horizons, expected_direction)
                        
                        # Create comprehensive dataset (using filtered matching mask)
                        comprehensive_df = create_comprehensive_dataframe(
                            price_data, ticker_weights, indicators, 
                            dependent_var, filtered_matching_mask, individual_conditions, 
                            rolling_return_columns, cumulative_sum_columns, forward_returns_all, horizons
                        )
                                                # Enhanced Metrics with Clustering Info
                        col1, col2, col3, col4, col5 = st.columns(5)
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
                            else:
                                st.metric("Current Value", f"{dependent_var.iloc[-1]:.4f}")
                        with col5:
                            st.metric("Current Value", f"{dependent_var.iloc[-1]:.4f}")
                        
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
                        
                        # Analysis 1: Cluster-Free Forward Return Analysis
                        if forward_returns_cluster_free and any(not df.empty for df in forward_returns_cluster_free.values()):
                            st.subheader("Cluster-Free Forward Return Analysis")
                            st.markdown(f"**Method:** {cluster_free_days}-day cooldown after each signal | **Expected Direction:** {expected_direction}")
                            
                            # Summary statistics with Win Rate for Cluster-Free
                            summary_data_cf = []
                            for horizon in horizons:
                                horizon_key = f'{horizon}D'
                                if horizon_key in forward_returns_cluster_free and not forward_returns_cluster_free[horizon_key].empty:
                                    df_fwd = forward_returns_cluster_free[horizon_key]
                                    
                                    # Calculate Win Rate and standard deviation
                                    win_rate = df_fwd['Hit'].mean() * 100 if len(df_fwd) > 0 else 0
                                    std_dev = df_fwd['Nominal_Change'].std()
                                    
                                    summary_data_cf.append({
                                        'Horizon': f'{horizon}D',
                                        'Sample Size': len(df_fwd),
                                        'Avg Nominal': df_fwd['Nominal_Change'].mean(),
                                        'Median Nominal': df_fwd['Nominal_Change'].median(),
                                        'Std Dev': std_dev,
                                        'Win Rate': win_rate
                                    })
                            
                            if summary_data_cf:
                                # Display metrics with Win Rate for Cluster-Free
                                horizon_cols = st.columns(len(summary_data_cf))
                                for i, row in enumerate(summary_data_cf):
                                    with horizon_cols[i]:
                                        st.metric(f"{row['Horizon']} Sample", f"{int(row['Sample Size']):,}")
                                        st.metric("Avg Nominal", f"{row['Avg Nominal']:.4f}")
                                        st.metric("Median Nominal", f"{row['Median Nominal']:.4f}")
                                        st.metric("Win Rate", f"{row['Win Rate']:.1f}%", 
                                                help=f"% of times dependent variable moved in expected direction ({expected_direction.lower()})")
                                
                                # Summary table for Cluster-Free
                                st.markdown("**Cluster-Free Summary Statistics:**")
                                summary_df_cf = pd.DataFrame(summary_data_cf)
                                st.dataframe(summary_df_cf.round({'Avg Nominal': 4, 'Median Nominal': 4, 'Win Rate': 1}), 
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
                                        
                                        # Add histogram
                                        fig_dist_cf.add_trace(go.Histogram(x=df_fwd['Nominal_Change'], marker_color=color, opacity=0.7, nbinsx=20), row=1, col=col_idx)
                                        
                                        # Calculate statistics
                                        median_val = df_fwd['Nominal_Change'].median()
                                        std_val = df_fwd['Nominal_Change'].std()
                                        
                                        # Add median line
                                        fig_dist_cf.add_vline(x=median_val, line_dash="dash", line_color="blue", line_width=2, row=1, col=col_idx)
                                        
                                        # Add +1 std deviation line
                                        fig_dist_cf.add_vline(x=median_val + std_val, line_dash="dot", line_color="red", line_width=2, row=1, col=col_idx)
                                        
                                        # Add -1 std deviation line
                                        fig_dist_cf.add_vline(x=median_val - std_val, line_dash="dot", line_color="red", line_width=2, row=1, col=col_idx)
                                        
                                        col_idx += 1

                                fig_dist_cf.update_layout(title="Cluster-Free Forward Return Distributions", template="plotly_dark", height=400, showlegend=False)
                                st.plotly_chart(fig_dist_cf, use_container_width=True)
                        
                        # Analysis 2: All Signals Forward Return Analysis
                        if forward_returns_all_signals and any(not df.empty for df in forward_returns_all_signals.values()):
                            st.subheader("All Signals Forward Return Analysis")
                            st.markdown(f"**Method:** All original signals (no clustering filter) | **Expected Direction:** {expected_direction}")
                            
                            # Summary statistics with Win Rate for All Signals
                            summary_data_all = []
                            for horizon in horizons:
                                horizon_key = f'{horizon}D'
                                if horizon_key in forward_returns_all_signals and not forward_returns_all_signals[horizon_key].empty:
                                    df_fwd = forward_returns_all_signals[horizon_key]
                                    
                                    # Calculate Win Rate and standard deviation
                                    win_rate = df_fwd['Hit'].mean() * 100 if len(df_fwd) > 0 else 0
                                    std_dev = df_fwd['Nominal_Change'].std()
                                    
                                    summary_data_all.append({
                                        'Horizon': f'{horizon}D',
                                        'Sample Size': len(df_fwd),
                                        'Avg Nominal': df_fwd['Nominal_Change'].mean(),
                                        'Median Nominal': df_fwd['Nominal_Change'].median(),
                                        'Std Dev': std_dev,
                                        'Win Rate': win_rate
                                    })
                            
                            if summary_data_all:
                                # Display metrics with Win Rate for All Signals
                                horizon_cols = st.columns(len(summary_data_all))
                                for i, row in enumerate(summary_data_all):
                                    with horizon_cols[i]:
                                        st.metric(f"{row['Horizon']} Sample", f"{int(row['Sample Size']):,}")
                                        st.metric("Avg Nominal", f"{row['Avg Nominal']:.4f}")
                                        st.metric("Median Nominal", f"{row['Median Nominal']:.4f}")
                                        st.metric("Win Rate", f"{row['Win Rate']:.1f}%", 
                                                help=f"% of times dependent variable moved in expected direction ({expected_direction.lower()})")
                                
                                # Summary table for All Signals
                                st.markdown("**All Signals Summary Statistics:**")
                                summary_df_all = pd.DataFrame(summary_data_all)
                                st.dataframe(summary_df_all.round({'Avg Nominal': 4, 'Median Nominal': 4, 'Win Rate': 1}), 
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
                                        
                                        # Add histogram
                                        fig_dist_all.add_trace(go.Histogram(x=df_fwd['Nominal_Change'], marker_color=color, opacity=0.7, nbinsx=20), row=1, col=col_idx)
                                        
                                        # Calculate statistics
                                        median_val = df_fwd['Nominal_Change'].median()
                                        std_val = df_fwd['Nominal_Change'].std()
                                        
                                        # Add median line
                                        fig_dist_all.add_vline(x=median_val, line_dash="dash", line_color="blue", line_width=2, row=1, col=col_idx)
                                        
                                        # Add +1 std deviation line
                                        fig_dist_all.add_vline(x=median_val + std_val, line_dash="dot", line_color="red", line_width=2, row=1, col=col_idx)
                                        
                                        # Add -1 std deviation line
                                        fig_dist_all.add_vline(x=median_val - std_val, line_dash="dot", line_color="red", line_width=2, row=1, col=col_idx)
                                        
                                        col_idx += 1

                                fig_dist_all.update_layout(title="All Signals Forward Return Distributions", template="plotly_dark", height=400, showlegend=False)
                                st.plotly_chart(fig_dist_all, use_container_width=True)

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
                            fig_season_bar.update_layout(title=f"Average {seasonality_freq} {value_label} (± 1 Std Dev)", template="plotly_dark",
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
                            fig_season_heat.update_layout(title=f"{seasonality_freq} {value_label} Heatmap by Year", template="plotly_dark",
                                                           height=600, xaxis_title=period_axis_title, yaxis_title="Year", xaxis_side='top')
                            fig_season_heat.update_yaxes(autorange='reversed')
                            st.plotly_chart(fig_season_heat, use_container_width=True)
                        else:
                            st.info(f"Not enough history in the selected date range to compute {seasonality_freq.lower()} seasonality.")

                        # Time series plot
                        st.subheader("Dependent Variable with Signal Analysis")
                        fig = go.Figure()

                        with st.spinner("Fetching OHLC data for candlestick..."):
                            ohlc_data = fetch_yahoo_ohlc(active_tickers, start_date, end_date)
                        dep_ohlc = calculate_dependent_variable_ohlc(ohlc_data, ticker_weights)
                        dep_ohlc = dep_ohlc.reindex(dependent_var.index).dropna()

                        if not dep_ohlc.empty:
                            fig.add_trace(go.Candlestick(
                                x=dep_ohlc.index, open=dep_ohlc['Open'], high=dep_ohlc['High'],
                                low=dep_ohlc['Low'], close=dep_ohlc['Close'], name='Dependent Variable',
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
                        
                        fig.update_layout(title="Dependent Variable Time Series with Dual Analysis", template="plotly_dark", height=500)
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

                        fig_components.update_layout(title="Weighted Components and Total", template="plotly_dark", height=400, showlegend=True)
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
    - **Analyze forward returns** with dual analysis approach
    - **Download comprehensive datasets** for further analysis
    
    **Data Source: Yahoo Finance**
    - Use Yahoo Finance ticker symbols (e.g. `AAPL`, `^VIX`, `^TNX`, `EURUSD=X`, `CL=F`)
    - Daily close prices are pulled via `yfinance`
    - Economic-release series (CPI, GDP, etc.) aren't available through Yahoo Finance - the
      "Economic Data" forward-fill option is best used for genuinely gappy market series
    
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
