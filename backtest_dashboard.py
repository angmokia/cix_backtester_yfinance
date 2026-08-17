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

# Main Content
if calculate_button:
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
                        
                        # Time series plot
                        st.subheader("Dependent Variable with Signal Analysis")
                        fig = go.Figure()
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