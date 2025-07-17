import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st # Import streamlit for the cache decorator

def calculate_seasonal_strength(
    series: pd.Series, 
    period: int, 
    model: str = 'additive',
    fill_method: str = 'zero'  # CHANGED: Default is now 'zero'
) -> float:
    """
    Calculates the strength of seasonality for a single time series.
    Returns a float between 0 and 1, or np.nan if calculation fails.
    """
    # A series needs at least two full periods of data to be reliable
    if series.dropna().shape[0] < 2 * period:
        return np.nan

    try:
        # Ensure the series has a complete daily DatetimeIndex
        full_range_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq='D')
        series_reindexed = series.reindex(full_range_idx)

        # Fill missing values based on the selected method
        if fill_method == 'zero':
            series_filled = series_reindexed.fillna(0)
        elif fill_method == 'interpolate':
            series_filled = series_reindexed.interpolate(method='linear', limit_direction='both')
        elif fill_method == 'ffill':
            series_filled = series_reindexed.ffill().bfill().fillna(0) # Also fill any remaining NaNs
        else:
            # Default to zero fill if method is invalid
            series_filled = series_reindexed.fillna(0)
        
        # If all values are zero after filling, there's no seasonality to calculate
        if series_filled.sum() == 0:
            return 0.0

        # Multiplicative model requires values > 0
        if model == 'multiplicative':
            series_filled = series_filled.replace(0, 1e-6)

        decomposition = seasonal_decompose(
            series_filled,
            model=model,
            period=period,
            extrapolate_trend='freq' # More robust trend extrapolation
        )

        # Calculate strength of seasonality vs. randomness (residuals)
        seasonal_variation = np.var(decomposition.seasonal.dropna())
        remainder_variation = np.var(decomposition.resid.dropna())

        if (seasonal_variation + remainder_variation) == 0:
            return 0.0 # No variation at all

        # Strength is the proportion of variance attributable to the seasonal component
        return float(seasonal_variation / (seasonal_variation + remainder_variation))

    except Exception as e:
        # st.warning(f"Could not decompose series: {e}") # Optional: show warning in UI
        return np.nan


# ADDED: Streamlit's caching decorator to speed up the app
@st.cache_data 
def identify_seasonal_skus(
    sales_data: pd.DataFrame,
    min_seasonal_strength: float = 0.5,
    period_days: int = 365,
    time_col: str = 'timestamp',
    id_col: str = 'item_id',
    target_col: str = 'target',
    model: str = 'additive',
    fill_method: str = 'zero' # CHANGED: Default is now 'zero'
) -> pd.DataFrame:
    """
    Identifies seasonal SKUs from sales data and calculates their properties.
    This function is cached, so it only re-runs when input parameters change.
    """
    print(f"\n❄️ (Re-calculating) Identifying seasonal SKUs...") # This print will only show on the first run

    df = sales_data.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=[id_col, time_col])

    # Group by SKU and apply the seasonality calculation
    # This is more efficient than a for-loop for very large numbers of SKUs
    seasonal_results = df.groupby(id_col).apply(
        lambda g: calculate_seasonal_strength(
            g.set_index(time_col)[target_col].resample('D').sum(),
            period=period_days,
            model=model,
            fill_method=fill_method
        )
    ).rename('seasonal_strength').reset_index()

    # Get other stats
    stats = df.groupby(id_col)[target_col].agg(['count', 'mean', 'std']).reset_index()
    seasonal_results = pd.merge(seasonal_results, stats, on=id_col)
    
    # Finalize the DataFrame
    seasonal_results['seasonal_strength'] = seasonal_results['seasonal_strength'].fillna(0.0)
    seasonal_results['is_seasonal'] = seasonal_results['seasonal_strength'] >= min_seasonal_strength
    
    print(f"✅ Seasonal SKU identification complete.")
    return seasonal_results
