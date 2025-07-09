import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def calculate_seasonal_strength(
    series: pd.Series, 
    period: int, 
    model: str = 'additive',
    fill_method: str = 'interpolate'  # new param for flexibility
) -> float:
    """
    Calculates the strength of seasonality in a time series.

    Args:
        series (pd.Series): Time series with DatetimeIndex.
        period (int): Seasonality period (e.g., 365 for yearly).
        model (str): 'additive' or 'multiplicative'.
        fill_method (str): How to fill missing dates ('interpolate', 'zero', 'ffill').

    Returns:
        float: Seasonal strength (0 to 1).
    """
    # Check minimum data
    if series.isnull().all() or len(series.dropna()) < 2 * period:
        return np.nan

    try:
        # Ensure datetime index
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)

        # Fill missing dates
        full_range_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq='D')
        series_reindexed = series.reindex(full_range_idx)

        if fill_method == 'interpolate':
            series_filled = series_reindexed.interpolate(limit_direction='both')
        elif fill_method == 'zero':
            series_filled = series_reindexed.fillna(0)
        elif fill_method == 'ffill':
            series_filled = series_reindexed.fillna(method='ffill').fillna(method='bfill')
        else:
            raise ValueError(f"Invalid fill_method: {fill_method}")

        # If all zeros or NaNs
        if series_filled.sum() == 0 or series_filled.isnull().all():
            return np.nan

        # For multiplicative model, no zeros allowed
        if model == 'multiplicative':
            series_filled = series_filled.replace(0, 1e-6)

        # Decompose
        decomposition = seasonal_decompose(
            series_filled,
            model=model,
            period=period,
            extrapolate_trend=period
        )

        seasonal_variation = np.var(decomposition.seasonal.dropna())
        remainder_variation = np.var(decomposition.resid.dropna())

        # Avoid division by zero
        if (seasonal_variation + remainder_variation) == 0:
            return 0.0

        seasonal_strength = seasonal_variation / (seasonal_variation + remainder_variation)
        return float(seasonal_strength)

    except Exception as e:
        print(f"⚠️ Decomposition error: {e}")
        return np.nan


def identify_seasonal_skus(
    sales_data: pd.DataFrame,
    min_seasonal_strength: float = 0.5,
    period_days: int = 365,
    time_col: str = 'timestamp',
    id_col: str = 'item_id',
    target_col: str = 'target',
    model: str = 'additive',
    fill_method: str = 'interpolate'
) -> pd.DataFrame:
    """
    Identifies SKUs with significant seasonality.

    Args:
        sales_data (pd.DataFrame): Sales history.
        min_seasonal_strength (float): Threshold to classify as seasonal.
        period_days (int): Seasonality period.
        time_col (str): Time column name.
        id_col (str): SKU column name.
        target_col (str): Sales column name.
        model (str): Decomposition model ('additive' or 'multiplicative').
        fill_method (str): How to fill missing dates.

    Returns:
        pd.DataFrame: SKU seasonality metrics.
    """
    print(
        f"\n❄️ Identifying seasonal SKUs with period={period_days} days, "
        f"min_seasonal_strength={min_seasonal_strength}, model={model}..."
    )

    sales_data = sales_data.copy()
    sales_data[time_col] = pd.to_datetime(sales_data[time_col])
    sales_data = sales_data.sort_values(by=[id_col, time_col])

    results = []

    for item_id, group in sales_data.groupby(id_col):
        series = group.set_index(time_col)[target_col].resample('D').sum()
        n_points = len(series)
        mean_sales = series.mean()
        std_sales = series.std()

        strength = calculate_seasonal_strength(
            series,
            period=period_days,
            model=model,
            fill_method=fill_method
        )

        results.append({
            id_col: item_id,
            'seasonal_strength': strength,
            'is_seasonal': (strength >= min_seasonal_strength) if not np.isnan(strength) else False,
            'n_points': n_points,
            'mean_sales': mean_sales,
            'std_sales': std_sales
        })

    seasonal_df = pd.DataFrame(results)
    seasonal_df['is_seasonal'] = seasonal_df['is_seasonal'].fillna(False)
    seasonal_df['seasonal_strength'] = seasonal_df['seasonal_strength'].fillna(0.0)

    print(
        f"✅ Seasonal SKU identification complete. "
        f"Found {seasonal_df['is_seasonal'].sum()} seasonal SKUs."
    )
    return seasonal_df
