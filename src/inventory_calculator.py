# inventory_calculator.py
"""
Contains functions for calculating inventory metrics like reorder point and EOQ.
"""

import math
import pandas as pd
import numpy as np
from scipy.stats import norm

def calculate_inventory_metrics(forecast_df, service_level, config):
    """
    Calculates safety stock, reorder point, and EOQ based on forecasts.
    
    Args:
        forecast_df (pd.DataFrame): DataFrame of forecasts for a single item.
        service_level (float): The desired service level (e.g., 0.95 for 95%).
        config (module): The configuration module with inventory parameters.

    Returns:
        dict: A dictionary with calculated inventory metrics.
    """
    if forecast_df.empty or 'mean' not in forecast_df.columns:
        return {'Lead Time Demand': 0, 'Safety Stock': 0, 'Reorder Point': 0, 'Validated EOQ': 0}

    # Extract config values for readability
    lead_time = config.LEAD_TIME_DAYS
    moq = config.MINIMUM_ORDER_QUANTITY
    ordering_cost = config.ORDERING_COST
    holding_cost = config.ANNUAL_HOLDING_COST_PER_UNIT

    # Calculations
    avg_daily_forecast = forecast_df['mean'].mean()
    forecast_demand_std = forecast_df['mean'].std()
    
    # Ensure standard deviation is not NaN (can happen for flat forecasts)
    if pd.isna(forecast_demand_std):
        forecast_demand_std = 0

    z_score = norm.ppf(service_level)
    
    safety_stock = z_score * forecast_demand_std * np.sqrt(lead_time)
    lead_time_demand = avg_daily_forecast * lead_time
    reorder_point = lead_time_demand + safety_stock
    
    annual_demand = max(0, avg_daily_forecast * 365)
    
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost) if holding_cost > 0 else 0
    
    # Ensure EOQ respects the Minimum Order Quantity
    validated_eoq = max(eoq, moq)

    return {
        'Lead Time Demand': lead_time_demand,
        'Safety Stock': safety_stock,
        'Reorder Point': reorder_point,
        'Validated EOQ': validated_eoq
    }

def generate_recommendations(predictions, item_to_class_map, config):
    """
    Iterates through predictions to generate final inventory recommendations.
    """
    print("\nCalculating inventory metrics for all forecasted items...")
    results = []
    
    for item_id_channel in predictions.item_ids:
        item_forecast_df = predictions.loc[item_id_channel]
        original_sku, channel = item_id_channel.rsplit('_', 1)
        
        # Determine service level from ABC class
        item_class = item_to_class_map.get(original_sku, 'C') # Default to 'C' if not found
        service_level = config.ABC_CONFIG[f'service_level_{item_class}']

        for horizon_name, days in config.HORIZONS.items():
            forecast_slice = item_forecast_df.head(days)
            total_demand_forecast = forecast_slice['mean'].sum()
            
            # Calculate inventory metrics for the slice
            metrics = calculate_inventory_metrics(forecast_slice, service_level, config)
            
            results.append({
                "item_id": original_sku,
                "channel": channel,
                "recommended_warehouse": config.CHANNEL_TO_WH_RECOMMENDATION.get(channel, 'unknown_wh'),
                "horizon": horizon_name,
                "forecast_days": days,
                "total_forecasted_demand": round(total_demand_forecast, 2),
                "safety_stock": int(round(metrics.get('Safety Stock', 0), 0)),
                "reorder_point": int(round(metrics.get('Reorder Point', 0), 0)),
                "economic_order_quantity (EOQ)": int(round(metrics.get('Validated EOQ', 0), 0))
            })
            
    return pd.DataFrame(results)

def apply_business_rules(recommendations_df):
    """
    Placeholder function to apply custom business logic to the final recommendations.
    For example, you could cap order quantities, adjust for supplier holidays, etc.
    """
    print("Applying custom business rules (placeholder)...")
    # Example: Cap reorder point at a certain level
    # recommendations_df['reorder_point'] = recommendations_df['reorder_point'].clip(upper=1000)
    return recommendations_df
