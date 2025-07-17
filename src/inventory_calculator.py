import math
import pandas as pd
import numpy as np
from scipy.stats import norm

def calculate_inventory_metrics(forecast_df, service_level, config):
    if forecast_df.empty or 'mean' not in forecast_df.columns:
        return {'Lead Time Demand': 0, 'Safety Stock': 0, 'Reorder Point': 0, 'Validated EOQ': 0}

    forecast_df['mean'] = forecast_df['mean'].clip(lower=0)

    lead_time = config.LEAD_TIME_DAYS
    moq = config.MINIMUM_ORDER_QUANTITY
    ordering_cost = config.ORDERING_COST
    holding_cost = config.ANNUAL_HOLDING_COST_PER_UNIT

    
    avg_daily_forecast = forecast_df['mean'].mean()
    forecast_demand_std = forecast_df['mean'].std()
    
    
    if pd.isna(forecast_demand_std):
        forecast_demand_std = 0

    z_score = norm.ppf(service_level)
    safety_stock = z_score * forecast_demand_std * np.sqrt(lead_time)
    lead_time_demand = avg_daily_forecast * lead_time
    reorder_point = lead_time_demand + safety_stock
    annual_demand = avg_daily_forecast * 365
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost) if holding_cost > 0 and annual_demand > 0 else 0
    validated_eoq = max(eoq, moq)
    return {
        'Lead Time Demand': max(0, lead_time_demand),
        'Safety Stock': max(0, safety_stock),
        'Reorder Point': max(0, reorder_point),
        'Validated EOQ': max(0, validated_eoq)
    }

def generate_recommendations(predictions, item_to_class_map, config):
    print("\nCalculating inventory metrics for all forecasted items...")
    results = []
    for item_id_channel in predictions.item_ids:
        item_forecast_df = predictions.loc[item_id_channel]
        original_sku, channel = item_id_channel.rsplit('_', 1)
        item_class = item_to_class_map.get(original_sku, 'C') 
        service_level = config.ABC_CONFIG[f'service_level_{item_class}']

        for horizon_name, days in config.HORIZONS.items():
            forecast_slice = item_forecast_df.head(days).copy() 
            metrics = calculate_inventory_metrics(forecast_slice, service_level, config)
            total_demand_forecast = forecast_slice['mean'].clip(lower=0).sum()
            results.append({
                "item_id": original_sku,
                "channel": channel,
                "recommended_warehouse": config.CHANNEL_TO_WH_RECOMMENDATION.get(channel, 'unknown_wh'),
                "horizon": horizon_name,
                "forecast_days": days,
                "total_forecasted_demand": round(total_demand_forecast, 2),
                "safety_stock": int(round(metrics.get('Safety Stock', 0))),
                "reorder_point": int(round(metrics.get('Reorder Point', 0))),
                "economic_order_quantity (EOQ)": int(round(metrics.get('Validated EOQ', 0)))
            })
            
    return pd.DataFrame(results)

def apply_business_rules(recommendations_df):
    print("Applying custom business rules (placeholder)...")
    return recommendations_df
