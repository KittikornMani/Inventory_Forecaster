"""Simple order-up-to calculation for the forecast horizon."""

import math

import pandas as pd


def recommend(forecast: pd.DataFrame, current_stock: float, lead_days: int, safety_pct: float):
    daily = float(forecast["predicted_sales"].mean())
    total = float(forecast["predicted_sales"].sum())
    lead_demand = daily * lead_days
    safety = lead_demand * safety_pct / 100
    reorder_point = lead_demand + safety
    # Order enough to cover the chosen horizon plus safety stock; lead-time
    # demand is already part of the horizon when lead_days <= horizon.
    order_up_to = total + safety
    quantity = math.ceil(max(0, order_up_to - current_stock))
    remaining = current_stock - forecast["predicted_sales"].cumsum()
    crossing = forecast.loc[remaining <= reorder_point, "date"]
    return {
        "average_daily_demand": daily,
        "forecasted_demand": total,
        "safety_stock": safety,
        "reorder_point": reorder_point,
        "suggested_order_quantity": quantity,
        "risk": "Reorder now" if current_stock <= reorder_point else "Above reorder point",
        "first_reorder_date": crossing.iloc[0] if len(crossing) else None,
        "remaining_stock": remaining,
    }
