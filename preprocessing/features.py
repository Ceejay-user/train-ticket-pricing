import pandas as pd
import numpy as np

def clean_base_data(data):
    """Data cleaning"""

    data = data.copy()

    # standardizing all column case
    if not all(col.islower() for col in data.columns):
        data.columns = data.columns.str.lower()

    # Filling missing values for loyalty status
    if 'loyalty_status' in data.columns:
        data['loyalty_status'] = data['loyalty_status'].fillna('Not loyal')

    # Filling in missing data for Base_Price_At_Booking and Price_Premium ensuring the equation above is consistent
    if 'base_price_at_booking' in data.columns:
        if 'ticket_price_gbp' in data.columns:
            mask = data['base_price_at_booking'].isna()
            data.loc[mask, 'base_price_at_booking'] = data.loc[mask, 'ticket_price_gbp']
        
        # Ensure price_premium is handled
        if 'price_premium' in data.columns:
            data['price_premium'] = data['price_premium'].fillna(0)
        else:
            # If premium is missing from raw input, default it to 0
            data['price_premium'] = 0

    return data




def engineer_time_features(data):
    """time feature engineering"""
    data = data.copy()
    
    if 'booking_date' in data.columns:
        data['booking_month'] = data['booking_date'].dt.month
        data['booking_day_of_week'] = data['booking_date'].dt.dayofweek

    if 'travel_date' in data.columns:
        data['travel_month'] = data['travel_date'].dt.month
        data['travel_day_of_week'] = data['travel_date'].dt.dayofweek
        data['is_weekend_travel'] = data['travel_day_of_week'].isin([4, 5, 6]).astype(int)

    cols_to_drop = ['booking_date', 'travel_date', 'booking_timestamp']
    existing_cols_to_drop = [col for col in cols_to_drop if col in data.columns]
    # Apply the drop only if there is something to drop
    if existing_cols_to_drop:
        data = data.drop(columns=existing_cols_to_drop)
    return data




def engineer_customer_strength_features(data):
    """Customer strength feature engineering"""
    data = data.copy()

    if 'booking_frequency_qtr' in data.columns and 'average_spend_gbp' in data.columns:
        data['customer_quarterly_revenue'] = data['booking_frequency_qtr'] * data['average_spend_gbp']

    cols_to_drop = ['booking_frequency_qtr', 'average_spend_gbp']
    existing_cols = [col for col in cols_to_drop if col in data.columns]
    # Apply the drop only if there is something to drop
    if existing_cols:
        data = data.drop(columns=existing_cols)
    return data




def engineer_scarcity_features(data):
    """Scarcity features engineering"""
    data = data.copy()

    # Seat utilization ratio
    if 'seats_sold_realized' in data.columns and 'total_seats' in data.columns:
        data['seat_utilization_ratio'] = (data['seats_sold_realized'] / data['total_seats'].replace(0, np.nan))

    # Scarcity index
    if 'demand_index' in data.columns and 'seat_utilization_ratio' in data.columns:
        data['scarcity_index'] = (data['demand_index'] * data['seat_utilization_ratio'])

    cols_to_drop = ['seat_utilization_ratio', 'seats_sold_realized', 'total_seats', 'demand_index']
    existing_cols = [col for col in cols_to_drop if col in data.columns]
    # Apply the drop only if there is something to drop
    if existing_cols:
        data = data.drop(columns=existing_cols)
        
    return data





def drop_features(data):
    """Dropping features not required for modeling"""
    data = data.copy()

    cols_to_drop = ['travel_day_of_week', 'base_price_at_booking', 'price_premium', 'load_factor', 'booking_day_of_week']
    existing_cols_to_drop = [col for col in cols_to_drop if col in data.columns]
    # Apply the drop only if there is something to drop
    if existing_cols_to_drop:
        data = data.drop(columns=existing_cols_to_drop)
    return data