import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler

def extract_state_vector(row, df, agent_vendor=None):
    timestamp = pd.to_datetime(row['OBSERVED_TIME'])  # <- convert explicitly
    shop = row['SHOP']

    #time_window = timedelta(hours=12)
    time_window = timedelta(days=3)
    df['OBSERVED_TIME'] = pd.to_datetime(df['OBSERVED_TIME'])  

    if agent_vendor is None:
        agent_vendor = row['SHOP_VENDOR_NAME']

    competitors = df[
        (df['SHOP'] == shop) &
        (df['SHOP_VENDOR_NAME'] != agent_vendor) &  # <- now dynamic
        (df['OBSERVED_TIME'] >= timestamp - time_window) &
        (df['OBSERVED_TIME'] <= timestamp + time_window)
    ]

    comp_price = competitors['EFFECTIVE_PRICE'].astype(float).mean() if not competitors.empty else 0.0
    comp_avail = competitors['AVAILABILITY'].apply(lambda x: 'disp' in str(x).lower()).astype(int).mean() if not competitors.empty else 0.0

    elasticity = float(row['elasticity'])
    day_of_week = int(row['day_of_week'])
    obs_hour = pd.to_datetime(timestamp).hour

    return np.array([comp_price, comp_avail, day_of_week, obs_hour, elasticity], dtype=np.float32)

def get_state(row, df, scaler):
    raw = extract_state_vector(row, df, row['SHOP_VENDOR_NAME'])  # <- Pass agent name explicitly
    return scaler.transform([raw])[0]

def build_scaler(df, vendor_filter=None):
    if vendor_filter:
        filtered = df[df['SHOP_VENDOR_NAME'] == vendor_filter]
    else:
        filtered = df
    raw_states = filtered.apply(lambda r: extract_state_vector(r, df, r['SHOP_VENDOR_NAME']), axis=1).tolist()
    scaler = StandardScaler()
    scaler.fit(raw_states)
    return scaler

