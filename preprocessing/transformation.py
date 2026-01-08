import numpy as np
import pandas as pd
import streamlit as st

def clean_price_columns(df, price_cols=['REGULAR_PRICE', 'OFFER_PRICE']):
    for col in price_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace(',', '.', regex=False)
            .str.extract(r'(\d+\.?\d*)')[0]
            .astype(float)
            .fillna(0)
        )
    return df

def calculate_discount(df):
    df['DISCOUNT'] = np.where(
        (df['REGULAR_PRICE'] > 0) & (df['OFFER_PRICE'] > 0),
        ((df['REGULAR_PRICE'] - df['OFFER_PRICE']) / df['REGULAR_PRICE']) * 100,
        0
    ).astype(int)
    return df

def clean_vendor_desc(df):
    df['SHOP_VENDOR_NAME'] = df['SHOP_VENDOR_NAME'].astype(str).str.lower().str.strip()
    df['DESC'] = df['DESC'].astype(str).str.replace(r'(?i)^na$', 'na', regex=True)

    mask_def_na = (df['SHOP'] == 'def') & (df['SHOP_VENDOR_NAME'].isin(['na', 'n/a', 'nan', '', 'null']))
    df.loc[mask_def_na, 'SHOP_VENDOR_NAME'] = 'abc'

    df = df[~((df['SHOP_VENDOR_NAME'] == 'na') & (df['SHOP'] != 'def'))]
    return df

def calculate_effective_price(df):
    df['EFFECTIVE_PRICE'] = df.apply(
        lambda row: row['OFFER_PRICE'] if row['OFFER_PRICE'] > 0 else row['REGULAR_PRICE'],
        axis=1
    )
    return df

def add_datetime_features(df):
    if 'OBSERVED_TIME' in df.columns:
        df['OBSERVED_TIME'] = pd.to_datetime(df['OBSERVED_TIME'])
        df['promotion_flag'] = (df['DISCOUNT'] > 0).astype(int)
        df['day_of_week'] = df['OBSERVED_TIME'].dt.dayofweek
        df['month'] = df['OBSERVED_TIME'].dt.month
    else:
        st.warning("Warning: 'OBSERVED_TIME' column is missing, skipping datetime features.")
    return df