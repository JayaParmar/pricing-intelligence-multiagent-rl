import os
import pandas as pd
import snowflake.connector

from preprocessing.transformation import (
    clean_price_columns,
    calculate_discount,
    clean_vendor_desc,
    calculate_effective_price,
    add_datetime_features
)
from preprocessing.simulation import (
    simulate_volume_driven_market_all_agents,
    add_competitor_price_column
)

def load_snowflake_data():
    key_path = os.path.join('ssh', 'rsa_key_snowflake.der')
    with open(key_path, 'rb') as key_file:
        private_key = key_file.read()

    conn = snowflake.connector.connect(
        user='webcrawler',
        private_key=private_key,
        account='DEV_ACCOUNT',
        authenticator="externalbrowser",
        warehouse='RETAILER_WH',
        database='CRAWLER',
        schema='CRAWLER_DATA',
        role='CRAWLER_ROLE'
    )

    query = "SELECT * FROM OBSERVATIONS_2"
    return pd.read_sql(query, conn)

def load_and_preprocess():
    df = load_snowflake_data()
    df = clean_price_columns(df)
    df = calculate_discount(df)
    df = clean_vendor_desc(df)
    df = calculate_effective_price(df)
    df = add_datetime_features(df)
    df = simulate_volume_driven_market_all_agents(df)
    df = add_competitor_price_column(df, your_vendor='abc', how='mean')
    return df