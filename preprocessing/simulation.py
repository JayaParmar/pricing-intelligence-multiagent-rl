import numpy as np
import pandas as pd

def simulate_volume_driven_market_all_agents(
    df_slice: pd.DataFrame,
    beta_0: float,
    beta_1: float,
    total_demand: float = 100.0  # optional scaling for realism
) -> pd.DataFrame:
    """
    Simulate market share using logit utilities based on price.

    Args:
        df_slice: Vendors' pricing info for one date/shop.
        beta_0, beta_1: Logit model parameters for this shop.
        total_demand: Optional scalar for total market volume.

    Returns:
        DataFrame enriched with market shares and revenue.
    """
    prices = df_slice['EFFECTIVE_PRICE'].astype(float).values
    costs = df_slice['Cost'].astype(float).values if 'Cost' in df_slice.columns else np.full(len(prices), 35.0)

    # Step 1: Compute utility and softmax shares
    utility = beta_0 + beta_1 * prices
    exp_u = np.exp(utility - np.max(utility))  # softmax stability
    softmax_shares = exp_u / np.sum(exp_u)

    # Step 2: Compute volumes and revenues
    volumes = total_demand * softmax_shares
    revenue = prices * volumes
    profit = (prices - costs) * volumes

    df_slice = df_slice.copy()
    df_slice['MarketShareSim'] = softmax_shares
    df_slice['VolumeSoldSim'] = volumes
    df_slice['RevenueSimulated'] = revenue
    df_slice['ProfitSimulated'] = profit

    return df_slice

# Optional utility
def rescale_action(action, min_price, max_price):
    return min_price + (action + 1.0) * 0.5 * (max_price - min_price)


def add_competitor_price_column(df, your_vendor='abc', how='mean'):
    competitor_prices = df[df['SHOP_VENDOR_NAME'] != your_vendor]
    agg_func = {'mean': 'mean', 'min': 'min', 'max': 'max'}[how]
    comp_price_summary = (
        competitor_prices
        .groupby(['SHOP', 'OBSERVED_TIME'])['EFFECTIVE_PRICE']
        .agg(comp_price=agg_func)
        .reset_index()
    )
    df = df.merge(comp_price_summary, on=['SHOP', 'OBSERVED_TIME'], how='left')
    df.rename(columns={'comp_price': 'COMPETITOR_EFFECTIVE_PRICE'}, inplace=True)
    return df

def get_vendor_logit_model(df, base_vendor):
    vendor_group = df.groupby('SHOP_VENDOR_NAME').agg(
        avg_price=('EFFECTIVE_PRICE', 'mean'),
        avg_share=('Market Share (%)', 'mean')
    ).reset_index()

    if base_vendor not in vendor_group['SHOP_VENDOR_NAME'].values:
        base_vendor = vendor_group.loc[vendor_group['avg_share'].idxmax(), 'SHOP_VENDOR_NAME']

    base_stats = vendor_group[vendor_group['SHOP_VENDOR_NAME'] == base_vendor].iloc[0]
    base_price = base_stats['avg_price']
    base_share = base_stats['avg_share'] / 100

    beta_1 = -1 / base_price
    beta_0 = np.log(base_share / (1 - base_share)) - beta_1 * base_price
    return beta_0, beta_1, vendor_group, base_vendor, base_share

def calculate_elasticity(df, beta_1):
    df['elasticity'] = beta_1 * df['EFFECTIVE_PRICE'] * (1 - df['Market Share (%)'] / 100)
    return df

def calculate_utility(df, beta_0, beta_1):
    df['utility'] = beta_0 + beta_1 * df['EFFECTIVE_PRICE']
    return df

def load_shop_data(df, selected_shop):
    #return df[df['SHOP'] == selected_shop].copy()
    return df[df['SHOP'] == selected_shop].copy().reset_index(drop=True)

def simulate_future_day(date, shop, vendors, df_reference, scaler):
    rows = []
    df_abc = df_reference[df_reference['SHOP_VENDOR_NAME'] == 'abc']
    reference_rows = df_abc.sample(n=len(vendors), replace=True)  # base pool for diversity

    for vendor, ref_row in zip(vendors, reference_rows.iterrows()):
        _, row = ref_row
        new_row = row.copy()

        # Assign synthetic timestamp
        new_row['OBSERVED_TIME'] = pd.to_datetime(date)
        new_row['SHOP'] = shop
        new_row['SHOP_VENDOR_NAME'] = vendor

        # Perturb features
        new_row['EFFECTIVE_PRICE'] = max(1.0, row['EFFECTIVE_PRICE'] * np.random.normal(1, 0.03))
        new_row['AVAILABILITY'] = 'disp' if np.random.rand() > 0.1 else 'no stock'
        new_row['elasticity'] = row['elasticity'] * np.random.normal(1, 0.1)
        new_row['day_of_week'] = date.weekday()
        new_row['Cost'] = max(0.5, row['Cost'] * np.random.normal(1, 0.02))

        rows.append(new_row)

    return pd.DataFrame(rows)