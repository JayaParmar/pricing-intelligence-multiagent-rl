import pandas as pd
from preprocessing.data_loader import load_and_preprocess
from preprocessing.simulation import (
    get_vendor_logit_model, calculate_elasticity,
    calculate_utility, load_shop_data
)


df = load_and_preprocess()
print(df.columns)

all_shops = df['SHOP'].unique()
output_path = "../data/df_simulated_all.csv"
simulated_list = []
for shop in all_shops:
    print(f"Processing shop: {shop}")

    df_shop = load_shop_data(df, shop)

    # Skip if empty
    if df_shop.empty:
        print(f"Skipping {shop} — no data")
        continue

    # Build logit model and simulate
    stats = df_shop.groupby('SHOP_VENDOR_NAME').agg(
        total_market_share=('Market Share (%)', 'sum'),
        average_effective_price=('EFFECTIVE_PRICE', 'mean')
    ).reset_index()

    base_vendor = stats.loc[stats['total_market_share'].idxmax(), 'SHOP_VENDOR_NAME']
    beta_0, beta_1, _, base_vendor, _ = get_vendor_logit_model(df_shop, base_vendor)

    df_sim = calculate_elasticity(df_shop, beta_1)
    df_sim = calculate_utility(df_sim, beta_0, beta_1)

    simulated_list.append(df_sim)

# Combine all shops' simulated data
df_all_simulated = pd.concat(simulated_list, ignore_index=True)

# Optional: drop duplicates (if needed)
df_all_simulated.drop_duplicates(
    subset=['OBSERVED_TIME', 'SHOP', 'SHOP_VENDOR_NAME'], inplace=True
)

# Save final output
df_all_simulated.to_csv(output_path, index=False)
print(f"\n✅ All shops processed. Output saved to: {output_path}")