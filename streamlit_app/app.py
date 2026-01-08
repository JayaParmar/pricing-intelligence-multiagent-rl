import streamlit as st
import pandas as pd
import numpy as np
import ast
import os

st.set_page_config(page_title="Multi-Day Promotion Simulator", layout="wide")
st.title("📊 Multi-Agent Pricing Simulator with Promotion Planning")

# --- Load available shops ---
data_dir = "models_sac_pricing"
available_shops = [f.replace("_predicted.csv", "") for f in os.listdir(data_dir) if f.endswith("_predicted.csv")]

# --- Shop Selection ---
st.sidebar.header("🏬 Shop Selection")
selected_shop = st.sidebar.selectbox("Choose Shop", available_shops)

# --- Load selected shop data ---
df = pd.read_csv(os.path.join(data_dir, f"{selected_shop}_predicted.csv"))

# Parse values
def extract_scalar(val):
    if isinstance(val, str):
        val = ast.literal_eval(val)
    if isinstance(val, list):
        return float(val[0])
    return float(val)

df["PRED_PRICE"] = df["PRED_PRICE"].apply(extract_scalar)
df["PRED_SHARE"] = df["PRED_SHARE"].apply(extract_scalar)

# --- Sidebar inputs ---
vendors = sorted(df["VENDOR"].unique())
days = sorted(df["DAY"].unique())

st.sidebar.header("🛠️ Promotion Setup")
promo_vendor = st.sidebar.selectbox("Vendor on Promotion", vendors)
start_day = st.sidebar.selectbox("📆 Promotion starts on Day:", days)
duration = st.sidebar.slider("📆 Promotion lasts for (days):", min_value=1, max_value=30, value=7)
override_price = st.sidebar.slider("💶 Override Price (EUR)", min_value=50, max_value=200, value=120)

# Define promo period
promo_days = list(range(start_day, min(start_day + duration, max(days) + 1)))
beta = 0.1

# --- Simulate competitor response ---
df_sim = df.copy()

# 1. Average price per vendor per day
df_avg = (
    df_sim.groupby(["DAY", "VENDOR"], as_index=False)
    .agg({"PRED_PRICE": "mean"})
)

# 2. Apply promotion override (across all matching vendor-day rows)
mask = (df_avg["VENDOR"] == promo_vendor) & (df_avg["DAY"].isin(promo_days))
df_avg.loc[mask, "PRED_PRICE"] = override_price

# 3. Recompute softmax per day
def recompute_day_shares(group):
    prices = group["PRED_PRICE"].values
    utilities = -beta * prices
    probs = np.exp(utilities - np.max(utilities))
    shares = probs / np.sum(probs)
    group["SHARE_DYNAMIC"] = shares
    group["REVENUE_DYNAMIC"] = group["PRED_PRICE"].values * shares
    return group

df_result = df_avg.groupby("DAY").apply(recompute_day_shares).reset_index(drop=True)

# --- Plotting section ---
df_grouped = df_result.copy()
pivot_price = df_grouped.pivot(index="DAY", columns="VENDOR", values="PRED_PRICE")
pivot_share = df_grouped.pivot(index="DAY", columns="VENDOR", values="SHARE_DYNAMIC")
pivot_revenue = df_grouped.pivot(index="DAY", columns="VENDOR", values="REVENUE_DYNAMIC")

st.subheader("📈 Price Trajectories")
st.line_chart(pivot_price)

st.subheader("📊 Market Share Over Time")
st.line_chart(pivot_share)

st.subheader("💶 Revenue Over Time")
st.line_chart(pivot_revenue)

# ---- Summary Table Across Periods ----
df_grouped = df_grouped.rename(columns={
    "PRED_PRICE": "Price",
    "SHARE_DYNAMIC": "Share",
    "REVENUE_DYNAMIC": "Revenue"
})

def label_period(day):
    if day in promo_days:
        return "Promo"
    elif day < promo_days[0]:
        return "Before"
    else:
        return "After"

df_grouped["Period"] = df_grouped["DAY"].apply(label_period)

st.subheader("📋 Summary Tables by Period")

summary_df = (
    df_grouped.groupby(["VENDOR", "Period"])
    .agg(
        Avg_Price=("Price", "mean"),
        Avg_Share=("Share", "mean"),
        Total_Revenue=("Revenue", "sum")
    )
    .round(2)
    .reset_index()
)

before_df = summary_df[summary_df["Period"] == "Before"].sort_values("VENDOR")
promo_df = summary_df[summary_df["Period"] == "Promo"].sort_values("VENDOR")
after_df = summary_df[summary_df["Period"] == "After"].sort_values("VENDOR")

st.markdown("### ⏮️ Before Promotion")
st.dataframe(before_df)

st.markdown("### 🎯 During Promotion")
st.dataframe(promo_df)

st.markdown("### ⏭️ After Promotion")
st.dataframe(after_df)

# --- CSV Save Option ---
st.sidebar.header("📁 Export")
if st.sidebar.button("Save simulation as CSV"):
    save_path = os.path.join(data_dir, f"simulation_{selected_shop}_day{start_day}_len{duration}.csv")
    df_result.to_csv(save_path, index=False)
    st.sidebar.success(f"Saved to {save_path}")
