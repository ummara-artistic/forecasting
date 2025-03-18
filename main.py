import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

# Set Streamlit Page Config
st.set_page_config(layout="wide")
st.title("📊 Supply Chain Forecasting Dashboard")


# Load Data from JSON
file_path = r"D:\stock_forecasting\data\cust_stock.json"

try:
    with open(file_path, "r") as file:
        data = json.load(file)
    df = pd.DataFrame(data["items"])
    df["txndate"] = pd.to_datetime(df["txndate"])
    df = df.sort_values("txndate")

    # Ensure Required Columns Exist
    if "daily_demand" not in df.columns:
        
        df["daily_demand"] = df["qty"] / 30  # Default estimation (monthly avg)

    if "leadtime" not in df.columns:
        df["leadtime"] = 5  # Assign a default lead time

    df["lead_time_stock"] = df["qty"] - (df["daily_demand"] * df["leadtime"])

    # Create 'year_month' Column
    df["year_month"] = df["txndate"].dt.to_period("M").astype(str)

    # KPIs
    total_stock = df["qty"].sum()
    total_value = df["stockvalue"].sum()
    avg_stock_age = df[["aging_60", "aging_90", "aging_180", "aging_180plus"]].sum(axis=1).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Total Stock Quantity", f"{total_stock:,}")
    col2.metric("💰 Total Inventory Value", f"${total_value:,.2f}")
    col3.metric("⏳ Avg Stock Age", f"{avg_stock_age:.1f} days")

    col1, col2 = st.columns(2)

    # Inventory Position Donut Chart
    col1.subheader("📊 Current Inventory Position")
    position_counts = {
        "Excess": (df["qty"] > 100).sum(),
        "Out of Stock": (df["qty"] == 0).sum(),
        "Below Panic Point": ((df["qty"] < 10) & (df["qty"] > 0)).sum(),
    }
    fig = px.pie(
        names=list(position_counts.keys()),
        values=list(position_counts.values()),
        title="Stock Status Distribution",
    )
    col1.plotly_chart(fig)

    # Inventory at Lead Time Donut Chart
    col2.subheader("📊 Inventory Position at Lead Time")
    leadtime_counts = {
        "Safe": (df["lead_time_stock"] > 50).sum(),
        "At Risk": ((df["lead_time_stock"] > 10) & (df["lead_time_stock"] <= 50)).sum(),
        "Critical": (df["lead_time_stock"] <= 10).sum(),
    }
    fig = px.pie(
        names=list(leadtime_counts.keys()),
        values=list(leadtime_counts.values()),
        title="Expected Stock at Lead Time",
    )
    col2.plotly_chart(fig)

    # Usage Pattern Types Donut Chart
    col1.subheader("📊 Usage Pattern Types")
    df["usage_type"] = np.select(
        [
            df["qty"] == 0,
            df["qty"] < 10,
            (df["qty"] >= 10) & (df["qty"] < 50),
            df["qty"] >= 50,
        ],
        ["Dead", "Slow", "Sporadic", "Recurring"],
        default="New",
    )
    usage_counts = df["usage_type"].value_counts()
    fig = px.pie(
        names=usage_counts.index,
        values=usage_counts.values,
        title="Consumption Patterns",
    )
    col1.plotly_chart(fig)

    # Time Series Forecasting
    col2.subheader("📈 Time Series Forecasting")
    model = ARIMA(df["qty"], order=(5, 1, 0))
    model_fit = model.fit()
    df["forecast"] = model_fit.predict(start=len(df) - 10, end=len(df) + 10, dynamic=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["txndate"], df["qty"], label="Actual")
    ax.plot(df["txndate"], df["forecast"], label="Forecast", linestyle="dashed")
    ax.set_title("Inventory Forecast")
    ax.legend()
    col2.pyplot(fig)

    # Aging Analysis
    col1.subheader("📊 Aging Analysis")
    aging_columns = ["aging_60", "aging_90", "aging_180", "aging_180plus"]
    aging_df = df[aging_columns].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    aging_df.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Inventory Aging Analysis")
    col1.pyplot(fig)

    # Category-Wise Inventory
    col2.subheader("📊 Category-Wise Inventory")
    category_df = df.groupby("major")["qty"].sum().reset_index()
    fig = px.bar(category_df, x="major", y="qty", title="Stock Distribution by Category", text="qty", width=800, height=400)
    col2.plotly_chart(fig)

    # Demand vs Stock Value
    col1.subheader("🔍 Demand vs Stock Value")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    df.dropna(subset=["qty", "stockvalue"], inplace=True)
    rf.fit(df[["qty"]], df["stockvalue"])
    df["predicted_stockvalue"] = rf.predict(df[["qty"]])
    fig = px.scatter(df, x="qty", y="stockvalue", title="Stock Value vs. Demand", trendline="ols", width=800, height=400)
    col1.plotly_chart(fig)

    # Seasonal Trends
    col2.subheader("🌡️ Seasonal Trends (Clustering)")
    df["month"] = df["txndate"].dt.month
    scaler = StandardScaler()
    X = scaler.fit_transform(df[["qty", "stockvalue"]])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)
    fig = px.scatter(df, x="month", y="qty", color=df["cluster"].astype(str), title="Seasonal Demand Clusters", width=800, height=400)
    col2.plotly_chart(fig)

    # Monthly Stock Trends
    col1.subheader("📅 Monthly Stock Trends")
    monthly_df = df.groupby("year_month")["qty"].sum().reset_index()
    fig = px.line(monthly_df, x="year_month", y="qty", title="Monthly Stock Trends", width=800, height=400)
    col1.plotly_chart(fig)

    # Inventory Turnover Rate
    col2.subheader("🔄 Inventory Turnover Rate")
    turnover_rate = df.groupby("major")["qty"].sum() / (df.groupby("major")["stockvalue"].sum() + 1)
    turnover_df = turnover_rate.reset_index()
    fig = px.bar(turnover_df, x="major", y=0, title="Inventory Turnover Rate by Category", width=800, height=400)
    col2.plotly_chart(fig)

except FileNotFoundError:
    st.error(f"File not found: {file_path}. Please check the path and try again.")
