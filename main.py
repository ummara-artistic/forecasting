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

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import json

# Set Streamlit Page Config
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

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
    df["year_month"] = df["txndate"].dt.to_period("M").astype(str)

    # KPIs
    total_stock = df["qty"].sum()
    total_value = df["stockvalue"].sum()
    avg_stock_age = df[["aging_60", "aging_90", "aging_180", "aging_180plus"]].sum(axis=1).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Total Stock Quantity", f"{total_stock:,}")
    col2.metric("💰 Total Inventory Value", f"${total_value:,.2f}")
    col3.metric("⏳ Avg Stock Age", f"{avg_stock_age:.1f} days")

    # Aging Analysis Graph
   # Aggregate aging data
    aging_df = df.groupby('txndate')[["aging_60", "aging_90", "aging_180", "aging_180plus"]].sum()

    # Fit SARIMAX Model
    model = sm.tsa.statespace.SARIMAX(
        aging_df["aging_60"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )
    result = model.fit()

    # Forecast next 12 periods
    forecast_steps = 12
    forecast = result.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(aging_df.index[-1], periods=forecast_steps + 1, freq='M')[1:]
    forecast_values = forecast.predicted_mean

    # Convert forecast to DataFrame
    forecast_df = pd.DataFrame({"txndate": forecast_index, "aging_60": forecast_values})
    forecast_df.set_index("txndate", inplace=True)

    # Combine actual and forecast data
    full_data = pd.concat([aging_df, forecast_df])
    full_data.reset_index(inplace=True)

    # Plot aging analysis with forecast
    fig_aging = px.line(
        full_data, x="txndate", y="aging_60",
        title="Aging Distribution with Forecast",
        labels={"txndate": "Date", "aging_60": "Stock Count"},
        markers=True
    )
    fig_aging.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='white')

    st.subheader("📈 Aging Analysis with Forecast")
    st.plotly_chart(fig_aging)




    # 1 Inventory Position Donut Chart
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
        color_discrete_sequence=["pink", "green", "purple"]
    )
    col1.plotly_chart(fig)

    # 2 Inventory at Lead Time Donut Chart
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
        color_discrete_sequence=["green", "green", "purple"]
    )
    col2.plotly_chart(fig)

    # 3 Usage Pattern Types Donut Chart
    col3.subheader("📊 Usage Pattern Types")
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
        color_discrete_sequence=["sea green", "purple", "purple"]
    )
    col3.plotly_chart(fig)

    # ML Model 1: Logistic Regression for Risk Classification
  
    df["risk_category"] = np.select(
        [
            df["lead_time_stock"] > 50,
            (df["lead_time_stock"] > 10) & (df["lead_time_stock"] <= 50),
            df["lead_time_stock"] <= 10,
        ],
        ["Safe", "At Risk", "Critical"],
        default="Unknown",
    )

    features = df[["lead_time_stock", "qty", "daily_demand"]]
    labels = df["risk_category"].map({"Safe": 0, "At Risk": 1, "Critical": 2})

    model_lr = LogisticRegression()
    model_lr.fit(features, labels)

    df["predicted_risk"] = model_lr.predict(features)
    df["predicted_risk"] = df["predicted_risk"].map({0: "Safe", 1: "At Risk", 2: "Critical"})



    # ML Model 2: Random Forest Classifier for Risk Classification
    st.subheader("🌲 Inventory Risk Classification ")
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(features, labels)

    df["rf_predicted_risk"] = model_rf.predict(features)
    df["rf_predicted_risk"] = df["rf_predicted_risk"].map({0: "Safe", 1: "At Risk", 2: "Critical"})

    fig_rf = px.scatter(
        df,
        x="qty",
        y="lead_time_stock",
        color="rf_predicted_risk",
        title="Random Forest Predicted Inventory Risk",
        labels={"rf_predicted_risk": "Risk Level"},
    )
    st.plotly_chart(fig_rf)

    # ML Model 3: XGBoost for Stock Demand Forecasting
    st.subheader("📈 Stock Demand Forecasting (XGBoost)")
    df["month"] = df["txndate"].dt.month
    df["year"] = df["txndate"].dt.year

    demand_features = df[["month", "year", "qty", "lead_time_stock"]]
    demand_labels = df["daily_demand"]

    X_train, X_test, y_train, y_test = train_test_split(demand_features, demand_labels, test_size=0.2, random_state=42)

    model_xgb = XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model_xgb.fit(X_train, y_train)

    df["predicted_demand"] = model_xgb.predict(demand_features)

    fig_xgb = px.line(
        df,
        x="txndate",
        y=["daily_demand", "predicted_demand"],
        title="Actual vs Predicted Demand",
        labels={"value": "Demand", "txndate": "Date"},
    )
    st.plotly_chart(fig_xgb)

    # Monthly Stock Trends
    col1.subheader("📅 Monthly Stock Trends")
    monthly_df = df.groupby("year_month")["qty"].sum().reset_index()
    fig = px.line(monthly_df, x="year_month", y="qty", title="Monthly Stock Trends")
    col1.plotly_chart(fig)





    # Inventory Turnover Rate
    col2.subheader("🔄 Inventory Turnover Rate")
    turnover_df = df.groupby("major")["qty"].sum() / (df.groupby("major")["stockvalue"].sum() + 1)
    fig = px.bar(turnover_df.reset_index(), x="major", y=0, title="Inventory Turnover Rate by Category")
    col2.plotly_chart(fig)




  

    # Demand vs Stock Value
    col3.subheader("🔍 Demand vs Stock Value")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    df.dropna(subset=["qty", "stockvalue"], inplace=True)
    rf.fit(df[["qty"]], df["stockvalue"])
    df["predicted_stockvalue"] = rf.predict(df[["qty"]])
    fig = px.scatter(df, x="qty", y="stockvalue", title="Stock Value vs. Demand", trendline="ols", width=800, height=400)
    col3.plotly_chart(fig)


    
except FileNotFoundError:
    st.error(f"File not found: {file_path}. Please check the path and try again.")
