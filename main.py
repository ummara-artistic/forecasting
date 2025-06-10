import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.express as px
import streamlit as st

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="🎨 Stock Forecasting Analysis", layout="wide")
with st.expander("📍  Dashboard Overview"):
    st.markdown("""
   We did data processing use txndate , chnge to utc format 
sort values through sort value builtin function sort the rows of data frame by column name
inplace=true, sort wil modify the original dataframe and return a new sorted data frame
define default value for lead time
for lead time stock , we get the formula of =  qty - ( daily demand * leadtime)
for daily demand = qty / 30 

For forecasting we filter data by 2024 only ing
        - Turnover rates
    """)




# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: white;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
    color: white !important;
}
.css-1d391kg input[type="radio"]:checked + div {
    background: #ff7e5f !important;
    color: white !important;
    border-radius: 8px;
    font-weight: 700;
}
.css-1d391kg div[role="radio"]:hover {
    background-color: rgba(255, 126, 95, 0.3);
    border-radius: 8px;
}
.metric-container {
    background-color: white; 
    border-radius: 10px; 
    padding: 20px; 
    box-shadow: 0 0 10px rgb(0 0 0 / 0.1);
    margin-bottom: 15px;
}
h2, h3 {
    color: #2575fc;
}
footer {
    text-align: center;
    color: #999;
    padding: 10px;
}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)






# ------------------- HEADER -------------------
st.markdown("<h1 style='text-align: center; color: #2575fc;'> Stock Forecasting Analysis</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border-top: 3px solid #2575fc;'>", unsafe_allow_html=True)


# ------------------- LOAD AND PREPROCESS DATA -------------------
# ----------- Load Data -----------



@st.cache_data(show_spinner=False)
def load_data(file_path):
    with open(file_path, "r") as f:
        raw_data = json.load(f)
    return pd.DataFrame(raw_data["items"])

# Define the file path
file_path = os.path.join(os.getcwd(), "data", "cust_stock.json")

# Check if file exists
if not os.path.exists(file_path):
    st.error("File not found: cust_stock.json")
    st.stop()

# Load data using the cached function
df = load_data(file_path)


# Preprocessing
df["txndate"] = pd.to_datetime(df["txndate"], utc=True)
df.sort_values("txndate", inplace=True)
df["daily_demand"] = df["qty"] / 30
df["leadtime"] = 5
df["lead_time_stock"] = df["qty"] - (df["daily_demand"] * df["leadtime"])
df["year"] = df["txndate"].dt.year
df["year_month"] = df["txndate"].dt.to_period("M").astype(str)
df["month"] = df["txndate"].dt.month
df["description"] = df["description"].astype(str)

filtered_data = df[(df["txndate"].dt.year == 2024)]
# Automatically filter data for 2024 only


#----------------------------------------------Navigation side Bar------------------------------------------
st.sidebar.title("📦 Inventory Aging Forecast (2025)")
with st.sidebar.expander("📊 Store behind this ", expanded=True):
    st.markdown("""

- 🟢 **Fresh stock (≤ 60 days):** {aging_60} units are still relatively new and moving well.
- 🟠 **Moderately aged (61-90 days):** {aging_90} units are approaching a moderate age and might need attention.
- 🔴 **Aged stock (91-180 days):** {aging_180} units are getting old — consider strategies to clear them.
- ⚠️ **Very old stock (>180 days):** {aging_180plus} units have been sitting too long and may lead to losses.

    """)
# Group by month and sum aging buckets
df["year_month"] = df["txndate"].dt.to_period("M").astype(str)
monthly_aging = df.groupby("year_month")[["aging_60", "aging_90", "aging_180", "aging_180plus"]].sum().reset_index()

# Convert year_month to numeric (e.g., 2022-08 → 202208)
monthly_aging["ym_numeric"] = monthly_aging["year_month"].str.replace("-", "").astype(int)

# Function to predict aging values for 2025
def forecast_aging(feature):
    X = monthly_aging[["ym_numeric"]]
    y = monthly_aging[feature]
    model = LinearRegression()
    model.fit(X, y)

    forecast_months = [202501, 202502, 202503, 202504, 202505, 202506,
                       202507, 202508, 202509, 202510, 202511, 202512]
    future_df = pd.DataFrame({"ym_numeric": forecast_months})
    future_df[feature] = model.predict(future_df[["ym_numeric"]])
    return future_df[[feature]].sum().values[0]  # Return total for 2025

# Forecast totals for each aging bucket
pred_60 = round(forecast_aging("aging_60"), 2)
pred_90 = round(forecast_aging("aging_90"), 2)
pred_180 = round(forecast_aging("aging_180"), 2)
pred_180plus = round(forecast_aging("aging_180plus"), 2)
total_pred = pred_60 + pred_90 + pred_180 + pred_180plus

# Show predictions in sidebar
st.sidebar.subheader("🔮 Predicted Aging for 2025")
st.sidebar.markdown(f"- **Aged ≤ 60 days**: {pred_60} units")
st.sidebar.markdown(f"- **Aged ≤ 90 days**: {pred_90} units")
st.sidebar.markdown(f"- **Aged ≤ 180 days**: {pred_180} units")
st.sidebar.markdown(f"- **Aged > 180 days**: {pred_180plus} units")
st.sidebar.markdown(f"---\n- **Total Forecast Qty (2025)**: {round(total_pred)} units")


with st.expander("KPIS"):
    st.markdown("""
    #for KPI we have decided to move with random forest as we have to predict 
                kpis based on the historical data before 2024 and 2024 as well, 
                in random forest we select features , then use features for test and train for training,
                for the model we selected feature , do prediction and get mean value as well
    **
    """)

#----------------------------------------------------For KPIS-----------------------------------

# Filter the data
df_2024 = df[df["year"] == 2024]
train_df = df[df["year"] < 2024]
test_df = df_2024.copy()

# Features to use
features = ["qty", "daily_demand", "lead_time_stock"]

# Train and predict Quantity
model_qty = RandomForestRegressor(random_state=42)
model_qty.fit(train_df[features], train_df["qty"]) 
pred_qty_2025 = model_qty.predict(test_df[features]).sum()

# Train and predict Daily Demand
model_demand = RandomForestRegressor(random_state=42)
model_demand.fit(train_df[features], train_df["daily_demand"])
pred_demand_2025 = model_demand.predict(test_df[features]).mean()

# Train and predict Lead Time Stock
model_lead = RandomForestRegressor(random_state=42)
model_lead.fit(train_df[features], train_df["lead_time_stock"])
pred_lead_2025 = model_lead.predict(test_df[features]).mean()



k1, k2, k3, k4 = st.columns(4)
k1.metric("🛒 Predicted Total Quantity 2025", f"{pred_qty_2025:.0f}")
k2.metric("📊 Predicted Avg Daily Demand 2025", f"{pred_demand_2025:.2f}")
k3.metric("⏳ Predicted Avg Lead Time Stock 2025", f"{pred_lead_2025:.2f}")



#----------------------------------Graph -------------------------------


predicted_dates = pd.date_range("2025-01-01", "2025-12-01", freq="MS")

# 1st Row Charts - three columns in one line
r1c1, r1c2, r1c3 = st.columns(3)

# --- Preprocessing ---
filtered_data["month"] = filtered_data["txndate"].dt.to_period("M")
grouped = filtered_data.groupby(["month", "description", "stockvalue"])["qty"].sum().reset_index()
grouped["month"] = grouped["month"].dt.to_timestamp()

# --- Linear Regression Forecasting ---
predicted_df_list = []
unique_combinations = grouped[["description", "stockvalue"]].drop_duplicates()

for _, row in unique_combinations.iterrows():
    desc = row["description"]
    stock = row["stockvalue"]

    subset = grouped[(grouped["description"] == desc) & (grouped["stockvalue"] == stock)]
    subset = subset.set_index("month").asfreq("MS")  # Ensure monthly frequency

    # Fill missing values if needed
    subset["qty"] = subset["qty"].fillna(method='ffill')

    # Prepare data for Linear Regression
    subset = subset.reset_index()
    subset["time_index"] = np.arange(len(subset))  # 0,1,2,...

    X = subset[["time_index"]]
    y = subset["qty"]

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    # Forecast for next 12 months
    future_time_index = np.arange(len(subset), len(subset) + 12)
    future_qty = model.predict(future_time_index.reshape(-1, 1))

    temp_df = pd.DataFrame({
        "txndate": predicted_dates,
        "qty": future_qty,
        "description": desc,
        "stockvalue": stock
    })
    predicted_df_list.append(temp_df)

predicted_df = pd.concat(predicted_df_list, ignore_index=True)
predicted_df["txndate"] = pd.to_datetime(predicted_df["txndate"])

# --- Combine historical and forecast data ---
historical_df = grouped.rename(columns={"month": "txndate"})
combined_df = pd.concat([historical_df, predicted_df])

# --- Plotting ---
with r1c1:
    st.header("Quantity Forecast by Description and Stockvalue")
   
    fig = px.line(combined_df, x="txndate", y="qty", color="description",
                  line_dash="stockvalue", markers=True, template="plotly_dark",
                  labels={"qty": "Quantity", "txndate": "Date"}
                )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📘 Story behind this graph"):
        st.write("""
        This line chart shows the combined forecasted quantity for all items, aggregated monthly for 2025,
        generated by a simple Linear Regression model. It captures the linear trend of demand without 
        considering seasonality or complex time-series patterns.
        """)


#for second chart

# --- Lead Time Stock Forecasting (Month-wise) ---

monthly_leadtime_avg = filtered_data.groupby(["month", "major"])["lead_time_stock"].mean().reset_index()

leadtime_predicted_data = []
for _, row in monthly_leadtime_avg.iterrows():
    for date in predicted_dates:
        if date.month == row["month"].month:
            leadtime_predicted_data.append({
                "txndate": date,
                "lead_time_stock": row["lead_time_stock"],
                "major": row["major"]
            })

leadtime_predicted_df = pd.DataFrame(leadtime_predicted_data)

with r1c2:
    st.header("Forecasted Lead Time Stock by Major for 2025")

    fig = px.scatter(
        leadtime_predicted_df,
        x="txndate",
        y="lead_time_stock",
        color="major",
        symbol="major",
        template="plotly_dark",
        labels={
            "lead_time_stock": "Lead Time Stock",
            "txndate": "Month",
            "major": "Major"
        },
        title="Monthly Forecasted Lead Time Stock for Each Major Category (2025)"
    )

    fig.update_traces(marker=dict(size=10))  # adjust marker size if needed
    fig.update_layout(xaxis=dict(dtick="M1", tickformat="%b\n%Y"))  # Month format

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📘 Story behind this graph"):
        st.write("""
        This scatter chart displays the **forecasted average lead time stock for each major category, month-wise in 2025**.

        - Each point represents the **lead time stock requirement for a specific major in a specific month**.
        - This visualization helps in understanding how stock requirements vary month-to-month for each major.
        - The forecast is based on historical monthly averages for each corresponding month.
        """)


# --- Forecasting section for selected description with SARIMAX ---
# ------------------- FORECASTING FUNCTIONS -------------------



# Function to perform Exponential Smoothing Forecasting
def ets_forecast(ts, steps=365):
    try:
        model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps)
        return forecast
    except Exception as e:
        st.error(f"ETS model error: {e}")
        return pd.Series([np.nan]*steps)

with r1c3:
    st.header("📈 Quantity Forecasting for 2025")

    # Assume df is pre-loaded DataFrame
    descriptions = df["description"].unique()
    df['txndate'] = pd.to_datetime(df['txndate'])

    # Small filters in one row
    col1, col2, col3 = st.columns([1, 1, 2])  # Adjust widths as needed

    with col1:
        selected_desc = st.selectbox("Item", sorted(descriptions), key="forecast_desc", label_visibility="collapsed")
    with col2:
        month_options = [f"2025-{str(m).zfill(2)}" for m in range(1, 13)]
        selected_month = st.selectbox("Month", month_options, label_visibility="collapsed")
    with col3:
        period_option = st.radio("Period", ["Full Month", "First 7 Days", "First 15 Days"], horizontal=True)

    # Expander Info
    with st.expander("📘 About this Forecast"):
        st.markdown("""
        - Forecasts daily quantity for 2025 using Exponential Smoothing (ETS).
        - Select item, month, and period.
        """)

    df_desc = df[df["description"] == selected_desc]

    ts_monthly = df_desc.groupby(df_desc["txndate"].dt.to_period("M")).agg({"qty": "sum"})["qty"]
    ts_monthly.index = ts_monthly.index.to_timestamp()

    if len(ts_monthly) < 24:
        st.warning("At least 24 months of data required for reliable forecast.")
    else:
        forecast_days = 365
        ets_pred = ets_forecast(ts_monthly, steps=forecast_days)

        last_date = ts_monthly.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=forecast_days, freq='D')

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecasted Quantity": ets_pred.values
        })

        sel_year, sel_month = map(int, selected_month.split('-'))
        mask = (forecast_df['Date'].dt.year == sel_year) & (forecast_df['Date'].dt.month == sel_month)
        month_forecast = forecast_df.loc[mask]

        if month_forecast.empty:
            st.warning("No forecast data available for the selected month.")
        else:
            if period_option == "First 7 Days":
                period_forecast = month_forecast.iloc[:7]
            elif period_option == "First 15 Days":
                period_forecast = month_forecast.iloc[:15]
            else:  # Full Month
                period_forecast = month_forecast

            fig = px.line(
                period_forecast,
                x="Date",
                y="Forecasted Quantity",
                markers=True,
                title=f"Forecasted Quantity for {selected_month} - {period_option}",
                template="plotly_dark"
            )
            fig.update_traces(texttemplate='%{y:.0f}', textposition='top center')
            fig.update_layout(showlegend=False)

            st.plotly_chart(fig, use_container_width=True)


      

# ------------------- 2nd ROW: Additional 6 Graphs -------------------



def linear_regression_forecast(ts, steps=12):
    """
    Forecast using Linear Regression.
    ts: pandas Series with DateTimeIndex.
    """
    try:
        ts = ts.dropna()
        if len(ts) < 2:  # Not enough data to build model
            raise ValueError("Not enough data points for Linear Regression")
        
        # Convert time index to integer (for fitting)
        X = np.arange(len(ts)).reshape(-1, 1)   # time step as feature
        y = ts.values.reshape(-1, 1)           # target qty

        # Fit Linear Regression
        model = LinearRegression()
        model.fit(X, y)

        # Predict next 'steps' months
        future_X = np.arange(len(ts), len(ts) + steps).reshape(-1, 1)
        forecast = model.predict(future_X).flatten()

        # Ensure no negative predictions
        forecast = np.where(forecast < 0, 0, forecast)

        # Build forecast index
        index = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(),
                              periods=steps, freq='MS')
        return pd.Series(forecast, index=index)
    
    except Exception as e:
        print(f"Forecasting error: {e}")
        return pd.Series([np.nan] * steps)


def forecast_all_descriptions(df, steps=12):
    forecast_data = {}
    desc_list = df["description"].unique()
    for d in desc_list:
        df_d = df[df["description"] == d]
        ts = df_d.groupby(df_d["txndate"].dt.to_period("M")).agg({"qty": "sum"})["qty"]
        ts.index = ts.index.to_timestamp()

        if len(ts) < 3:  # need at least 3 months to fit regression
            continue

        fcast = linear_regression_forecast(ts, steps=steps)
        forecast_data[d] = fcast.sum()  # total predicted qty for next 12 months
    return forecast_data


# Forecast total qty per description for 2025
forecast_all = forecast_all_descriptions(df)
top_10 = sorted(forecast_all.items(), key=lambda x: x[1], reverse=True)[:10]
top_10_df = pd.DataFrame(top_10, columns=["Description", "Predicted_Total_Qty_2025"])

# 2025 turnover prediction
df["year"] = df["txndate"].dt.year
qty_before_2025 = df[df["year"] < 2025].groupby("description")["qty"].sum()
turnover_pred_2025 = {}

for desc, pred_qty in forecast_all.items():
    prev_qty = qty_before_2025.get(desc, 0)
    turnover_pred_2025[desc] = pred_qty / (prev_qty + 1)  # +1 to avoid division by zero

turnover_pred_2025 = sorted(turnover_pred_2025.items(), key=lambda x: x[1], reverse=True)[:10]
turnover_pred_2025_df = pd.DataFrame(turnover_pred_2025, columns=["Description", "Predicted_Turnover_2025"])







df["month_index"] = (df["txndate"].dt.year - 2020) * 12 + df["txndate"].dt.month  # use as numeric feature
df["description"] = df["description"].astype(str)

# Filter from 2022 onwards if you want clean trends
df = df[df["txndate"].dt.year >= 2022]

# Group by description and month
monthly_data = df.groupby(["description", "month_index"]).agg({"qty": "sum"}).reset_index()



# Forecast function
def forecast_qty_linear_reg(item_df):
    model = LinearRegression()
    X = item_df[["month_index"]]
    y = item_df["qty"]
    model.fit(X, y)

    # Predict for 2025 (months 61 to 72)
    future_months = pd.DataFrame({"month_index": range(61, 73)})
    future_qty = model.predict(future_months)
    total_pred_2025 = future_qty.sum()
    return total_pred_2025

# Forecast for each item
forecast_results = []
for item in monthly_data["description"].unique():
    item_df = monthly_data[monthly_data["description"] == item]
    predicted_qty = forecast_qty_linear_reg(item_df)
    forecast_results.append({"description": item, "predicted_2025_qty": predicted_qty})

forecast_df = pd.DataFrame(forecast_results).sort_values(by="predicted_2025_qty", ascending=False)




# ==== Define Layout ====
row1_col1, row1_col2, row1_col3 = st.columns(3)

# ==== Select Filters ====
col1, col2 = st.columns(2)
aging_options = ['aging_60', 'aging_90', 'aging_180', 'aging_180plus']
aging_labels = {
    'aging_60': 'Aging > 60 Days',
    'aging_90': 'Aging > 90 Days',
    'aging_180': 'Aging > 180 Days',
    'aging_180plus': 'Aging > 180+ Days'
}


# ==== Load Data ====
df_2024['date'] = pd.date_range(start='2024-01-01', periods=len(df_2024), freq='MS')



# Assume df_2024, aging_options, aging_labels are predefined
# Display the header above the chart


with row1_col1:
    # Nested columns for selectboxes side by side, smaller width
    st.header(f"Aging Forecast for 2025")
    sel_col1, sel_col2 = st.columns([2, 2])  # adjust ratios for width
    
    with sel_col1:
        descriptions = ['All'] + sorted(df_2024['description'].dropna().unique().tolist())
        selected_desc = st.selectbox("Description", descriptions)

    with sel_col2:
        selected_aging = st.selectbox(
            "Aging Category", 
            aging_options, 
            format_func=lambda x: aging_labels[x]
        )

    # Filter and preprocess
    filtered_df = df_2024.copy()
    if selected_desc != "All":
        filtered_df = filtered_df[filtered_df['description'] == selected_desc]

    filtered_df = filtered_df[['date', selected_aging]].dropna()
    monthly_data = filtered_df.groupby('date')[selected_aging].sum().reset_index()

    # Forecast & Plot
    try:
        model = ARIMA(monthly_data[selected_aging], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=12)

        forecast_dates = pd.date_range(start='2025-01-01', periods=12, freq='MS')
        forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast': forecast})
        forecast_df['MonthLabel'] = forecast_df['date'].dt.strftime('%b %Y')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['MonthLabel'],
            y=forecast_df['forecast'],
            mode='lines+markers',
            name='2025 Forecast',
            line=dict(color='orange', width=3)
        ))


        # Update Plotly figure layout
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Predicted Aging Value',
            plot_bgcolor='white',
            hovermode='x unified',
            height=500
        )

        # Show the chart
        st.plotly_chart(fig, use_container_width=True)

        # Explanation / Storyboard
        with st.expander("📘 Story behind this", expanded=False):
            st.markdown(f"""
            - The graph displays a **12-month forecast** of the selected aging category (e.g., **{aging_labels[selected_aging]}**) for the year **2025**.
            - Forecasting is based on historical monthly totals from your filtered selection.

            ### How is the aging value calculated?
            1. Data is filtered based on your selected **description**.
            2. Only the selected **aging column** is considered (e.g., aging_60).
            3. Missing values are dropped.
            4. The remaining data is **grouped by month**, and the **sum of aging values** is calculated.

            ### Forecasting Model Used
            - We use the **ARIMA (AutoRegressive Integrated Moving Average)** model to project values for the next 12 months.
            - ARIMA analyzes past trends and fluctuations in your data to make accurate future predictions.

            ### What does `{selected_aging}` mean?
            - This aging category tracks records/items that are approximately:
              - **`aging_30`**: 0–29 days old
              - **`aging_60`**: 30–59 days old
              - **`aging_90`**: 60–89 days old
              - **`aging_120`**: 90+ days old
            - For example, `aging_60` means the value of records that have been stagnant or unpaid for **30–59 days**.
            """)

    except Exception as e:
        st.error(f"ARIMA prediction failed: {e}")


                

with row1_col2:
    st.header("Top 10 Items Predicted to be Bought in 2025")
    fig = px.bar(top_10_df, x="Description", y="Predicted_Total_Qty_2025",
               
                 template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📘 Story behind this graph"):
            st.write("""
           Focused view of the top 10 items expected to have the highest purchase quantities.
        This visualization allows for targeted inventory planning and prioritization for these key products..
            """)

with row1_col3:
    st.header("2025 Predicted Top 10 Items Share")
    fig = px.pie(top_10_df, names="Description", values="Predicted_Total_Qty_2025",
               template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📘 Story behind this graph"):
        st.write("""
        This pie chart represents the proportion of total predicted quantities attributed to each
        of the top 10 items, helping visualize their relative importance in the 2025 forecast.
        """)




# Forecast helper function


# --- Forecast Functions ---
def linear_regression_forecast(ts, steps=12):
    try:
        ts = ts.dropna()
        if len(ts) < 2:
            raise ValueError("Not enough data points for Linear Regression")
        X = np.arange(len(ts)).reshape(-1, 1)
        y = ts.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(ts), len(ts) + steps).reshape(-1, 1)
        forecast = model.predict(future_X).flatten()
        forecast = np.where(forecast < 0, 0, forecast)
        index = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq='MS')
        return pd.Series(forecast, index=index)
    except Exception as e:
        print(f"Linear Regression error: {e}")
        return pd.Series([np.nan] * steps)

def exponential_smoothing_forecast(ts, steps=12):
    try:
        ts = ts.dropna()
        if len(ts) < 2:
            raise ValueError("Not enough data points for SES")
        model = SimpleExpSmoothing(ts).fit(optimized=True)
        forecast = model.forecast(steps)
        forecast = np.where(forecast < 0, 0, forecast)
        index = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq='MS')
        return pd.Series(forecast, index=index)
    except Exception as e:
        print(f"SES error: {e}")
        return pd.Series([np.nan] * steps)

# --- 1. Total Monthly Forecast for 2025 ---
def forecast_monthly_2025(df, steps=12):
    monthly_forecasts = []
    for desc in df["description"].unique():
        df_desc = df[df["description"] == desc]
        ts = df_desc.groupby(df_desc["txndate"].dt.to_period("M")).agg({"qty": "sum"})["qty"]
        ts.index = ts.index.to_timestamp()
        if len(ts) < 3:
            continue
        fcast = linear_regression_forecast(ts, steps=steps)
        monthly_forecasts.append(fcast)
    if monthly_forecasts:
        return pd.concat(monthly_forecasts, axis=1).sum(axis=1)
    else:
        return pd.Series()

# --- Assume df and top_10_df already exist ---
monthly_pred_2025 = forecast_monthly_2025(df)




# 2) Monthly forecast trends for top 5 items
top_5_desc = [desc for desc, _ in top_10[:5]]
monthly_forecasts_top5 = {}
for d in top_5_desc:
    df_d = df[df["description"] == d]
    ts = df_d.groupby(df_d["txndate"].dt.to_period("M")).agg({"qty": "sum"})["qty"]
    ts.index = ts.index.to_timestamp()
    if len(ts) < 3:
        continue
    monthly_forecasts_top5[d] = linear_regression_forecast(ts, steps=12)

# 3) Heatmap for top 10 items monthly forecast
heatmap_data = []
heatmap_index = None
for d in top_10_df["Description"]:
    df_d = df[df["description"] == d]
    ts = df_d.groupby(df_d["txndate"].dt.to_period("M")).agg({"qty": "sum"})["qty"]
    ts.index = ts.index.to_timestamp()
    if len(ts) < 3:
        continue
    fcast = linear_regression_forecast(ts, steps=12)
    heatmap_data.append(fcast.values)
    if heatmap_index is None:
        heatmap_index = fcast.index.strftime('%Y-%m')

heatmap_array = np.array(heatmap_data)


# --- 4. Quantity and Turnover Predictions for Each Item ---
predicted_data = []
for desc in top_10_df["Description"]:
    df_d = df[df["description"] == desc]
    ts_qty = df_d.groupby(df_d["txndate"].dt.to_period("M")).agg({"qty": "sum"})["qty"]
    ts_qty.index = ts_qty.index.to_timestamp()
    if len(ts_qty) < 2:
        continue
    qty_forecast = exponential_smoothing_forecast(ts_qty)
    predicted_total_qty = qty_forecast.sum()

    if "turnover_ratio" in df_d.columns:
        ts_turn = df_d.groupby(df_d["txndate"].dt.to_period("M")).agg({"turnover_ratio": "mean"})["turnover_ratio"]
        ts_turn.index = ts_turn.index.to_timestamp()
        predicted_turnover = exponential_smoothing_forecast(ts_turn).mean() if len(ts_turn) >= 2 else 0
    else:
        predicted_turnover = 0

    predicted_data.append({
        "Description": desc,
        "Predicted_Total_Qty_2025": predicted_total_qty,
        "Predicted_Turnover_2025": predicted_turnover
    })

turnover_pred_2025_df = pd.DataFrame(predicted_data)
turnover_pred_2025_df.rename(columns={"Predicted_Total_Qty_2025": "Predicted_Total_Qty_2025_forecast"}, inplace=True)

# --- Merge & Create Scatter Plot ---
merged_df = turnover_pred_2025_df.merge(top_10_df, on="Description", how="inner")
fig_scatter = px.scatter(
    merged_df,
    x="Predicted_Turnover_2025",
    y="Predicted_Total_Qty_2025_forecast",
    hover_name="Description",
    labels={
        "Predicted_Turnover_2025": "Predicted Turnover Ratio",
        "Predicted_Total_Qty_2025_forecast": "Predicted Qty"
    }
)


# STREAMLIT UI BLOCK
row2_col1, row2_col2, row2_col3 = st.columns(3)

# 1. Total monthly predicted quantity chart
with row2_col1:
    st.header("Total Monthly Predicted Quantity for 2025")
    st.plotly_chart(
        px.line(
            monthly_pred_2025,
            labels={"index": "Month", "value": "Predicted Qty"}
        ),
        use_container_width=True
    )
    with st.expander("📘 Story behind this"):
        st.markdown("""
        This line chart illustrates the **aggregated predicted quantity** for all items in each month of **2025**.

        The forecast was generated using a **Linear Regression model** trained on past transaction data. This helps stakeholders identify expected demand trends across the year and plan inventory accordingly.
        """)



with row2_col2:
    st.header("Qty vs Turnover (Top Items)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# (Optional: you can use `heatmap_array` for heatmap visualization in row2_col3)


# Ensure year_month is datetime
df_fab = df_2024.copy()
df_fab["year_month"] = pd.to_datetime(df_fab["year_month"])
df_fab["month"] = df_fab["year_month"].dt.month
df_fab["year"] = df_fab["year_month"].dt.year

# Group by fabtype and year_month
fab_grouped = df_fab.groupby(["fabtype", "year_month"]).agg({"stockvalue": "sum"}).reset_index()

# List to store predictions
fab_predictions_list = []

# Unique fab types
fab_types = fab_grouped["fabtype"].dropna().unique()

# Loop through each fab type and forecast
for fab in fab_types:
    df_temp = fab_grouped[fab_grouped["fabtype"] == fab].copy().sort_values("year_month")
    df_temp["month"] = df_temp["year_month"].dt.month
    df_temp["year"] = df_temp["year_month"].dt.year

    # Create lag features
    for lag in [1, 2, 3]:
        df_temp[f"lag_{lag}"] = df_temp["stockvalue"].shift(lag)

    df_temp = df_temp.dropna()
    if df_temp.empty:
        continue

    # Prepare training data
    X = df_temp[["month", "lag_1", "lag_2", "lag_3"]]
    y = df_temp["stockvalue"]

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    # Prepare for prediction
    future_months = pd.date_range(start="2025-01-01", end="2025-12-01", freq='MS')
    predictions = []

    # Get last 3 known stock values for lags
    lag_vals = list(df_temp["stockvalue"].iloc[-3:])

    # Generate predictions for each future month
    for i, date in enumerate(future_months):
        month_val = date.month
        x_pred = np.array([[month_val, lag_vals[-1], lag_vals[-2], lag_vals[-3]]])
        pred = model.predict(x_pred)[0]
        predictions.append(pred)
        lag_vals.append(pred)

    # Build prediction DataFrame for current FAB type
    fab_future_df = pd.DataFrame({
        "year_month": future_months,
        "fabtype": fab,
        "predicted_stock_value": predictions
    })
    fab_predictions_list.append(fab_future_df)

# Combine all predictions
fab_predictions_df = pd.concat(fab_predictions_list, ignore_index=True)

# Plotting
with row2_col3:
    st.header("📈 Predicted Stock Value by FAB Type (2025)")
    fig_fab = px.line(
        fab_predictions_df,
        x="year_month",
        y="predicted_stock_value",
        color="fabtype",
        
    )
    st.plotly_chart(fig_fab, use_container_width=True)

    with st.expander("📘 Story behind this"):
        st.markdown("""
        This forecasting system is built to predict monthly stock values for different FAB types for the year 2025. we use random forest, it start by grouping the data by FAB type and 
                    month-year to compute total stock value over time. For each FAB type: We extract the month and year from the date and calculate lag features: stock values from the last 1, 2, and 3 months. These help the model understand short-term memory in time.

A separate Random Forest model is trained for each FAB type using current month, stock values for last three months like lag wise lag1, lag 2, ..
The actual stock value for that month as the target.

Forecasting Future
For each month in 2025, the model:

Takes the last 3 predicted (or actual) values as input.

Predicts the next month’s stock value.


        """)



st.title("📦 Inventory Analytics & Forecasting Dashboard")

# Select item
item_options = filtered_data["description"].unique()
selected_item = st.selectbox("🔍 Select Inventory Item", item_options)

item_data = filtered_data[filtered_data["description"] == selected_item].copy()
item_data = item_data.set_index("txndate").asfreq('D')  # daily freq, fill missing dates
item_data["qty"] = item_data["qty"].fillna(method='ffill')

# Aggregate daily data to monthly for SARIMAX
monthly_data = item_data["qty"].resample('MS').sum()

# SARIMAX Model Forecast
try:
    model = SARIMAX(monthly_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)

    forecast_periods = 12
    forecast = results.get_forecast(steps=forecast_periods)
    forecast_index = pd.date_range(start=monthly_data.index[-1] + pd.offsets.MonthBegin(1), periods=forecast_periods, freq='MS')
    forecast_values = forecast.predicted_mean.values

except Exception as e:
    st.error(f"Forecasting error: {e}")
    forecast_values = [monthly_data.iloc[-1]] * 12  # fallback to stagnant qty
    forecast_index = pd.date_range(start=monthly_data.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq='MS')

# Inventory Value Depreciation Forecast (5% monthly)
latest_stockvalue = item_data["stockvalue"].dropna().iloc[-1]
depreciation_rate = 0.05
depreciated_values = [latest_stockvalue * ((1 - depreciation_rate) ** i) for i in range(forecast_periods)]

# Create Plotly figures

# Graph 1: Inventory Value Depreciation
fig_depreciation = go.Figure()
fig_depreciation.add_trace(go.Scatter(
    x=forecast_index, y=depreciated_values,
    mode='lines+markers',
    name='Inventory Value'
))
fig_depreciation.update_layout(
    title=f"📉 Inventory Value Depreciation (5% Monthly) for '{selected_item}'",
    xaxis_title='Date',
    yaxis_title='Inventory Value',
    template='plotly_white'
)

# Graph 2: Stock Quantity Forecast
fig_stock = go.Figure()
fig_stock.add_trace(go.Scatter(
    x=monthly_data.index, y=monthly_data.values,
    mode='lines+markers',
    name='Historical Qty'
))
fig_stock.add_trace(go.Scatter(
    x=forecast_index, y=forecast_values,
    mode='lines+markers',
    name='Forecasted Qty'
))
fig_stock.update_layout(
    title=f"🚫 Stock Movement Insight & Forecast for '{selected_item}'",
    xaxis_title='Date',
    yaxis_title='Quantity (KGS)',
    template='plotly_white'
)

# Display plots side by side
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_depreciation, use_container_width=True)
with col2:
    st.plotly_chart(fig_stock, use_container_width=True)

# Add explanation expander
with st.expander("📘 Story behind this"):
    st.markdown("""
    This forecasting system is designed to provide actionable insights on inventory management for selected items.

    **Inventory Value Depreciation:**  
    We assume a simple 5% monthly depreciation rate to account for storage costs, spoilage, or obsolescence. This helps forecast the potential reduction in the monetary value of unsold inventory over time.

    **Stock Movement Insight and Forecast:**  
    Using a SARIMAX time series model, we analyze the historical monthly quantity data to predict stock movement trends for the year 2025.  
    - The model captures both seasonality and trends in the stock quantity and forecast data for potential growth or decline

    **How it works:**  
    1. Historical daily data is aggregated monthly.  
    2. SARIMAX fits the historical quantity series to capture trend and seasonality.  
    3. Forecasts are generated for the next 12 months (2025).  
    4. The depreciation forecast is calculated independently, assuming a fixed monthly percentage decrease in stock value.  
    5. Both graphs provide a comprehensive view of inventory health and future outlook.

    Use these insights to optimize inventory orders, pricing, and promotional strategies.
    """)
