import streamlit as st
import pandas as pd
import plotly.express as px
from utilis import get_gdp_data
from forecast import forecast_prophet, forecast_lstm

st.set_page_config(layout='wide', page_title='Global GDP Explorer 🌍')
st.title("📊 Real-time Global GDP Explorer and Forecasting")

# --- Sidebar ---
st.sidebar.header("Configuration")
countries_df = pd.read_csv("data/countries.csv")
countries = countries_df['Country'].tolist()
selected_countries = st.sidebar.multiselect("Select Countries", countries, default=["Kenya", "Nigeria", "India"])

model_choice = st.sidebar.selectbox("Forecast Model", ["Prophet", "LSTM"])
forecast_years = st.sidebar.slider("Years to Predict", min_value=1, max_value=10, value=5)

# --- Data Fetching ---
all_data = []
for country in selected_countries:
    df = get_gdp_data(country)
    if df is not None and not df.empty:
        latest = df.iloc[-1:].copy()
        latest["Country"] = country
        all_data.append(latest)

# --- Choropleth Map ---
if all_data:
    map_df = pd.concat(all_data)
    map_df["GDP"] = map_df["GDP"] / 1e9  # Convert to billions for better readability
    map_fig = px.choropleth(
        map_df,
        locations="Country",
        locationmode="country names",
        color="GDP",
        hover_name="Country",
        color_continuous_scale="Blues",
        labels={"GDP": "GDP (Billion USD)"}
    )
    map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(map_fig, use_container_width=True)

    # --- Forecast & Trend ---
    for country in selected_countries:
        df = get_gdp_data(country)
        if df is None or df.empty:
            st.warning(f"No GDP data for {country}")
            continue

        st.subheader(f"📈 {country} GDP Trend & Forecast")
        
        # Convert GDP to USD Billions for readability
        df_display = df.copy()
        df_display["GDP"] = df_display["GDP"] / 1e9
        st.line_chart(df_display.set_index("Year")["GDP"])

        if model_choice == "Prophet":
            forecast = forecast_prophet(df, periods=forecast_years)
        else:
            forecast = forecast_lstm(df, periods=forecast_years)

        forecast["yhat"] = forecast["yhat"] / 1e9  # Also in billions
        forecast_fig = px.line(forecast, x='ds', y='yhat', title=f"{country} Forecasted GDP (Billions USD)")
        forecast_fig.update_traces(mode='lines+markers')
        st.plotly_chart(forecast_fig, use_container_width=True)

else:
    st.error("No data to display. Please select valid countries.")
