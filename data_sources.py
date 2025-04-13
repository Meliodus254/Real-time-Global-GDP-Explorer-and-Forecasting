import requests
import pandas as pd

# --- World Bank GDP ---
def fetch_worldbank_gdp(country_code, indicator="NY.GDP.MKTP.CD"):
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&per_page=100"
    res = requests.get(url)
    data = res.json()
    if not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame()
    records = data[1]
    df = pd.DataFrame.from_records(records)[["date", "value"]]
    df.columns = ["Year", "GDP"]
    return df.dropna().astype({"Year": int, "GDP": float}).sort_values("Year")

# --- IMF API (dummy URL placeholder) ---
def fetch_imf_data():
    # IMF provides XML/SDMX structure. Integration requires more parsing (can use pandasdmx or sdmx1).
    # Placeholder: Return a mock dataset
    return pd.DataFrame({
        "Indicator": ["Inflation", "Interest Rate", "Unemployment"],
        "Kenya": [6.1, 12.5, 5.6],
        "Nigeria": [10.3, 14.1, 7.5],
        "South Africa": [5.7, 7.0, 4.9]
    })

# --- Trading Economics API ---
def fetch_te_gdp(api_key, country):
    url = f"https://api.tradingeconomics.com/historical/country/{country}/indicator/gdp?c={api_key}"
    res = requests.get(url)
    if res.status_code != 200:
        return pd.DataFrame()
    data = res.json()
    df = pd.DataFrame(data)
    df = df[["Date", "Value"]].rename(columns={"Date": "Year", "Value": "GDP"})
    df["Year"] = pd.to_datetime(df["Year"]).dt.year
    return df.groupby("Year")["GDP"].mean().reset_index()
