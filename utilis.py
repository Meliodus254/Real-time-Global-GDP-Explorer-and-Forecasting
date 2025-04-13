import requests
import pandas as pd
import pycountry

def get_gdp_data(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        if not country:
            return None
        country_code = country.alpha_3
        url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/NY.GDP.MKTP.CD?format=json&per_page=1000"
        response = requests.get(url)
        if response.status_code != 200:
            return None
        data = response.json()[1]
        df = pd.DataFrame(data)[['date', 'value']]
        df.columns = ['Year', 'GDP']
        df = df.dropna()
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')
        df = df.sort_values('Year')
        return df
    except Exception as e:
        print(f"Error fetching {country_name}: {e}")
        return None
