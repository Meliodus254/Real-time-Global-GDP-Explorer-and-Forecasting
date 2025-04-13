from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def forecast_prophet(df, periods=5):
    df_prophet = df.rename(columns={'Year': 'ds', 'GDP': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(periods)


def forecast_lstm(df, periods=5):
    df = df.copy()
    df['GDP'] = df['GDP'].astype(float)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df['GDP'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(5, len(data_scaled)):
        X.append(data_scaled[i-5:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    input_seq = data_scaled[-5:].reshape(1, 5, 1)
    predictions = []
    for _ in range(periods):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    preds = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    last_year = df['Year'].max().year
    forecast_years = pd.date_range(start=f"{last_year+1}", periods=periods, freq='Y')

    return pd.DataFrame({'ds': forecast_years, 'yhat': preds})
