import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Download the stock dataset for the specific company, AAPL = Apple
data = yf.download('AAPL', start='2015-01-01', end='2024-12-31')

# resize dataset to the format that Prophet need
df = data.reset_index()[['Date', 'Close']]
df.columns = ['ds', 'y']
print(df.head())

model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True,
    weekly_seasonality=True
)

# Training the Prophet to fit the specific company stock
model.fit(df)

future = model.make_future_dataframe(periods=5)
forecast = model.predict(future)

# yhat_lower and yhat_upper is confidence interval
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

prophet_fig = model.plot(forecast)

prophet_ax = prophet_fig.gca()
prophet_ax.set_xlim(df['ds'].max(), forecast['ds'].max())

prophet_ax.set_title('AAPL Stock Price Forecast (Prophet)')
prophet_ax.set_xlabel('Date')
prophet_ax.set_ylabel('Price ($)')
plt.show()







