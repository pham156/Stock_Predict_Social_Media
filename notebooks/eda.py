#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import sweetviz as sv


# In[ ]:


sp500_historic_df = pd.read_csv("../data/sp500_historical.csv")
display(sp500_historic_df.head())


# In[ ]:


print(sp500_historic_df.columns)


# In[ ]:


# Lib for EDA visual

report = sv.analyze(sp500_historic_df)
report.show_html("sweetviz_eda.html")


# In[ ]:


tickers = sp500_historic_df["Ticker"]
unique_tickers = set(tickers.values)
print(len(unique_tickers), unique_tickers)
print(tickers.isna().any())
print(tickers.dtype)


# In[ ]:


dates = sp500_historic_df["Date"]
print(dates.isna().any())
print(dates.dtype)


# In[ ]:


adj_close = sp500_historic_df["Adj Close"]
print(adj_close.isna().any())
print(adj_close.dtype)


# Data Cleaning

# In[ ]:


# Utilizing counts for tickers

ticker_counts = sp500_historic_df['Ticker'].value_counts()
print(ticker_counts)
valid_tickers_t = ticker_counts[ticker_counts > 1826]
print(valid_tickers_t)

ticker_counts_map = {}
for index, row in sp500_historic_df.iterrows():
    ticker = row["Ticker"]
    if ticker in ticker_counts_map:
        ticker_counts_map[ticker] += 1
    else:
        ticker_counts_map[ticker] = 1


# In[ ]:


# Create a set of valid tickers with 5 years or more data
valid_tickers = set()

for ticker, count in ticker_counts_map.items():
    if count > 1826:
        valid_tickers.add(ticker)


# In[ ]:


print((len(valid_tickers)), valid_tickers)


# In[ ]:


sp500_historic_df = sp500_historic_df[sp500_historic_df['Ticker'].isin(valid_tickers)]


# In[ ]:


display(sp500_historic_df)


# In[ ]:


# Forward fill NA values

sp500_historic_df = sp500_historic_df.groupby('Ticker').apply(lambda g: g.fillna(method='ffill')).reset_index(drop=True)
sp500_historic_df = sp500_historic_df.dropna(subset=["Adj Close"])


# In[ ]:


display(sp500_historic_df)


# In[ ]:


sp500_historic_df.to_csv("../data/sp500_historical_clean.csv", index=False)


# Exploratory Data Analysis

# In[ ]:


sp500_historic_df = pd.read_csv("../data/sp500_historical_clean.csv")


# In[ ]:


display(sp500_historic_df)


# In[ ]:


apple_data = sp500_historic_df[sp500_historic_df["Ticker"] == "AAPL"]


# In[ ]:


apple_data


# In[ ]:


# Plot APPLE data
x = np.arange(0, apple_data.shape[0])
plt.plot(x, apple_data["Adj Close"], label="Adj Close")
plt.plot(x, apple_data["Close"], label="Close")
# plt.plot(x, apple_data["Volume"])
plt.legend()
plt.show()


# In[ ]:


# Plot data for APPLE, NVIDIA, MSFT, META, GOOGL, AMZN
major_stocks = ["AAPL", "NVDA", "MSFT", "META", "GOOGL", "AMZN"]
df_major = sp500_historic_df[sp500_historic_df["Ticker"].isin(major_stocks)].copy()

# Days since start
start_date = pd.Timestamp("2015-07-27")
df_major['Date'] = pd.to_datetime(df_major['Date'])
df_major['Days_Since_Start'] = (df_major['Date'] - start_date).dt.days

df_major['AdjClose_0'] = df_major.groupby('Ticker')['Adj Close'].transform('first')
df_major['AdjClose_Percent'] = ((df_major['Adj Close'] - df_major['AdjClose_0']) / df_major["AdjClose_0"]) * 100


# In[ ]:


display(df_major)


# In[ ]:


plt.figure(figsize=(12,6))
for ticker in major_stocks:
    subset = df_major[df_major['Ticker']==ticker]
    plt.plot(subset['Days_Since_Start'], subset['Adj Close'], label=ticker)
plt.title("Major Stock Price Trends")
plt.xlabel("Days Since July 27, 2015")
plt.ylabel("Adj Close Price ($)")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


# With NVIDIA
plt.figure(figsize=(12,6))
for ticker in major_stocks:
    subset = df_major[df_major['Ticker']==ticker]
    plt.plot(subset['Days_Since_Start'], subset['AdjClose_Percent'], label=ticker)
plt.title("Major Stock Price Trends")
plt.xlabel("Days Since July 27, 2015")
plt.ylabel("% Increase in Adj Close Price ($)")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


# Cleaner, Without NVIDIA -- too much increase
plt.figure(figsize=(12,6))
for ticker in major_stocks:
    if ticker != "NVDA":
        subset = df_major[df_major['Ticker']==ticker]
        plt.plot(subset['Days_Since_Start'], subset['AdjClose_Percent'], label=ticker)
plt.title("Major Stock Price Trends")
plt.xlabel("Days Since July 27, 2015")
plt.ylabel("% Increase in Adj Close Price ($)")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


# Processing for all stocks
start_date = pd.Timestamp("2015-07-27")
sp500_historic_df['Date'] = pd.to_datetime(sp500_historic_df['Date'])
sp500_historic_df['Days_Since_Start'] = (sp500_historic_df['Date'] - start_date).dt.days

sp500_historic_df['AdjClose_0'] = sp500_historic_df.groupby('Ticker')['Adj Close'].transform('first')
sp500_historic_df['AdjClose_Percent'] = ((sp500_historic_df['Adj Close'] - sp500_historic_df['AdjClose_0']) / sp500_historic_df["AdjClose_0"]) * 100


# In[ ]:


intel_data = sp500_historic_df[sp500_historic_df["Ticker"] == "INTC"]
plt.plot(intel_data["Days_Since_Start"], intel_data["AdjClose_Percent"], label = "Intel")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


display(intel_data)

