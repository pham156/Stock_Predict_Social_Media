import yfinance as yf
import pandas as pd
import numpy as np


def compute_tech_indicators(df_t):
    """
    df_t: ['date', 'ticker', 'close', 'volume', ...]
    """


    df_t["sma_10"] = df_t["close"].rolling(window=10, min_periods=5).mean()


    df_t["ema_10"] = df_t["close"].ewm(span=10, adjust=False).mean()


    delta = df_t["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    roll_up = gain.rolling(window=14, min_periods=14).mean()
    roll_down = loss.rolling(window=14, min_periods=14).mean()

    rs = roll_up / (roll_down + 1e-9)
    df_t["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))


    ema_12 = df_t["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df_t["close"].ewm(span=26, adjust=False).mean()
    df_t["macd"] = ema_12 - ema_26
    df_t["macd_signal"] = df_t["macd"].ewm(span=9, adjust=False).mean()
    df_t["macd_hist"] = df_t["macd"] - df_t["macd_signal"]

    df_t["return"] = df_t["close"].pct_change()
    df_t["vol_20"] = df_t["return"].rolling(window=20, min_periods=10).std()

    return df_t


sp500_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

data = yf.download(
    sp500_tickers,
    start="2016-01-01",
    end="2024-12-31",
    group_by="ticker",
    auto_adjust=False
)

data.head()

all_rows = []

for ticker in sp500_tickers:
    df_t = data[ticker].copy()
    df_t["ticker"] = ticker
    df_t = df_t.reset_index()
    df_t.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        },
        inplace=True
    )
    all_rows.append(df_t[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]])

price_df = pd.concat(all_rows, ignore_index=True)
price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)

price_df.head()

tech_list = []

for ticker, df_t in price_df.groupby("ticker"):
    df_t = df_t.sort_values("date").reset_index(drop=True)
    df_t = compute_tech_indicators(df_t)
    tech_list.append(df_t)

price_feat_df = pd.concat(tech_list, ignore_index=True)

price_feat_df = price_feat_df.dropna().reset_index(drop=True)
price_feat_df.head()

feature_cols = [
    "date", "ticker",
    "open", "high", "low", "close", "volume",
    "sma_10", "ema_10", "rsi_14", "macd", "vol_20",
]

final_df = price_feat_df[feature_cols].copy()
final_df = final_df.sort_values(["ticker", "date"]).reset_index(drop=True)
final_df.to_csv("sp500_features.csv", index=False)



