import yfinance as yf
import pandas as pd

# 1. Get Ticker List from Wikipedia with Error Handling
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

try:
    # Added a 'headers' parameter to mimic a browser request, which can help avoid SSL issues
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    tables = pd.read_html(sp500_url, header=0, attrs={'class': 'wikitable'}, storage_options=headers)
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
    print(f"Successfully retrieved {len(tickers)} tickers.")
    
except Exception as e:
    print(f"Failed to download the S&P 500 list: {e}")
    # Exit if we can't get the list
    exit()

# 2. Download Stock Data with a Fallback Strategy
print("Starting to download stock data. This may take a few minutes...")

# Strategy: Download in smaller chunks if the full batch fails
try:
    # Attempt a full download first
    # *Change period as needed
    data = yf.download(tickers, period="10y", auto_adjust=False, threads=True, progress=True)
    
except Exception as e:
    print(f"Batch download failed: {e}. Trying an alternative method...")
    # Fallback: Download one ticker at a time and combine
    data_list = []
    successful_tickers = []
    
    for ticker in tickers:
        try:
            temp_data = yf.download(ticker, period="10y", auto_adjust=False)
            if not temp_data.empty:
                # Mark this ticker's data
                for col in temp_data.columns:
                    temp_data[('Ticker', '')] = ticker
                data_list.append(temp_data)
                successful_tickers.append(ticker)
                print(f"Retrieved data for {ticker}")
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
    
    if data_list:
        # Combine all individually downloaded data
        data = pd.concat(data_list, axis=0)
        tickers = successful_tickers  # Update ticker list to only successful ones
        print(f"Successfully downloaded data for {len(successful_tickers)} out of {len(tickers)} tickers.")
    else:
        print("Failed to download any data. Please check your connection and try again later.")
        exit()

# 3. Restructure and Save Data
try:
    # If batch download was successful, stack the data
    if 'Ticker' not in data.columns:  # Check if it's the wide format from batch download
        long_format_data = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
    else:
        # It's already from the fallback method
        long_format_data = data.reset_index().rename(columns={'Date': 'Date'})  # Adjust as needed

    # Save to CSV
    long_format_data.to_csv('../data/sp500_historical.csv', index=False)
    print("Data successfully saved to 'sp500_historical.csv'")
    print(long_format_data.head())
    
except Exception as e:
    print(f"An error occurred while processing or saving the data: {e}")