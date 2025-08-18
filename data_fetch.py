import yfinance as yf

# Last 1 year of daily data
data = yf.download("AAPL", period="1y")

print(data.head())
data.to_csv("AAPL_1year.csv")
