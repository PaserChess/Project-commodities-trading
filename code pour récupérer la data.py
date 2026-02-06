import yfinance as yf
import pandas as pd

# Define the universe (20 assets)
anchor = ["HG=F"]
proxies = ["CPER", "JJCTF", "PICK", "XME"]
equities = [
    "FCX", "SCCO", "IVPAF", "LUNMF", "TECK", 
    "HBM", "ERO", "AAUKF", "BHP", "RIO",
    "GLNCY", "ANFGF", "FM.TO", "ZIJMF", "TGB"
]

full_universe = anchor + proxies + equities

# Data parameters
start_date = "2020-01-01"
end_date = "2026-01-01"
output_file = "copper_project_data.csv"

# Download market data
print(f"Downloading data for {len(full_universe)} tickers...")
raw_data = yf.download(
    full_universe, 
    start=start_date, 
    end=end_date, 
    auto_adjust=True
)

# Extract Adjusted Close prices
prices = raw_data['Close']

# Handle missing values (forward fill for asynchronous market holidays)
# then drop remaining rows that are entirely empty (weekends)
prices = prices.ffill().dropna(how='all')

# Export to CSV
prices.to_csv(output_file)

print("-" * 40)
print(f"Export Complete: {output_file}")
print(f"Dimensions: {prices.shape[0]} rows x {prices.shape[1]} columns")
print("-" * 40)