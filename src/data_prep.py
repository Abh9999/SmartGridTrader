import pandas as pd

# --- File Path Check ---
# The script is in 'src/', so we need '../' to go up one level to find 'data/'
DATA_PATH = '../data/market_prices.csv' 

# 1. Load the data (read the JSON we saved)
df = pd.read_csv(DATA_PATH) 

# The API response is nested, the actual price data is inside the 'data' key (row 0)
# We need to extract that list of prices into our main DataFrame
#df = pd.DataFrame(df['data'].iloc[0]) 

# 2. Convert the timestamp from milliseconds to datetime object
df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], unit='ms')

# 3. Feature Engineering: Extract time features
df['hour'] = df['start_timestamp'].dt.hour
df['day_of_week'] = df['start_timestamp'].dt.dayofweek

# 4. Data Cleaning: Price Normalization
df['price_c_kwh'] = df['marketprice'] * 0.1

# 5. Feature Encoding: One-Hot Encode time features
df = pd.get_dummies(df, columns=['hour', 'day_of_week'])


# --- Define the function ---
def clean_and_feature_engineer(df_raw):
    """
    Cleans raw market data, converts timestamps, and creates features (hour, day).
    """
    # 1. Convert the timestamp from milliseconds to datetime object
    df_raw['start_timestamp'] = pd.to_datetime(df_raw['start_timestamp'], unit='ms')

    # 2. Feature Engineering: Extract time features
    df_raw['hour'] = df_raw['start_timestamp'].dt.hour
    df_raw['day_of_week'] = df_raw['start_timestamp'].dt.dayofweek
    
    # 3. Data Cleaning: Price Normalization (Eur/MWh to Cents/kWh)
    df_raw['price_c_kwh'] = df_raw['marketprice'] * 0.1

    # 4. Feature Encoding: One-Hot Encode time features
    df_raw = pd.get_dummies(df_raw, columns=['hour', 'day_of_week'])

    return df_raw

# --- Testing area (optional, can be deleted later) ---
if __name__ == "__main__":
    DATA_PATH = '../data/market_prices.csv'
    try:
        df_test = pd.read_csv(DATA_PATH)
        df_cleaned = clean_and_feature_engineer(df_test)
        print("Test run successful. Cleaned DataFrame shape:", df_cleaned.shape)
    except FileNotFoundError:
        print("Test failed: Data file not found.")