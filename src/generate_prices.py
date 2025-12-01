import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
DAYS = 365  # Simulate 1 year of data
START_DATE = datetime(2024, 1, 1)

print(f"Generating {DAYS} days of synthetic Smart Grid data...")

# Create hourly timestamps
timestamps = [START_DATE + timedelta(hours=i) for i in range(DAYS * 24)]
df = pd.DataFrame({'start_timestamp': timestamps})

# --- Simulate Price Logic (The "Physics" of the Grid) ---
# Base price: 20 cents/kWh
df['marketprice'] = 20.0 

# Add Daily Patterns (Hour of Day)
# Morning Peak (7-9 AM) & Evening Peak (17-21 PM)
df['hour'] = df['start_timestamp'].dt.hour
df['marketprice'] += np.where((df['hour'] >= 7) & (df['hour'] <= 9), 10.0, 0) # +10 cents
df['marketprice'] += np.where((df['hour'] >= 17) & (df['hour'] <= 21), 15.0, 0) # +15 cents
# Night dip (0-5 AM)
df['marketprice'] -= np.where((df['hour'] >= 0) & (df['hour'] <= 5), 5.0, 0) # -5 cents

# Add Weekly Patterns (Day of Week)
# Weekends (Sat=5, Sun=6) are cheaper
df['day_of_week'] = df['start_timestamp'].dt.dayofweek
df['marketprice'] -= np.where(df['day_of_week'] >= 5, 8.0, 0) # -8 cents on weekends

# Add Random Noise (Real world isn't perfect)
# Random fluctuation between -3 and +3 cents
df['marketprice'] += np.random.uniform(-3, 3, size=len(df))

# Ensure no negative prices for this simulation (optional, but keeps it simple)
df['marketprice'] = df['marketprice'].clip(lower=0)

# --- Save to CSV ---
# Note: We save it as 'marketprice' (Eur/MWh style) to match our AI's expectation
# But here we simulated cents directly for simplicity. 
# Let's multiply by 10 so our AI's "divide by 10" logic works perfectly.
df['marketprice'] = df['marketprice'] * 10 

OUTPUT_PATH = '../data/synthetic_prices.csv'
df.to_csv(OUTPUT_PATH, index=False)

print(f"Success! Saved {len(df)} rows to {OUTPUT_PATH}")
print(df.head())