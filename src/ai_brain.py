import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURATION ---
# We use the large synthetic dataset for training to ensure high accuracy
DATA_PATH = '../data/synthetic_prices.csv' 

print("--- 1. INITIALIZING SMART GRID TRADING AGENT ---")

# --- 2. DATA LOADING ---
try:
    print(f"Loading historical data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Convert timestamp to datetime objects
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    
    print(f"Success. Loaded {len(df)} hours of market data.")

except FileNotFoundError:
    print(f"CRITICAL ERROR: Data file not found at {DATA_PATH}.")
    print("Please run 'python src/generate_prices.py' first to create the simulation.")
    exit()

# --- 3. FEATURE ENGINEERING ---
# We extract patterns that the AI can learn from
print("Engineering features (Hour, Day, Trends)...")

df['hour'] = df['start_timestamp'].dt.hour
df['day_of_week'] = df['start_timestamp'].dt.dayofweek

# Normalize Price: Convert Eur/MWh to Cents/kWh for easier reading
# (If your simulation already produced Cents, this keeps the scale consistent)
# Assuming simulation saved raw values, we ensure it's float.
df['price_c_kwh'] = df['marketprice'].astype(float)

# One-Hot Encoding (The "Dummy" Variables)
# This prevents the AI from thinking Monday (0) is "less" than Sunday (6)
df = pd.get_dummies(df, columns=['hour', 'day_of_week'])

# --- 4. DATA SPLIT ---
# Define Target (y) -> What we want to predict (Price)
y = df['price_c_kwh']

# Define Features (X) -> What the AI knows (Time info)
# Drop the target and the original raw columns
drop_cols = ['marketprice', 'start_timestamp', 'price_c_kwh']
X = df.drop(columns=[c for c in drop_cols if c in df.columns], axis=1)

# Split: 80% for Training, 20% for Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. MODEL TRAINING ---
print("Training Random Forest Regressor (This may take a moment)...")

# Initialize the Brain
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Brain
model.fit(X_train, y_train)

# --- 6. EVALUATION ---
print("--- MODEL RESULTS ---")

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Scores
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Model R² Score (Accuracy): {r2:.4f} (Target: > 0.80)")
print(f"Mean Absolute Error (MAE): {mae:.4f} Cents/kWh")

# Sanity Check: Print a real prediction
print("\n--- SAMPLE PREDICTION ---")
print(f"True Price:      {y_test.iloc[0]:.2f} Cents")
print(f"Predicted Price: {y_pred[0]:.2f} Cents")