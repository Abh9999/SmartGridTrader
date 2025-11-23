import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#from sklearn.linear_model import LinearRegression

# --- 1. DATA LOADING AND UNNESTING ---
DATA_PATH = '../data/market_prices.json' # Reading the clean JSON file

try:
    print("--- Starting Data Prep ---")
    # Load the JSON as a single object (it's in the structure: [{...}])
    df = pd.read_json(DATA_PATH) 

except FileNotFoundError:
    print(f"ERROR: Data file not found at {DATA_PATH}. Please run the n8n workflow once.")
    exit()

# --- 2. PREPROCESSING & FEATURE ENGINEERING ---

# Convert timestamp (Fixes previous KeyError if the conversion failed)
df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], unit='ms')

# Feature Engineering (Extracting time features)
df['hour'] = df['start_timestamp'].dt.hour
df['day_of_week'] = df['start_timestamp'].dt.dayofweek
df['price_c_kwh'] = df['marketprice'] * 0.1 # Normalization

# Feature Encoding (OHE)
df = pd.get_dummies(df, columns=['hour', 'day_of_week'])

# --- 3. ML PREP & TRAINING ---

# Define non-feature columns to drop (removing 'object' and 'url' which were top-level keys)
columns_to_drop = ['marketprice', 'start_timestamp', 'end_timestamp', 'unit'] 

# Define Target (y) and Features (X)
y = df['price_c_kwh']
# Drop the target itself and all redundant original columns
X = df.drop(columns=[col for col in columns_to_drop if col in df.columns] + ['price_c_kwh'], axis=1)
print(X.columns.tolist())
# Train-Test Split (20% test data, 80% train data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

#linearRegression
# 6. Model Initialization
#model = LinearRegression()

# 7. Model Training
#print("Training Linear Regressor...")
#model.fit(X_train, y_train)

# Model Evaluation
score = model.score(X_test, y_test) # R^2 Score
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Training Complete ---")
print(f"Model R^2 Score: {score:.4f}")
print(f"Model MAE (Avg. Price Error): {mae:.4f} Cents/kWh")