import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

forecast_df = pd.read_csv("data/client_forecast.csv")
sla_df = pd.read_csv("data/sla_volume.csv")

# ---------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------

forecast_df['date'] = pd.to_datetime(forecast_df['date'])
sla_df['date'] = pd.to_datetime(sla_df['date'])

# Merge datasets
df = pd.merge(
    forecast_df,
    sla_df,
    on=['date', 'interval', 'skill', 'location'],
    how='left'
)

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

# Date Features
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

# Interval Features
df['hour'] = df['interval'].str.split(':').str[0].astype(int)

# Peak Hour Flag
df['is_peak_hour'] = np.where(
    (df['hour'] >= 10) & (df['hour'] <= 14),
    1,
    0
)

# Encode categorical columns
df['skill_encoded'] = df['skill'].astype('category').cat.codes
df['location_encoded'] = df['location'].astype('category').cat.codes

# Lag Features
df['prev_volume'] = df['forecasted_volume'].shift(1)
df['rolling_avg_3'] = (
    df['forecasted_volume']
    .rolling(window=3, min_periods=1)
    .mean()
)

# Fill null values
df.fillna(method='bfill', inplace=True)

# ---------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------

feature_columns = [
    'day',
    'month',
    'year',
    'day_of_week',
    'week_of_year',
    'hour',
    'is_peak_hour',
    'skill_encoded',
    'location_encoded',
    'forecasted_hours',
    'aht_sec',
    'service_level',
    'prev_volume',
    'rolling_avg_3'
]

X = df[feature_columns]

y = df['forecasted_volume']

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------

predictions = model.predict(X_test)

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n-------------------------------")
print("FORECAST MODEL PERFORMANCE")
print("-------------------------------")

print(f"Mean Absolute Error  : {round(mae, 2)}")
print(f"Root Mean Sq Error   : {round(rmse, 2)}")
print(f"R2 Score             : {round(r2, 2)}")

# ---------------------------------------------------
# OUTPUT RESULTS
# ---------------------------------------------------

results_df = X_test.copy()

results_df['actual_volume'] = y_test.values
results_df['predicted_volume'] = predictions.round(2)

print("\n-------------------------------")
print("FORECAST OUTPUT SAMPLE")
print("-------------------------------")

print(
    results_df[
        [
            'hour',
            'actual_volume',
            'predicted_volume'
        ]
    ].head(10)
)

# ---------------------------------------------------
# SAVE OUTPUT
# ---------------------------------------------------

results_df.to_csv(
    "outputs/forecast_results.csv",
    index=False
)

print("\nForecast results saved successfully.")
