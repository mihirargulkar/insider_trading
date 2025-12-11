import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import joblib
from pathlib import Path
import xgboost as xgb

df = pd.read_csv('src/data/insider_trades_with_returns.csv')

# Exclude rows where return_30d_close, return_60d_close, or return_90d_close are NaN
df_clean = df.dropna(subset=['return_30d_close', 'return_60d_close', 'return_90d_close'])
df_clean = df.drop(columns=['transaction_date', 'trade_date', 'company_name'])

# Function to clean numeric strings (remove $, +, commas, parentheses)
def clean_numeric(series):
    """Remove $, +, commas, and convert to float"""
    return series.astype(str).str.replace('$', '', regex=False)\
                              .str.replace('+', '', regex=False)\
                              .str.replace(',', '', regex=False)\
                              .str.replace('(', '-', regex=False)\
                              .str.replace(')', '', regex=False)\
                              .str.strip()

# Convert string columns
df_clean['ticker'] = df_clean['ticker'].astype(str)
df_clean['owner_name'] = df_clean['owner_name'].astype(str)
df_clean['Title'] = df_clean['Title'].astype(str)
df_clean['transaction_type'] = df_clean['transaction_type'].astype(str)

# Convert last_price (remove $ and convert to float)
df_clean['last_price'] = pd.to_numeric(clean_numeric(df_clean['last_price']), errors='coerce')

# Convert Qty (remove + and commas, convert to float)
df_clean['Qty'] = pd.to_numeric(clean_numeric(df_clean['Qty']), errors='coerce')

# Convert shares_held (remove commas, convert to float)
df_clean['shares_held'] = pd.to_numeric(clean_numeric(df_clean['shares_held']), errors='coerce')

# Convert Owned (remove % if present, convert to float)
df_clean['Owned'] = pd.to_numeric(df_clean['Owned'].astype(str).str.replace('%', '', regex=False), errors='coerce') / 100

# Convert Value (remove $, +, commas, convert to float)
df_clean['Value'] = pd.to_numeric(clean_numeric(df_clean['Value']), errors='coerce')

# Helper to save model + preprocessing artifacts for a given split
def save_artifacts(model, X_train_raw, y_train, suffix, base_numeric_features):
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)

    ticker_means = y_train.groupby(X_train_raw['ticker']).mean()
    global_mean = y_train.mean()

    X_art = X_train_raw.copy()
    X_art['ticker_encoded'] = X_art['ticker'].map(ticker_means)
    X_art['ticker_encoded'] = X_art['ticker_encoded'].fillna(global_mean)
    X_art = X_art.drop(columns=['ticker'])

    X_art = pd.get_dummies(X_art, columns=['transaction_type'], drop_first=True)
    transaction_type_columns = [c for c in X_art.columns if c.startswith('transaction_type_')]

    scaled_cols = base_numeric_features + ['ticker_encoded']
    scaler = StandardScaler()
    X_art[scaled_cols] = scaler.fit_transform(X_art[scaled_cols])

    feature_order = list(X_art.columns)

    model_path = MODEL_DIR / f"xgb_model_{suffix}.json"
    scaler_path = MODEL_DIR / f"scaler_{suffix}.pkl"
    means_path = MODEL_DIR / f"ticker_means_{suffix}.json"
    artifacts_path = MODEL_DIR / f"artifacts_{suffix}.json"

    def _save_booster(obj, path: Path) -> bool:
        try:
            obj.save_model(path)
            return True
        except Exception:
            if hasattr(obj, "get_booster"):
                booster = obj.get_booster()
                booster.save_model(path)
                return True
        return False

    if not _save_booster(model, model_path):
        raise RuntimeError(f"Failed to save XGBoost model for {suffix}")

    joblib.dump(scaler, scaler_path)

    with open(means_path, "w") as f:
        json.dump({
            "ticker_means": ticker_means.to_dict(),
            "global_mean": float(global_mean),
        }, f)

    with open(artifacts_path, "w") as f:
        json.dump({
            "transaction_type_columns": transaction_type_columns,
            "scaled_cols": scaled_cols,
            "feature_order": feature_order,
        }, f)

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved ticker means to {means_path}")
    print(f"Saved artifacts to {artifacts_path}")


df_clean

df_clean = df_clean.dropna(subset=['return_30d_close', 'return_60d_close', 'return_90d_close'])
df_clean

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# --- Configuration ---
# NOTE: df_clean must be defined and loaded before running this script.
BASE_NUMERIC_FEATURES = ['last_price', 'Qty', 'shares_held', 'Owned', 'Value']
CATEGORICAL_FEATURES = ['transaction_type', 'ticker']
TARGET_COLUMN = 'return_30d_close'
ALL_FEATURES = BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ==========================================
# 1. Data Processing Function
# ==========================================
def preprocess_splits(X_train, X_test, y_train):
    """Handles Target Encoding, One-Hot Encoding, and Scaling."""

    X_train = X_train.copy()
    X_test = X_test.copy()

    # --- A. Target Encoding for Ticker ---
    # Calculate mean return per ticker ONLY on training data
    ticker_means = y_train.groupby(X_train['ticker']).mean()
    global_mean = y_train.mean()

    X_train['ticker_encoded'] = X_train['ticker'].map(ticker_means)
    X_test['ticker_encoded'] = X_test['ticker'].map(ticker_means)

    # Fill unknown tickers in test set with the global average
    X_test['ticker_encoded'] = X_test['ticker_encoded'].fillna(global_mean)

    # Drop the original text column
    X_train = X_train.drop(columns=['ticker'])
    X_test = X_test.drop(columns=['ticker'])

    scaled_cols = BASE_NUMERIC_FEATURES + ['ticker_encoded']

    # --- B. One-Hot Encoding for Transaction Type ---
    X_train = pd.get_dummies(X_train, columns=['transaction_type'], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=['transaction_type'], drop_first=True)

    # Ensure test set has same columns as train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # --- C. Standardization (Scaling) ---
    scaler = StandardScaler()

    # Scale only the designated numeric and encoded columns
    X_train[scaled_cols] = scaler.fit_transform(X_train[scaled_cols])
    X_test[scaled_cols] = scaler.transform(X_test[scaled_cols])

    return X_train, X_test

# ==========================================
# 2. Visualization Function
# ==========================================
def plot_performance(y_test, y_pred, model):
    """Generates evaluation plots including 'weight' feature importance."""

    plt.figure(figsize=(15, 5))

    # Plot 1: Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.3)
    # Plot the ideal prediction line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Predicted vs Actual Returns')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # Plot 2: Residual Histogram
    residuals = y_test - y_pred
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.title(f'Residual Distribution (Mean: {residuals.mean():.4f})')

    # Plot 3: Feature Importance (Weight/Frequency)
    plt.subplot(1, 3, 3)

    xgb.plot_importance(
        model,
        max_num_features=10,
        importance_type='weight', # Changed from 'gain' to 'weight'
        ax=plt.gca()
    )
    plt.title("Top 10 Feature Importance (Weight/Frequency)")

    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================

# Check transaction type distribution (Original Step 1)
print("Transaction Type Distribution:")
print(df_clean['transaction_type'].value_counts())

# Filter data (Original Step 1)
df_model = df_clean[df_clean[TARGET_COLUMN].abs() <= 10].copy()

# 1. Splitting Data
X = df_model[ALL_FEATURES]
y = df_model[TARGET_COLUMN]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")

# 2. Processing Data
print("\nPreprocessing data (Target Encoding and Scaling)...")
X_train, X_test = preprocess_splits(X_train_raw, X_test_raw, y_train)

# 3. Compute Sample Weights (based on raw transaction type)
weights = compute_sample_weight(
    class_weight='balanced',
    y=X_train_raw['transaction_type']
)

# 4. Train Model
print(f"Training XGBoost on {X_train.shape[1]} features...")
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    objective='reg:squarederror',
    random_state=42,
    reg_alpha=0.1,
    reg_lambda=1.0
)

# Cross-validation on training set
cv_scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validation MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

model.fit(X_train, y_train, sample_weight=weights)

# 5. Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n🚀 === Test Set Performance ===")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Baseline comparison
baseline_mae = mean_absolute_error(y_test, [y_train.mean()] * len(y_test))
print(f"\nBaseline MAE (predicting mean): {baseline_mae:.4f}")
print(f"Improvement over baseline: {((baseline_mae - mae) / baseline_mae * 100):.2f}%")

# Performance by transaction type (using raw test data)
print("\n=== Performance by Transaction Type ===")
test_transaction_types = X_test_raw['transaction_type'].values
for trans_type in sorted(np.unique(test_transaction_types)):
    mask = test_transaction_types == trans_type
    if mask.sum() > 0:
        mae_type = mean_absolute_error(y_test[mask], y_pred[mask])
        r2_type = r2_score(y_test[mask], y_pred[mask])
        print(f"{trans_type:20s}: MAE={mae_type:.4f}, R²={r2_type:.4f}, n={mask.sum()}")


# 6. Visualize
plot_performance(y_test, y_pred, model)

# 7. Check Overfitting
train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, train_pred)
print(f"\n=== Overfitting Check ===")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Difference: {abs(train_mae - mae):.4f}")
save_artifacts(model, X_train_raw, y_train, "30d", BASE_NUMERIC_FEATURES)
save_artifacts(model, X_train_raw, y_train, "60d", BASE_NUMERIC_FEATURES)
save_artifacts(model, X_train_raw, y_train, "30d", BASE_NUMERIC_FEATURES)

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# --- Configuration ---
# NOTE: df_clean must be defined and loaded before running this script.
BASE_NUMERIC_FEATURES = ['last_price', 'Qty', 'shares_held', 'Owned', 'Value']
CATEGORICAL_FEATURES = ['transaction_type', 'ticker']
TARGET_COLUMN = 'return_60d_close'
ALL_FEATURES = BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ==========================================
# 1. Data Processing Function
# ==========================================
def preprocess_splits(X_train, X_test, y_train):
    """Handles Target Encoding, One-Hot Encoding, and Scaling."""

    X_train = X_train.copy()
    X_test = X_test.copy()

    # --- A. Target Encoding for Ticker ---
    # Calculate mean return per ticker ONLY on training data
    ticker_means = y_train.groupby(X_train['ticker']).mean()
    global_mean = y_train.mean()

    X_train['ticker_encoded'] = X_train['ticker'].map(ticker_means)
    X_test['ticker_encoded'] = X_test['ticker'].map(ticker_means)

    # Fill unknown tickers in test set with the global average
    X_test['ticker_encoded'] = X_test['ticker_encoded'].fillna(global_mean)

    # Drop the original text column
    X_train = X_train.drop(columns=['ticker'])
    X_test = X_test.drop(columns=['ticker'])

    scaled_cols = BASE_NUMERIC_FEATURES + ['ticker_encoded']

    # --- B. One-Hot Encoding for Transaction Type ---
    X_train = pd.get_dummies(X_train, columns=['transaction_type'], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=['transaction_type'], drop_first=True)

    # Ensure test set has same columns as train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # --- C. Standardization (Scaling) ---
    scaler = StandardScaler()

    # Scale only the designated numeric and encoded columns
    X_train[scaled_cols] = scaler.fit_transform(X_train[scaled_cols])
    X_test[scaled_cols] = scaler.transform(X_test[scaled_cols])

    return X_train, X_test

# ==========================================
# 2. Visualization Function
# ==========================================
def plot_performance(y_test, y_pred, model):
    """Generates evaluation plots including 'weight' feature importance."""

    plt.figure(figsize=(15, 5))

    # Plot 1: Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.3)
    # Plot the ideal prediction line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Predicted vs Actual Returns')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # Plot 2: Residual Histogram
    residuals = y_test - y_pred
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.title(f'Residual Distribution (Mean: {residuals.mean():.4f})')

    # Plot 3: Feature Importance (Weight/Frequency)
    plt.subplot(1, 3, 3)

    xgb.plot_importance(
        model,
        max_num_features=10,
        importance_type='weight', # Changed from 'gain' to 'weight'
        ax=plt.gca()
    )
    plt.title("Top 10 Feature Importance (Weight/Frequency)")

    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================

# Check transaction type distribution (Original Step 1)
print("Transaction Type Distribution:")
print(df_clean['transaction_type'].value_counts())

# Filter data (Original Step 1)
df_model = df_clean[df_clean[TARGET_COLUMN].abs() <= 10].copy()

# 1. Splitting Data
X = df_model[ALL_FEATURES]
y = df_model[TARGET_COLUMN]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")

# 2. Processing Data
print("\nPreprocessing data (Target Encoding and Scaling)...")
X_train, X_test = preprocess_splits(X_train_raw, X_test_raw, y_train)

# 3. Compute Sample Weights (based on raw transaction type)
weights = compute_sample_weight(
    class_weight='balanced',
    y=X_train_raw['transaction_type']
)

# 4. Train Model
print(f"Training XGBoost on {X_train.shape[1]} features...")
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    objective='reg:squarederror',
    random_state=42,
    reg_alpha=0.1,
    reg_lambda=1.0
)

# Cross-validation on training set
cv_scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validation MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

model.fit(X_train, y_train, sample_weight=weights)

# 5. Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n🚀 === Test Set Performance ===")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Baseline comparison
baseline_mae = mean_absolute_error(y_test, [y_train.mean()] * len(y_test))
print(f"\nBaseline MAE (predicting mean): {baseline_mae:.4f}")
print(f"Improvement over baseline: {((baseline_mae - mae) / baseline_mae * 100):.2f}%")

# Performance by transaction type (using raw test data)
print("\n=== Performance by Transaction Type ===")
test_transaction_types = X_test_raw['transaction_type'].values
for trans_type in sorted(np.unique(test_transaction_types)):
    mask = test_transaction_types == trans_type
    if mask.sum() > 0:
        mae_type = mean_absolute_error(y_test[mask], y_pred[mask])
        r2_type = r2_score(y_test[mask], y_pred[mask])
        print(f"{trans_type:20s}: MAE={mae_type:.4f}, R²={r2_type:.4f}, n={mask.sum()}")


# 6. Visualize
plot_performance(y_test, y_pred, model)

# 7. Check Overfitting
train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, train_pred)
print(f"\n=== Overfitting Check ===")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Difference: {abs(train_mae - mae):.4f}")
save_artifacts(model, X_train_raw, y_train, "60d", BASE_NUMERIC_FEATURES)

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# --- Configuration ---
# NOTE: df_clean must be defined and loaded before running this script.
BASE_NUMERIC_FEATURES = ['last_price', 'Qty', 'shares_held', 'Owned', 'Value']
CATEGORICAL_FEATURES = ['transaction_type', 'ticker']
TARGET_COLUMN = 'return_90d_close'
ALL_FEATURES = BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ==========================================
# 1. Data Processing Function
# ==========================================
def preprocess_splits(X_train, X_test, y_train):
    """Handles Target Encoding, One-Hot Encoding, and Scaling."""

    X_train = X_train.copy()
    X_test = X_test.copy()

    # --- A. Target Encoding for Ticker ---
    # Calculate mean return per ticker ONLY on training data
    ticker_means = y_train.groupby(X_train['ticker']).mean()
    global_mean = y_train.mean()

    X_train['ticker_encoded'] = X_train['ticker'].map(ticker_means)
    X_test['ticker_encoded'] = X_test['ticker'].map(ticker_means)

    # Fill unknown tickers in test set with the global average
    X_test['ticker_encoded'] = X_test['ticker_encoded'].fillna(global_mean)

    # Drop the original text column
    X_train = X_train.drop(columns=['ticker'])
    X_test = X_test.drop(columns=['ticker'])

    scaled_cols = BASE_NUMERIC_FEATURES + ['ticker_encoded']

    # --- B. One-Hot Encoding for Transaction Type ---
    X_train = pd.get_dummies(X_train, columns=['transaction_type'], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=['transaction_type'], drop_first=True)

    # Ensure test set has same columns as train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # --- C. Standardization (Scaling) ---
    scaler = StandardScaler()

    # Scale only the designated numeric and encoded columns
    X_train[scaled_cols] = scaler.fit_transform(X_train[scaled_cols])
    X_test[scaled_cols] = scaler.transform(X_test[scaled_cols])

    return X_train, X_test

# ==========================================
# 2. Visualization Function
# ==========================================
def plot_performance(y_test, y_pred, model):
    """Generates evaluation plots including 'weight' feature importance."""

    plt.figure(figsize=(15, 5))

    # Plot 1: Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.3)
    # Plot the ideal prediction line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Predicted vs Actual Returns')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # Plot 2: Residual Histogram
    residuals = y_test - y_pred
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.title(f'Residual Distribution (Mean: {residuals.mean():.4f})')

    # Plot 3: Feature Importance (Weight/Frequency)
    plt.subplot(1, 3, 3)

    xgb.plot_importance(
        model,
        max_num_features=10,
        importance_type='weight', # Changed from 'gain' to 'weight'
        ax=plt.gca()
    )
    plt.title("Top 10 Feature Importance (Weight/Frequency)")

    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================

# Check transaction type distribution (Original Step 1)
print("Transaction Type Distribution:")
print(df_clean['transaction_type'].value_counts())

# Filter data (Original Step 1)
df_model = df_clean[df_clean[TARGET_COLUMN].abs() <= 10].copy()

# 1. Splitting Data
X = df_model[ALL_FEATURES]
y = df_model[TARGET_COLUMN]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")

# 2. Processing Data
print("\nPreprocessing data (Target Encoding and Scaling)...")
X_train, X_test = preprocess_splits(X_train_raw, X_test_raw, y_train)

# 3. Compute Sample Weights (based on raw transaction type)
weights = compute_sample_weight(
    class_weight='balanced',
    y=X_train_raw['transaction_type']
)

# 4. Train Model
print(f"Training XGBoost on {X_train.shape[1]} features...")
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    objective='reg:squarederror',
    random_state=42,
    reg_alpha=0.1,
    reg_lambda=1.0
)

# Cross-validation on training set
cv_scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validation MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

model.fit(X_train, y_train, sample_weight=weights)

# 5. Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n🚀 === Test Set Performance ===")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Baseline comparison
baseline_mae = mean_absolute_error(y_test, [y_train.mean()] * len(y_test))
print(f"\nBaseline MAE (predicting mean): {baseline_mae:.4f}")
print(f"Improvement over baseline: {((baseline_mae - mae) / baseline_mae * 100):.2f}%")

# Performance by transaction type (using raw test data)
print("\n=== Performance by Transaction Type ===")
test_transaction_types = X_test_raw['transaction_type'].values
for trans_type in sorted(np.unique(test_transaction_types)):
    mask = test_transaction_types == trans_type
    if mask.sum() > 0:
        mae_type = mean_absolute_error(y_test[mask], y_pred[mask])
        r2_type = r2_score(y_test[mask], y_pred[mask])
        print(f"{trans_type:20s}: MAE={mae_type:.4f}, R²={r2_type:.4f}, n={mask.sum()}")


# 6. Visualize
plot_performance(y_test, y_pred, model)

# 7. Check Overfitting
train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, train_pred)
print(f"\n=== Overfitting Check ===")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Difference: {abs(train_mae - mae):.4f}")
save_artifacts(model, X_train_raw, y_train, "90d", BASE_NUMERIC_FEATURES)
