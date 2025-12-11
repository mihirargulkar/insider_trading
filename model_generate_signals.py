"""
Generate model predictions per horizon and create signals CSV for backtesting.

- Source: insider_trades_with_returns.csv
- Split: chronological, train < 2024-01-01, test >= 2024-01-01
- Horizons: 30/60/90 day returns (targets: return_30d_close, return_60d_close, return_90d_close)
- Features: last_price, Qty, shares_held, Owned, Value, transaction_type, ticker (target-encoded)
- Outputs: test_data.csv with columns Ticker, transaction_date, pred_30d, pred_60d, pred_90d
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "insider_trades_with_returns.csv"
OUT_SIGNALS = ROOT / "test_data.csv"
CUTOFF = pd.Timestamp("2024-01-01")

BASE_NUMERIC_FEATURES = ["last_price", "Qty", "shares_held", "Owned", "Value"]
CATEGORICAL_FEATURES = ["transaction_type", "ticker"]
HORIZONS = [30, 60, 90]
TARGET_COLS = {30: "return_30d_close", 60: "return_60d_close", 90: "return_90d_close"}
PRED_COLS = {30: "pred_30d", 60: "pred_60d", 90: "pred_90d"}


def clean_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip()
    )


def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    ticker_means = y_train.groupby(X_train["ticker"]).mean()
    global_mean = y_train.mean()
    X_train["ticker_encoded"] = X_train["ticker"].map(ticker_means)
    X_test["ticker_encoded"] = X_test["ticker"].map(ticker_means).fillna(global_mean)

    X_train = X_train.drop(columns=["ticker"])
    X_test = X_test.drop(columns=["ticker"])

    X_train = pd.get_dummies(X_train, columns=["transaction_type"], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=["transaction_type"], drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaled_cols = [c for c in X_train.columns if c in BASE_NUMERIC_FEATURES or c == "ticker_encoded"]
    scaler = StandardScaler()
    X_train[scaled_cols] = scaler.fit_transform(X_train[scaled_cols])
    X_test[scaled_cols] = scaler.transform(X_test[scaled_cols])
    return X_train, X_test


def train_and_predict(df: pd.DataFrame, horizon: int) -> Tuple[pd.Series, Dict[str, float]]:
    ret_col = TARGET_COLS[horizon]
    df_h = df.dropna(subset=[ret_col]).copy()
    df_h = df_h[df_h[ret_col].abs() <= 10]
    df_h = df_h.sort_values("transaction_date")

    train_df = df_h[df_h["transaction_date"] < CUTOFF]
    test_df = df_h[df_h["transaction_date"] >= CUTOFF]
    if train_df.empty or test_df.empty:
        raise ValueError(f"Need non-empty train/test for horizon {horizon}")

    features = BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train_raw, y_train = train_df[features], train_df[ret_col]
    X_test_raw, y_test = test_df[features], test_df[ret_col]
    X_train, X_test = preprocess(X_train_raw, X_test_raw, y_train)

    weights = compute_sample_weight(class_weight="balanced", y=X_train_raw["transaction_type"])

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        objective="reg:squarederror",
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )

    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes: List[float] = []
    for tr_idx, va_idx in tscv.split(X_train):
        m = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            objective="reg:squarederror",
            random_state=42,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], sample_weight=weights[tr_idx])
        pred_val = m.predict(X_train.iloc[va_idx])
        cv_maes.append(mean_absolute_error(y_train.iloc[va_idx], pred_val))

    model.fit(X_train, y_train, sample_weight=weights)
    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "baseline_mae": float(mean_absolute_error(y_test, [y_train.mean()] * len(y_test))),
        "cv_mae_mean": float(np.mean(cv_maes)) if cv_maes else np.nan,
        "cv_mae_std": float(np.std(cv_maes)) if cv_maes else np.nan,
    }
    return pd.Series(y_pred, index=test_df.index, name=PRED_COLS[horizon]), metrics


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Source file not found: {SRC}")
    df = pd.read_csv(SRC)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce").dt.normalize()
    df["trade_date"] = pd.to_datetime(df.get("trade_date"), errors="coerce").dt.normalize()
    df = df[df["transaction_date"].notna()]

    df["ticker"] = df["ticker"].astype(str)
    df["transaction_type"] = df["transaction_type"].astype(str)
    df["last_price"] = pd.to_numeric(clean_numeric(df["last_price"]), errors="coerce")
    df["Qty"] = pd.to_numeric(clean_numeric(df["Qty"]), errors="coerce")
    df["shares_held"] = pd.to_numeric(clean_numeric(df["shares_held"]), errors="coerce")
    df["Owned"] = pd.to_numeric(df["Owned"].astype(str).str.replace("%", "", regex=False), errors="coerce") / 100
    df["Value"] = pd.to_numeric(clean_numeric(df["Value"]), errors="coerce")

    df = df.dropna(subset=TARGET_COLS.values())
    df = df[df["transaction_date"].notna()]
    df = df.sort_values("transaction_date").reset_index(drop=True)

    preds: Dict[int, pd.Series] = {}
    for h in HORIZONS:
        pred_series, metrics = train_and_predict(df, h)
        preds[h] = pred_series
        print(
            f"[h={h}d] test_size={len(pred_series)}, "
            f"MAE={metrics['mae']:.4f} (cv {metrics['cv_mae_mean']:.4f}±{metrics['cv_mae_std']:.4f}, "
            f"baseline {metrics['baseline_mae']:.4f})"
        )

    # Assemble test set with preds
    test_idx = preds[30].index
    test_out = df.loc[test_idx, ["ticker", "transaction_date"]].copy()
    for h in HORIZONS:
        test_out[PRED_COLS[h]] = preds[h]
    test_out = test_out.rename(columns={"ticker": "Ticker"})
    test_out.to_csv(OUT_SIGNALS, index=False)
    print(f"[done] wrote {OUT_SIGNALS} (rows={len(test_out)})")


if __name__ == "__main__":
    main()
