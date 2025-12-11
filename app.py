import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
import joblib
import yfinance as yf

# Ensure we can import from src/
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from analyze_portfolio import PortfolioAnalyzer  # noqa: E402


DATA_DIR = SRC_DIR / "data" if (SRC_DIR / "data").exists() else BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Insider Trading Strategy",
    page_icon="📈",
    layout="wide",
)


# ---------- Data loading ----------
@st.cache_data
def load_insider():
    try:
        df = pd.read_csv(
            DATA_DIR / "insider_trades_with_returns.csv",
            parse_dates=["transaction_date", "trade_date"],
            low_memory=False,
        )
        df["ticker"] = df["ticker"].astype(str).str.upper()
        return df
    except Exception as e:
        st.warning(f"Insider data unavailable: {e}")
        return None


@st.cache_data
def load_backtest():
    try:
        return pd.read_csv(
            DATA_DIR / "backtest_results.csv",
            parse_dates=["date"],
            low_memory=False,
        )
    except Exception as e:
        st.warning(f"Backtest data unavailable: {e}")
        return None


@st.cache_resource
def load_artifacts(suffix: str):
    try:
        model_path = MODELS_DIR / f"xgb_model_{suffix}.json"
        scaler_path = MODELS_DIR / f"scaler_{suffix}.pkl"
        means_path = MODELS_DIR / f"ticker_means_{suffix}.json"
        artifacts_path = MODELS_DIR / f"artifacts_{suffix}.json"
        if not all(p.exists() for p in [model_path, scaler_path, means_path, artifacts_path]):
            return None

        booster = xgb.Booster()
        booster.load_model(model_path)
        scaler = joblib.load(scaler_path)
        with open(means_path) as f:
            means = json.load(f)
        with open(artifacts_path) as f:
            artifacts = json.load(f)
        return {"model": booster, "scaler": scaler, "means": means, "artifacts": artifacts}
    except Exception as e:
        st.warning(f"Could not load artifacts for {suffix}: {e}")
        return None


INSIDER_DF = load_insider()
BACKTEST_DF = load_backtest()
MODELS = {s: load_artifacts(s) for s in ["30d", "60d", "90d"]}
MAE_BY_SUFFIX = {"30d": 0.1112, "60d": 0.1531, "90d": 0.1866}


# ---------- Helpers ----------
def validate_ticker(t: str) -> Tuple[bool, str]:
    """
    Be lenient: only fail on empty; allow network hiccups by accepting unverified tickers.
    """
    if not t:
        return False, "Missing ticker"
    t = str(t).upper().strip()
    try:
        ticker_obj = yf.Ticker(t)
        fast_info = ticker_obj.fast_info or {}
        if fast_info.get("last_price") is not None:
            return True, "OK"
        hist = ticker_obj.history(period="1d")
        if not hist.empty:
            return True, "OK"
        return False, "Ticker not found"
    except Exception:
        # If API/network fails, allow and continue; downstream will surface any real issues.
        return True, "OK (unverified)"


def predict_suffix(payload: dict, suffix: str):
    bundle = MODELS.get(suffix)
    if not bundle:
        return None, f"{suffix} artifacts unavailable"
    try:
        model = bundle["model"]
        scaler = bundle["scaler"]
        means = bundle["means"]
        artifacts = bundle["artifacts"]

        df = pd.DataFrame([{
            "ticker": payload["ticker"].upper(),
            "transaction_type": payload["transaction_type"],
            "last_price": float(payload["last_price"]),
            "Qty": float(payload["Qty"]),
            "shares_held": float(payload["shares_held"]),
            "Owned": float(payload["Owned"]),
            "Value": float(payload["Value"]),
        }])

        ticker_means = means.get("ticker_means", {})
        global_mean = means.get("global_mean", 0.0)
        df["ticker_encoded"] = df["ticker"].map(ticker_means).fillna(global_mean)
        df = df.drop(columns=["ticker"])

        df = pd.get_dummies(df, columns=["transaction_type"], drop_first=True)
        trans_cols = artifacts.get("transaction_type_columns", [])
        for col in trans_cols:
            if col not in df.columns:
                df[col] = 0

        scaled_cols = artifacts.get("scaled_cols", [])
        df[scaled_cols] = scaler.transform(df[scaled_cols])

        feature_order = artifacts.get("feature_order", list(df.columns))
        df = df.reindex(columns=feature_order, fill_value=0)

        dmat = xgb.DMatrix(df, feature_names=feature_order)
        pred = float(model.predict(dmat)[0])
        return pred, None
    except Exception as e:
        return None, f"{suffix} prediction failed: {e}"


def _benchmark_equity_curve(ticker: str, start_date: datetime.date, target_start: float):
    """
    Build a simple buy/hold equity curve for any ticker using yfinance.
    """
    try:
        hist = yf.Ticker(ticker).history(period="max")
        hist = hist.reset_index()
        hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
        hist = hist.dropna(subset=["Date", "Close"])
        hist = hist[hist["Date"] >= pd.to_datetime(start_date)]
        if hist.empty:
            return None
        base = hist.iloc[0]["Close"]
        hist["equity"] = (hist["Close"] / float(base)) * target_start
        return hist[["Date", "equity"]]
    except Exception:
        return None


def build_backtest_fig(start_date: datetime.date, benchmark_amount: float, benchmark_ticker: str = "SPY"):
    """
    Combine precomputed model curves (30/60/90d) with a user-selected benchmark ticker buy/hold curve.
    """
    target_start = float(benchmark_amount) if (benchmark_amount and benchmark_amount > 0) else 10000.0

    fig = go.Figure()

    # Model curves from precomputed backtest (if available)
    if BACKTEST_DF is not None and not BACKTEST_DF.empty:
        df = BACKTEST_DF.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[df["date"] >= pd.to_datetime(start_date)]
        if not df.empty:
            for col in ["total_equity_30d", "total_equity_60d", "total_equity_90d", "spy_equity"]:
                if col in df.columns:
                    base = df.iloc[0].get(col, 0)
                    if base and base != 0:
                        df[col] = (df[col] / float(base)) * target_start

            curves = [
                ("total_equity_30d", "30d equity", "#1f77b4"),
                ("total_equity_60d", "60d equity", "#ff7f0e"),
                ("total_equity_90d", "90d equity", "#2ca02c"),
            ]
            # Only show the precomputed SPY curve when benchmarking SPY.
            if benchmark_ticker.upper() == "SPY" and "spy_equity" in df.columns:
                curves.append(("spy_equity", "SPY equity (precomputed)", "#d62728"))
            for col, name, color in curves:
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=name, line=dict(color=color)))

    # Benchmark curve from yfinance for any ticker
    bench = _benchmark_equity_curve(benchmark_ticker, start_date, target_start)
    if bench is not None:
        fig.add_trace(go.Scatter(
            x=bench["Date"],
            y=bench["equity"],
            mode="lines",
            name=f"{benchmark_ticker.upper()} buy/hold",
            line=dict(color="#d62728"),
        ))

    if not fig.data:
        return None

    fig.update_layout(
        title=f"Backtest Equity Curves (bench: {benchmark_ticker})",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ---------- Layout ----------
st.title("Insider Trading Strategy")
st.write("Build a portfolio, run backtests, and test insider-trade signals.")

if "rows" not in st.session_state:
    st.session_state.rows = [{"ticker": "SPY", "amount": 1000.0}]

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose page",
        ["Portfolio", "Model Playground"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Powered by Yahoo Finance / XGBoost artifacts")


# ---------- Portfolio page ----------
def _dash_children_to_plain(obj):
    """
    Convert Dash HTML components (with .children/props) into plain Python types for Streamlit rendering.
    """
    if isinstance(obj, (list, tuple)):
        return [_dash_children_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dash_children_to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "props"):
        props = getattr(obj, "props", None)
        if isinstance(props, dict) and "children" in props:
            return _dash_children_to_plain(props["children"])
    if hasattr(obj, "children"):
        try:
            return _dash_children_to_plain(getattr(obj, "children"))
        except Exception:
            pass
    return obj


def _normalize_stats(obj):
    """
    Convert nested list-like stats (label/value pairs) into a DataFrame when possible.
    """
    def _find_table_like(item):
        # Direct table: list of rows, each row is list/tuple of scalars
        if isinstance(item, (list, tuple)) and item:
            if all(isinstance(r, (list, tuple)) for r in item):
                if all(all(not isinstance(x, (list, tuple, dict)) for x in r) for r in item):
                    return item
            # Search deeper
            for r in item:
                tbl = _find_table_like(r)
                if tbl:
                    return tbl
        if isinstance(item, dict):
            for v in item.values():
                tbl = _find_table_like(v)
                if tbl:
                    return tbl
        return None

    table = _find_table_like(obj)
    if table:
        # If first row is header, use it; else use generic columns
        first = table[0]
        consistent = all(len(r) == len(first) for r in table if isinstance(r, (list, tuple)))
        if consistent and all(not isinstance(x, (list, tuple, dict)) for x in first):
            header = [str(x) for x in first]
            data_rows = table[1:]
            if data_rows and all(len(r) == len(header) for r in data_rows):
                return pd.DataFrame([[str(x) for x in r] for r in data_rows], columns=header)
        # Fallback: treat as rows without header
        return pd.DataFrame([[str(x) for x in r] for r in table])
    return obj


def _render_stats_block(title: str, obj: Any):
    st.markdown(f"**{title}**")
    plain = _normalize_stats(_dash_children_to_plain(obj))
    try:
        if isinstance(plain, pd.DataFrame):
            st.dataframe(plain, use_container_width=True)
            return
        if isinstance(plain, list) and plain and all(isinstance(r, list) for r in plain):
            st.table(pd.DataFrame(plain))
            return
        if isinstance(plain, dict):
            st.json(plain)
            return
    except Exception:
        pass
    st.write(plain)


if page == "Portfolio":
    st.subheader("Portfolio builder")
    start_date = st.date_input("Start date", datetime.date.today() - datetime.timedelta(days=365 * 5))

    cols = st.columns(3)
    if cols[0].button("Add row"):
        st.session_state.rows.append({"ticker": "", "amount": 0.0})
    if cols[1].button("Remove last row") and st.session_state.rows:
        st.session_state.rows.pop()

    errors: List[str] = []
    for i, row in enumerate(st.session_state.rows):
        c1, c2 = st.columns([1, 1])
        row["ticker"] = c1.text_input(f"Ticker {i + 1}", value=row["ticker"], key=f"ticker_{i}").upper().strip()
        row["amount"] = c2.number_input(
            f"Amount {i + 1}",
            value=float(row["amount"] or 0),
            min_value=0.0,
            step=100.0,
            key=f"amount_{i}",
        )
        ok, msg = validate_ticker(row["ticker"]) if row["ticker"] else (False, "Missing ticker")
        if not ok:
            errors.append(f"Row {i + 1}: {msg}")

    submit = st.button("Run analysis", type="primary")
    if submit and errors:
        st.error("Please fix all errors before submitting:\n" + "\n".join(errors))
    elif submit:
        positions: Dict[str, float] = {}
        for r in st.session_state.rows:
            positions[r["ticker"]] = positions.get(r["ticker"], 0.0) + float(r["amount"])

        analyzer = PortfolioAnalyzer(positions, datetime.datetime.combine(start_date, datetime.time()))
        portfolio_fig = analyzer.graph_portfolio()
        individual_fig = analyzer.graph_individual_stocks()
        portfolio_stats = analyzer.get_portfolio_stats()
        stock_stats = analyzer.get_stock_stats()

        st.success("Analysis complete.")
        st.plotly_chart(portfolio_fig, use_container_width=True)
        st.plotly_chart(individual_fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            _render_stats_block("Portfolio stats", portfolio_stats)
        with c2:
            _render_stats_block("Stock stats", stock_stats)

        # Backtest against any ticker in the portfolio (default SPY if present)
        bench_options = list(dict.fromkeys(["SPY"] + list(positions.keys())))  # ensure SPY present and first
        default_bench = "SPY"
        benchmark_ticker = st.selectbox(
            "Benchmark ticker for backtest",
            bench_options,
            index=0
        )
        benchmark_amount = positions.get(benchmark_ticker, list(positions.values())[0] if positions else 10000.0)

        backtest_fig = build_backtest_fig(start_date, benchmark_amount, benchmark_ticker)
        if backtest_fig:
            st.plotly_chart(backtest_fig, use_container_width=True)
        else:
            st.info("Backtest data unavailable for the selected benchmark.")


# ---------- Model Playground ----------
if page == "Model Playground":
    st.subheader("Model Playground")
    st.caption("Enter insider trade details and run the 30d/60d/90d models.")

    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker", "KVYO").upper().strip()
    txn_type = c2.text_input("Transaction type (P - Purchase / S - Sale)", "P - Purchase")
    last_price = c3.text_input("Last price", "27.48")

    c7, c8 = st.columns(2)
    insider_name = c7.text_input("Insider name", "Fagnan Jeff")
    title = c8.text_input("Title", "Dir")

    c4, c5, c6 = st.columns(3)
    qty = c4.text_input("Qty", "4000")
    shares_held = c5.text_input("Shares held", "60000")
    owned = c6.text_input("Owned (%)", "10")

    value = st.text_input("Value ($)", "100000")

    if st.button("See results", type="primary"):
        payload = {
            "ticker": ticker,
            "transaction_type": txn_type,
            "last_price": last_price,
            "Qty": qty,
            "shares_held": shares_held,
            "Owned": owned,
            "Value": value,
        }
        results = {}
        errs = []
        for suffix in ["30d", "60d", "90d"]:
            pred, err = predict_suffix(payload, suffix)
            if err:
                errs.append(err)
            else:
                results[suffix] = pred

        if errs and not results:
            st.error("; ".join(errs))
        else:
            rows = []
            for suffix in ["30d", "60d", "90d"]:
                if suffix in results:
                    pred = results[suffix]
                    mae = MAE_BY_SUFFIX.get(suffix)
                    conf = None
                    if mae:
                        conf = 1 / (1 + mae / (abs(pred) + 1e-9))
                    rows.append({
                        "Horizon": suffix,
                        "Predicted Return": f"{pred:.4f}",
                        "Confidence": f"{conf * 100:.1f}%" if conf else "—",
                    })
            st.markdown("**Predicted returns**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            if errs:
                st.info("Warnings: " + "; ".join(errs))

