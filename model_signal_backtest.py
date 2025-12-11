"""
Backtest model-driven signals using transaction_date and yfinance prices.

Inputs (test_data.csv expected columns):
  - Ticker
  - transaction_date
  - pred_30d, pred_60d, pred_90d (model predictions; >0 => long, else short)

Logic:
  - Entry: next trading day after transaction_date, price = Open
  - Exit: transaction_date + horizon days -> next available trading day, price = Close
  - Long return = (exit - entry) / entry
    Short return = (entry - exit) / entry
  - Positions opened with equal-weight of available cash per day, per horizon (10k start each), closed on exit date.

Outputs:
  - model_signal_backtest_results.csv: equity curves per horizon + summary row
  - test_data_with_realized_returns.csv: augmented signals with entry/exit, prices, returns, signals
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import time

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent

START_CAPITAL = 10_000.0
HORIZONS = [30, 60, 90]
PRED_COLS = {30: "pred_30d", 60: "pred_60d", 90: "pred_90d"}


def load_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Signals file not found: {path}")
    df = pd.read_csv(path)
    required = {"Ticker", "transaction_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for h, col in PRED_COLS.items():
        if col not in df.columns:
            raise ValueError(f"Missing prediction column for {h}d: {col}")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["Ticker", "transaction_date"]).reset_index(drop=True)
    return df


def fetch_prices_for_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Download daily OHLC for [start, end] (inclusive end via +1 day).
    Handles potential MultiIndex columns from yfinance.
    """
    data = yf.download(
        ticker,
        start=start.date(),
        end=(end + pd.Timedelta(days=1)).date(),
        progress=False,
        auto_adjust=False,
        actions=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        # try to extract Close/Open for this ticker
        if ("Open", ticker) in data.columns and ("Close", ticker) in data.columns:
            data = data[[("Open", ticker), ("Close", ticker)]]
            data.columns = ["Open", "Close"]
        elif "Open" in data.columns.get_level_values(0) and "Close" in data.columns.get_level_values(0):
            data = data.xs("Open", level=0, axis=1).to_frame("Open").join(
                data.xs("Close", level=0, axis=1).to_frame("Close"), how="inner"
            )
        else:
            data = data.droplevel(0, axis=1)
    if data is None or data.empty or "Open" not in data or "Close" not in data:
        return pd.DataFrame()
    out = data[["Open", "Close"]].copy()
    out.index = pd.to_datetime(out.index).normalize()
    return out


def fetch_prices_batch(
    tickers: List[str], start: pd.Timestamp, end: pd.Timestamp, batch_size: int = 50, sleep_s: float = 0.15
) -> Dict[str, pd.DataFrame]:
    """
    Batch download prices for many tickers to reduce API calls.
    Returns dict ticker -> OHLC dataframe (Open, Close, normalized index).
    """
    prices: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return prices

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            data = yf.download(
                tickers=batch,
                start=start.date(),
                end=(end + pd.Timedelta(days=1)).date(),
                progress=False,
                auto_adjust=False,
                actions=False,
                threads=False,
                group_by="ticker",
            )
        except Exception:
            continue
        # Single ticker returns a regular frame
        if isinstance(data, pd.DataFrame) and not isinstance(data.columns, pd.MultiIndex):
            df = data[["Open", "Close"]].copy()
            df.index = pd.to_datetime(df.index).normalize()
            prices[batch[0]] = df
            continue

        # MultiIndex columns: level 0 = fields, level 1 = ticker OR vice versa depending on yfinance version
        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
            # Normalize to (field, ticker)
            cols = data.columns
            if cols.names[0] in ["", "Price"] and cols.names[1] is None:
                # unexpected, skip
                continue
            # If first level are fields
            if set(cols.get_level_values(0)) >= {"Open", "Close"}:
                for t in batch:
                    if t not in cols.get_level_values(1):
                        continue
                    try:
                        df_t = data.xs(t, level=1, axis=1)[["Open", "Close"]].copy()
                        df_t.index = pd.to_datetime(df_t.index).normalize()
                        prices[t] = df_t
                    except Exception:
                        continue
            else:
                # Level 0 = ticker, level 1 = field
                for t in batch:
                    if t not in cols.get_level_values(0):
                        continue
                    try:
                        df_t = data.xs(t, level=0, axis=1)[["Open", "Close"]].copy()
                        df_t.index = pd.to_datetime(df_t.index).normalize()
                        prices[t] = df_t
                    except Exception:
                        continue
        # Throttle to stay under rate limits
        time.sleep(sleep_s)
    return prices


def next_trading_day_after(df_px: pd.DataFrame, date: pd.Timestamp) -> pd.Timestamp | None:
    """Return the first trading day strictly after date."""
    candidates = df_px.loc[df_px.index > date]
    if candidates.empty:
        return None
    return candidates.index[0]


def next_trading_day_on_or_after(df_px: pd.DataFrame, date: pd.Timestamp) -> pd.Timestamp | None:
    """Return the first trading day on/after date."""
    candidates = df_px.loc[df_px.index >= date]
    if candidates.empty:
        return None
    return candidates.index[0]


def prepare_signals_with_returns(df: pd.DataFrame, batch_size: int = 50, sleep_s: float = 0.15) -> pd.DataFrame:
    """
    For each signal row and each horizon, compute entry/exit exec dates, prices, returns, signals.
    Returns a wide DataFrame with appended columns per horizon.
    """
    results: List[Dict] = []
    all_min = df["transaction_date"].min() - pd.Timedelta(days=2)
    all_max = df["transaction_date"].max() + pd.Timedelta(days=max(HORIZONS) + 7)
    tickers = sorted(df["Ticker"].unique())
    price_cache = fetch_prices_batch(tickers, all_min, all_max, batch_size=batch_size, sleep_s=sleep_s)

    for ticker, g in df.groupby("Ticker"):
        px = price_cache.get(ticker, pd.DataFrame())
        for _, row in g.iterrows():
            base = row.to_dict()
            for h in HORIZONS:
                pred_col = PRED_COLS[h]
                signal_val = 1 if row[pred_col] > 0 else -1
                entry_day = next_trading_day_after(px, row["transaction_date"])
                exit_target = row["transaction_date"] + pd.Timedelta(days=h)
                exit_day = next_trading_day_on_or_after(px, exit_target) if entry_day is not None else None
                if entry_day is None or exit_day is None:
                    entry_px = exit_px = np.nan
                    long_ret = short_ret = np.nan
                else:
                    entry_px = px.loc[entry_day, "Open"]
                    exit_px = px.loc[exit_day, "Close"]
                    if pd.isna(entry_px) or pd.isna(exit_px):
                        long_ret = short_ret = np.nan
                    else:
                        long_ret = (exit_px - entry_px) / entry_px
                        short_ret = (entry_px - exit_px) / entry_px
                base.update(
                    {
                        f"entry_date_{h}d": entry_day,
                        f"exit_date_{h}d": exit_day,
                        f"entry_price_{h}d": entry_px,
                        f"exit_price_{h}d": exit_px,
                        f"signal_{h}d": signal_val,
                        f"long_return_{h}d": long_ret,
                        f"short_return_{h}d": short_ret,
                    }
                )
            results.append(base.copy())
    return pd.DataFrame(results)


@dataclass
class Position:
    close_date: pd.Timestamp
    amount: float
    ret: float
    signal: int  # +1 long, -1 short


def run_pnl(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simulate PnL path for one horizon using precomputed entry/exit and signal columns."""
    cols = {
        "entry_date": f"entry_date_{horizon}d",
        "exit_date": f"exit_date_{horizon}d",
        "ret_long": f"long_return_{horizon}d",
        "ret_short": f"short_return_{horizon}d",
        "signal": f"signal_{horizon}d",
    }
    df_h = df.dropna(subset=[cols["entry_date"], cols["exit_date"]]).copy()
    df_h = df_h.sort_values(cols["entry_date"])
    if df_h.empty:
        return pd.DataFrame(columns=["date", "cash", "positions_value", "total_equity", "open_positions"])

    cash = START_CAPITAL
    positions: List[Position] = []
    logs: List[Dict] = []

    event_dates = pd.Index(
        pd.concat([df_h[cols["entry_date"]], df_h[cols["exit_date"]]])
    ).unique().sort_values()
    trades_by_date = df_h.groupby(cols["entry_date"])

    for current_date in event_dates:
        # close positions
        remaining = []
        for pos in positions:
            if pos.close_date <= current_date:
                cash += pos.amount * (1.0 + pos.signal * pos.ret)
            else:
                remaining.append(pos)
        positions = remaining

        # open new positions
        if current_date in trades_by_date.indices and cash > 0:
            trades_today = trades_by_date.get_group(current_date)
            n_new = len(trades_today)
            if n_new > 0 and cash > 0:
                invest_per = cash / n_new
                for _, row in trades_today.iterrows():
                    ret_val = row[cols["ret_long"]] if row[cols["signal"]] == 1 else row[cols["ret_short"]]
                    if pd.isna(ret_val):
                        continue
                    positions.append(
                        Position(
                            close_date=row[cols["exit_date"]],
                            amount=invest_per,
                            ret=float(ret_val),
                            signal=int(row[cols["signal"]]),
                        )
                    )
                cash -= invest_per * n_new

        positions_value = sum(p.amount for p in positions)
        total_equity = cash + positions_value
        logs.append(
            {
                "date": current_date,
                "cash": cash,
                "positions_value": positions_value,
                "total_equity": total_equity,
                "open_positions": len(positions),
            }
        )

    return pd.DataFrame(logs).sort_values("date").dropna()


def merge_equity_logs(logs_by_h: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    all_dates = sorted(pd.unique(pd.concat([log["date"] for log in logs_by_h.values() if not log.empty])))
    equity_df = pd.DataFrame({"date": all_dates})
    for h, log in logs_by_h.items():
        if log.empty:
            equity_df[f"cash_{h}d"] = START_CAPITAL
            equity_df[f"positions_value_{h}d"] = 0.0
            equity_df[f"total_equity_{h}d"] = START_CAPITAL
            equity_df[f"open_positions_{h}d"] = 0
            continue
        merged = pd.merge_asof(equity_df[["date"]], log.sort_values("date"), on="date", direction="backward")
        equity_df[f"cash_{h}d"] = merged["cash"].ffill().fillna(START_CAPITAL)
        equity_df[f"positions_value_{h}d"] = merged["positions_value"].fillna(0.0)
        equity_df[f"total_equity_{h}d"] = merged["total_equity"].ffill().fillna(START_CAPITAL)
        equity_df[f"open_positions_{h}d"] = merged["open_positions"].ffill().fillna(0).astype(int)
    return equity_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest model signals using yfinance.")
    parser.add_argument(
        "--signals",
        type=Path,
        default=ROOT / "test_data.csv",
        help="Path to signals CSV (needs Ticker, transaction_date, pred_30d/pred_60d/pred_90d)",
    )
    parser.add_argument(
        "--out-signals",
        type=Path,
        default=ROOT / "test_data_with_realized_returns.csv",
        help="Where to write augmented signals with realized returns",
    )
    parser.add_argument(
        "--out-results",
        type=Path,
        default=ROOT / "model_signal_backtest_results.csv",
        help="Where to write equity curves and summary",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for yfinance downloads",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Sleep seconds between batches to respect rate limits",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        signals = load_signals(args.signals)
    except FileNotFoundError as exc:
        raise SystemExit(f"[error] {exc}. Please provide --signals pointing to your CSV.") from exc
    if signals.empty:
        raise ValueError("No signals after cleaning.")

    signals_aug = prepare_signals_with_returns(
        signals, batch_size=args.batch_size, sleep_s=args.sleep
    )
    signals_aug.to_csv(args.out_signals, index=False)
    print(f"[done] wrote {args.out_signals}")

    horizon_logs: Dict[int, pd.DataFrame] = {}
    summaries = {}
    for h in HORIZONS:
        log = run_pnl(signals_aug, h)
        horizon_logs[h] = log
        final_equity = log["total_equity"].iloc[-1] if not log.empty else START_CAPITAL
        summaries[h] = {
            "final_equity": final_equity,
            "return": final_equity / START_CAPITAL - 1.0,
        }
        print(f"[h={h}d] events={len(log)}, final_equity={final_equity:.2f}, ret={summaries[h]['return']:.2%}")

    equity_df = merge_equity_logs(horizon_logs)
    summary_row = {"date": "SUMMARY"}
    for h in HORIZONS:
        summary_row[f"total_equity_{h}d"] = summaries[h]["final_equity"]
        summary_row[f"return_{h}d"] = summaries[h]["return"]
    out_df = pd.concat([equity_df, pd.DataFrame([summary_row])], ignore_index=True)
    out_df.to_csv(args.out_results, index=False)
    print(f"[done] wrote {args.out_results}")


if __name__ == "__main__":
    main()
