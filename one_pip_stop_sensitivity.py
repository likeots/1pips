#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-pip stop-loss sensitivity simulator with internal BOS detection."""
import argparse
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-pip stop-loss sensitivity simulator")
    parser.add_argument("--csv", type=str, required=True, help="Tick CSV: time,bid,ask")
    parser.add_argument("--pip-size", type=float, required=True, help="Instrument pip size")
    parser.add_argument("--swing-lookback", type=int, default=3, help="Lookback window for swings")
    parser.add_argument(
        "--sessions",
        choices=["on", "off"],
        default="off",
        help="If on, restrict entries to UTC hours 07..17",
    )
    parser.add_argument("--sl-margin-pips", type=float, default=1.0, help="SL margin beyond swing")
    parser.add_argument("--tp-margin-pips", type=float, default=1.0, help="TP margin beyond swing (swing mode)")
    parser.add_argument("--min-sl-pips", type=float, default=2.0, help="Minimum SL distance in pips")
    parser.add_argument("--min-tp-pips", type=float, default=2.0, help="Minimum TP distance in pips")
    parser.add_argument("--tp-mode", choices=["rr", "swing"], default="rr", help="Take-profit mode")
    parser.add_argument("--rr", type=float, default=1.0, help="Risk-reward multiple for rr mode")
    parser.add_argument("--tf", type=int, default=1, help="Keep every Nth trade (1 = keep all)")
    return parser.parse_args()


def robust_parse_times(series: pd.Series) -> pd.Series:
    times = pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
    mask = times.isna()
    if mask.any():
        times.loc[mask] = pd.to_datetime(series.loc[mask], utc=True, errors="coerce")
    return times


def load_ticks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"time", "bid", "ask"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise SystemExit(f"Missing required columns: {missing}")

    df["time"] = robust_parse_times(df["time"])
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df = df.dropna(subset=["time", "bid", "ask"]).copy()
    if df.empty:
        raise SystemExit("No valid tick rows after cleaning")

    df.sort_values("time", inplace=True)
    df.drop_duplicates(subset="time", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_candles(ticks: pd.DataFrame) -> pd.DataFrame:
    ticks = ticks.copy()
    ticks["mid"] = (ticks["bid"] + ticks["ask"]) / 2.0
    candles = (
        ticks.set_index("time")["mid"].resample("1min").ohlc().dropna(how="any")
    )
    candles.columns = ["Open", "High", "Low", "Close"]
    return candles


def detect_swings(candles: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(candles)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    highs = candles["High"].to_numpy()
    lows = candles["Low"].to_numpy()

    if lookback <= 0:
        return swing_high, swing_low

    for i in range(lookback, n - lookback):
        high = highs[i]
        low = lows[i]
        if np.all(high > highs[i - lookback : i]) and np.all(high > highs[i + 1 : i + 1 + lookback]):
            swing_high[i] = True
        if np.all(low < lows[i - lookback : i]) and np.all(low < lows[i + 1 : i + 1 + lookback]):
            swing_low[i] = True
    return swing_high, swing_low


def find_entry_tick_index(tick_times: np.ndarray, entry_time: pd.Timestamp) -> int:
    idx = int(np.searchsorted(tick_times, entry_time, side="left"))
    if idx >= len(tick_times):
        return len(tick_times) - 1
    return idx


def simulate_trade(
    ticks: pd.DataFrame,
    entry_idx: int,
    direction: int,
    stop_price: float,
    tp_price: float,
) -> bool:
    if entry_idx < 0 or entry_idx >= len(ticks):
        return False

    for i in range(entry_idx + 1, len(ticks)):
        bid = ticks.iloc[i]["bid"]
        ask = ticks.iloc[i]["ask"]
        if direction == 1:
            if bid >= tp_price:
                return True
            if bid <= stop_price:
                return False
        else:
            if ask <= tp_price:
                return True
            if ask >= stop_price:
                return False
    return False


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    args = parse_args()

    if args.sl_margin_pips <= 0:
        raise SystemExit("--sl-margin-pips must be positive to keep stops beyond swings")
    if args.tp_mode == "rr":
        if not (0.5 <= args.rr <= 3.0):
            raise SystemExit("--rr must be within [0.5, 3.0]")
    pip = float(args.pip_size)
    tf = max(1, int(args.tf))

    ticks = load_ticks(args.csv)
    candles = build_candles(ticks)
    if candles.empty:
        raise SystemExit("No candles constructed from ticks")

    swing_high_flags, swing_low_flags = detect_swings(candles, int(args.swing_lookback))

    highs = candles["High"].to_numpy()
    lows = candles["Low"].to_numpy()
    closes = candles["Close"].to_numpy()
    tick_times = ticks["time"].to_numpy()

    bos_bullish = bos_bearish = 0
    filtered_session = filtered_sl = filtered_tp = 0
    skipped_no_opposite = 0

    trades: List[Dict[str, object]] = []
    trade_counter = 0

    last_swing_high_price = None
    last_swing_low_price = None
    last_broken_high = None
    last_broken_low = None

    for i in range(len(candles)):
        if swing_high_flags[i]:
            last_swing_high_price = float(highs[i])
            last_broken_high = None
        if swing_low_flags[i]:
            last_swing_low_price = float(lows[i])
            last_broken_low = None

        direction = 0
        broken_swing_price = None
        protective_price = None

        if (
            last_swing_high_price is not None
            and closes[i] > last_swing_high_price
            and last_swing_low_price is not None
            and last_broken_high != last_swing_high_price
        ):
            direction = 1
            broken_swing_price = last_swing_high_price
            protective_price = last_swing_low_price
            bos_bullish += 1
            last_broken_high = last_swing_high_price
        elif (
            last_swing_low_price is not None
            and closes[i] < last_swing_low_price
            and last_swing_high_price is not None
            and last_broken_low != last_swing_low_price
        ):
            direction = -1
            broken_swing_price = last_swing_low_price
            protective_price = last_swing_high_price
            bos_bearish += 1
            last_broken_low = last_swing_low_price

        if direction == 0:
            continue

        if protective_price is None:
            skipped_no_opposite += 1
            continue

        entry_minute_idx = i + 1 if i + 1 < len(candles) else i
        entry_time = candles.index[entry_minute_idx]

        if args.sessions == "on" and not (7 <= entry_time.hour <= 17):
            filtered_session += 1
            continue

        entry_tick_idx = find_entry_tick_index(tick_times, entry_time)
        entry_tick = ticks.iloc[entry_tick_idx]
        entry_price = float(entry_tick["ask"] if direction == 1 else entry_tick["bid"])

        if direction == 1:
            swing_stop_price = float(protective_price)
            stop_price_base = swing_stop_price - args.sl_margin_pips * pip
            stop_pips = (entry_price - stop_price_base) / pip
        else:
            swing_stop_price = float(protective_price)
            stop_price_base = swing_stop_price + args.sl_margin_pips * pip
            stop_pips = (stop_price_base - entry_price) / pip

        if stop_pips <= 0 or stop_pips + 1e-9 < args.min_sl_pips:
            filtered_sl += 1
            continue

        if args.tp_mode == "rr":
            tp_pips = stop_pips * args.rr
            if tp_pips <= 0 or tp_pips + 1e-9 < args.min_tp_pips:
                filtered_tp += 1
                continue
            if direction == 1:
                tp_price = entry_price + tp_pips * pip
            else:
                tp_price = entry_price - tp_pips * pip
        else:  # swing mode
            if broken_swing_price is None:
                filtered_tp += 1
                continue
            if direction == 1:
                tp_price = broken_swing_price + args.tp_margin_pips * pip
                tp_pips = (tp_price - entry_price) / pip
            else:
                tp_price = broken_swing_price - args.tp_margin_pips * pip
                tp_pips = (entry_price - tp_price) / pip
            if tp_pips <= 0 or tp_pips + 1e-9 < args.min_tp_pips:
                filtered_tp += 1
                continue

        stop_price_plus1 = stop_price_base - pip if direction == 1 else stop_price_base + pip

        trade_counter += 1
        if trade_counter % tf != 0:
            continue

        trades.append(
            {
                "entry_time": entry_time.isoformat(),
                "direction": "long" if direction == 1 else "short",
                "entry_price": entry_price,
                "swing_stop_price": swing_stop_price,
                "stop_price_base": stop_price_base,
                "stop_price_plus1": stop_price_plus1,
                "tp_price": tp_price,
                "stop_pips": stop_pips,
                "tp_pips": tp_pips,
                "entry_idx": entry_tick_idx,
                "direction_sign": direction,
            }
        )

    if not trades:
        print(f"Bullish BOS detected: {bos_bullish}")
        print(f"Bearish BOS detected: {bos_bearish}")
        print(f"Trades filtered by sessions: {filtered_session}")
        print(f"Trades filtered by min_sl_pips: {filtered_sl}")
        print(f"Trades filtered by min_tp_pips: {filtered_tp}")
        print(f"Trades skipped (no protective swing): {skipped_no_opposite}")
        print("No trades qualified. Summary: trades=0, base_wr=0.0, plus_wr=0.0, delta=0.0")
        print("McNemar b: 0, c: 0, exact p-value: 1.0")
        return

    results: List[Dict[str, object]] = []
    base_wins = plus1_wins = 0
    b = c = 0

    for trade in tqdm(trades, desc="Simulating", unit="trade"):
        base_win = simulate_trade(
            ticks, trade["entry_idx"], trade["direction_sign"], trade["stop_price_base"], trade["tp_price"]
        )
        plus_win = simulate_trade(
            ticks, trade["entry_idx"], trade["direction_sign"], trade["stop_price_plus1"], trade["tp_price"]
        )
        base_wins += int(base_win)
        plus1_wins += int(plus_win)
        if (not base_win) and plus_win:
            b += 1
        elif base_win and (not plus_win):
            c += 1

        results.append(
            {
                "entry_time": trade["entry_time"],
                "direction": trade["direction"],
                "entry_price": trade["entry_price"],
                "swing_stop_price": trade["swing_stop_price"],
                "stop_price_base": trade["stop_price_base"],
                "stop_price_plus1": trade["stop_price_plus1"],
                "tp_price": trade["tp_price"],
                "stop_pips": trade["stop_pips"],
                "tp_pips": trade["tp_pips"],
                "won_base": bool(base_win),
                "won_plus1": bool(plus_win),
            }
        )

    trades_count = len(results)
    base_wr = base_wins / trades_count
    plus_wr = plus1_wins / trades_count
    delta_wr = plus_wr - base_wr
    p_value = mcnemar_exact(b, c)

    print(f"Bullish BOS detected: {bos_bullish}")
    print(f"Bearish BOS detected: {bos_bearish}")
    print(f"Trades filtered by sessions: {filtered_session}")
    print(f"Trades filtered by min_sl_pips: {filtered_sl}")
    print(f"Trades filtered by min_tp_pips: {filtered_tp}")
    print(f"Trades skipped (no protective swing): {skipped_no_opposite}")

    print(f"Trades: {trades_count}")
    print(f"Base win rate: {base_wr:.6f}")
    print(f"Plus1 win rate: {plus_wr:.6f}")
    print(f"Delta win rate: {delta_wr:.6f}")
    print(f"McNemar b (base loss, plus1 win): {b}")
    print(f"McNemar c (base win, plus1 loss): {c}")
    print(f"McNemar exact p-value: {p_value:.6f}")

    pd.DataFrame(results).to_csv("one_pip_results_detailed.csv", index=False)
    print("Detailed results saved to one_pip_results_detailed.csv")


if __name__ == "__main__":
    main()
