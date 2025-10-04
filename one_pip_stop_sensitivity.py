#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-pip stop-loss sensitivity simulator with internal BOS detection."""
import argparse
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


SUMMARY_PATH = Path("one_pip_results_summary.json")
RESULTS_PREFIX = "one_pip_results_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-pip stop-loss sensitivity simulator")
    parser.add_argument("--csv", type=str, required=True, help="Tick CSV: time,bid,ask")
    parser.add_argument(
        "--instrument",
        type=str,
        default="EURUSD",
        help="Instrument symbol used for output files and summaries",
    )
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
    parser.add_argument(
        "--tp-mode",
        type=str,
        default="rr",
        help="Take-profit mode (rr|swing). Aliases: symmetric→rr, structure→swing.",
    )
    parser.add_argument("--rr", type=float, default=1.0, help="Risk-reward multiple for rr mode")
    parser.add_argument("--tf", type=int, default=1, help="Keep every Nth trade (1 = keep all)")
    args = parser.parse_args()

    tp_mode_raw = str(args.tp_mode).strip().lower()
    tp_mode_map = {
        "rr": "rr",
        "risk_reward": "rr",
        "risk-reward": "rr",
        "symmetric": "rr",
        "swing": "swing",
        "structure": "swing",
    }
    if tp_mode_raw not in tp_mode_map:
        parser.error(
            "--tp-mode must be one of: rr, swing, symmetric, structure"
        )
    args.tp_mode = tp_mode_map[tp_mode_raw]
    instrument = args.instrument.strip().upper()
    if not instrument:
        parser.error("--instrument must be a non-empty symbol")
    args.instrument = instrument
    return args


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

    if lookback * 2 >= n:
        return swing_high, swing_low

    # Use numpy sliding windows to minimise Python overhead per candle.
    window = lookback * 2 + 1
    high_windows = np.lib.stride_tricks.sliding_window_view(highs, window)
    low_windows = np.lib.stride_tricks.sliding_window_view(lows, window)

    centers = slice(lookback, n - lookback)
    center_highs = highs[centers]
    center_lows = lows[centers]

    left_high = high_windows[:, :lookback]
    right_high = high_windows[:, lookback + 1 :]
    left_low = low_windows[:, :lookback]
    right_low = low_windows[:, lookback + 1 :]

    swing_high[lookback : n - lookback] = (center_highs[:, None] > left_high).all(axis=1) & (
        center_highs[:, None] > right_high
    ).all(axis=1)
    swing_low[lookback : n - lookback] = (center_lows[:, None] < left_low).all(axis=1) & (
        center_lows[:, None] < right_low
    ).all(axis=1)
    return swing_high, swing_low


def find_entry_tick_index(tick_times: np.ndarray, entry_time: pd.Timestamp) -> int:
    idx = int(np.searchsorted(tick_times, entry_time, side="left"))
    if idx >= len(tick_times):
        return len(tick_times) - 1
    return idx


def evaluate_variants(
    bids: np.ndarray,
    asks: np.ndarray,
    entry_idx: int,
    direction: int,
    stop_prices: Iterable[float],
    tp_price: float,
) -> List[bool]:
    """Return win flags for the provided stop prices."""

    stops = list(stop_prices)
    if entry_idx >= len(bids) - 1:
        return [False for _ in stops]

    results: List[bool] = []
    if direction == 1:
        path = bids[entry_idx + 1 :]
        tp_hits = np.flatnonzero(path >= tp_price)
        tp_hit_idx = int(tp_hits[0]) if tp_hits.size else None
        for stop_price in stops:
            sl_hits = np.flatnonzero(path <= stop_price)
            sl_hit_idx = int(sl_hits[0]) if sl_hits.size else None
            if tp_hit_idx is None and sl_hit_idx is None:
                results.append(False)
            elif tp_hit_idx is None:
                results.append(False)
            elif sl_hit_idx is None:
                results.append(True)
            else:
                results.append(tp_hit_idx <= sl_hit_idx)
    else:
        path = asks[entry_idx + 1 :]
        tp_hits = np.flatnonzero(path <= tp_price)
        tp_hit_idx = int(tp_hits[0]) if tp_hits.size else None
        for stop_price in stops:
            sl_hits = np.flatnonzero(path >= stop_price)
            sl_hit_idx = int(sl_hits[0]) if sl_hits.size else None
            if tp_hit_idx is None and sl_hit_idx is None:
                results.append(False)
            elif tp_hit_idx is None:
                results.append(False)
            elif sl_hit_idx is None:
                results.append(True)
            else:
                results.append(tp_hit_idx <= sl_hit_idx)
    return results


def mcnemar_exact(b: int, c: int) -> Decimal:
    n = b + c
    if n == 0:
        return Decimal(1)

    limit = min(b, c)
    with localcontext() as ctx:
        ctx.prec = max(28, int(n * math.log10(2)) + 10)
        inv_two_pow = Decimal(1) / (Decimal(2) ** n)
        tail = Decimal(0)
        for k in range(limit + 1):
            tail += Decimal(math.comb(n, k)) * inv_two_pow
        p_value = tail * 2
    return p_value if p_value <= 1 else Decimal(1)


def format_p_value(p_value: Decimal) -> str:
    threshold = Decimal("1e-6")
    return f"{p_value:.3E}" if p_value < threshold else f"{p_value:.6f}"


def _sanitize_instrument_filename(symbol: str) -> str:
    safe = "".join(ch for ch in symbol if ch.isalnum() or ch in ("_", "-"))
    return safe or "INSTRUMENT"


def _load_summary_blob() -> Dict[str, object]:
    if SUMMARY_PATH.exists():
        try:
            payload = json.loads(SUMMARY_PATH.read_text())
            if isinstance(payload, dict):
                payload.setdefault("instruments", {})
                return payload
        except json.JSONDecodeError:
            pass
    return {"instruments": {}}


def _compute_overall(instruments: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    total_trades = sum(int(entry.get("trades", 0)) for entry in instruments.values())
    total_swing_wins = sum(int(entry.get("swing_wins", 0)) for entry in instruments.values())
    total_plus_wins = sum(int(entry.get("plus1_wins", 0)) for entry in instruments.values())
    total_b = sum(int(entry.get("b", 0)) for entry in instruments.values())
    total_c = sum(int(entry.get("c", 0)) for entry in instruments.values())
    if total_trades:
        swing_wr = total_swing_wins / total_trades
        plus_wr = total_plus_wins / total_trades
        delta_wr = plus_wr - swing_wr
    else:
        swing_wr = plus_wr = delta_wr = 0.0
    p_value = format_p_value(mcnemar_exact(total_b, total_c))
    return {
        "trades": total_trades,
        "swing_wins": total_swing_wins,
        "plus1_wins": total_plus_wins,
        "swing_win_rate": swing_wr,
        "plus1_win_rate": plus_wr,
        "delta_win_rate": delta_wr,
        "b": total_b,
        "c": total_c,
        "p_value": p_value,
    }


def _persist_summary(instrument: str, summary_entry: Dict[str, object]) -> None:
    payload = _load_summary_blob()
    instruments = payload.setdefault("instruments", {})
    instruments[instrument] = summary_entry
    payload["overall"] = _compute_overall(instruments)
    payload["updated_at"] = pd.Timestamp.utcnow().isoformat()
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()

    if args.sl_margin_pips <= 0:
        raise SystemExit("--sl-margin-pips must be positive to keep stops beyond swings")
    if args.tp_mode == "rr":
        if not (0.5 <= args.rr <= 3.0):
            raise SystemExit("--rr must be within [0.5, 3.0]")
    instrument = args.instrument
    safe_instrument = _sanitize_instrument_filename(instrument)
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
    tick_bids = ticks["bid"].to_numpy()
    tick_asks = ticks["ask"].to_numpy()

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
        entry_price = float(tick_asks[entry_tick_idx] if direction == 1 else tick_bids[entry_tick_idx])

        swing_extreme = float(protective_price)
        if direction == 1:
            stop_price_swing = swing_extreme - args.sl_margin_pips * pip
            stop_price_plus1 = stop_price_swing - pip
            stop_pips_swing = (entry_price - stop_price_swing) / pip
            stop_pips_plus1 = (entry_price - stop_price_plus1) / pip
        else:
            stop_price_swing = swing_extreme + args.sl_margin_pips * pip
            stop_price_plus1 = stop_price_swing + pip
            stop_pips_swing = (stop_price_swing - entry_price) / pip
            stop_pips_plus1 = (stop_price_plus1 - entry_price) / pip

        if stop_pips_swing <= 0 or stop_pips_swing + 1e-9 < args.min_sl_pips:
            filtered_sl += 1
            continue

        if args.tp_mode == "rr":
            tp_pips = stop_pips_swing * args.rr
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

        trade_counter += 1
        if trade_counter % tf != 0:
            continue

        trades.append(
            {
                "instrument": instrument,
                "entry_time": entry_time.isoformat(),
                "direction": "long" if direction == 1 else "short",
                "entry_price": entry_price,
                "swing_extreme": swing_extreme,
                "stop_price_swing": stop_price_swing,
                "stop_price_plus1": stop_price_plus1,
                "tp_price": tp_price,
                "stop_pips_swing": stop_pips_swing,
                "stop_pips_plus1": stop_pips_plus1,
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
        print("No trades qualified. Summary: trades=0, swing_wr=0.0, plus_wr=0.0, delta=0.0")
        zero_p = format_p_value(Decimal(1))
        print("McNemar b: 0, c: 0, exact p-value: " + zero_p)

        empty_df = pd.DataFrame(
            columns=[
                "instrument",
                "entry_time",
                "direction",
                "entry_price",
                "protective_swing",
                "stop_price_swing",
                "stop_pips_swing",
                "stop_price_plus1",
                "stop_pips_plus1",
                "tp_price",
                "tp_pips",
                "won_swing",
                "won_plus1",
            ]
        )
        detailed_path = Path(f"{RESULTS_PREFIX}{safe_instrument}.csv")
        empty_df.to_csv(detailed_path, index=False)
        empty_df.to_csv("one_pip_results_detailed.csv", index=False)
        summary_entry = {
            "instrument": instrument,
            "trades": 0,
            "swing_wins": 0,
            "plus1_wins": 0,
            "swing_win_rate": 0.0,
            "plus1_win_rate": 0.0,
            "delta_win_rate": 0.0,
            "b": 0,
            "c": 0,
            "p_value": zero_p,
            "last_run": pd.Timestamp.utcnow().isoformat(),
            "tp_mode": args.tp_mode,
            "rr": args.rr if args.tp_mode == "rr" else None,
            "sl_margin_pips": args.sl_margin_pips,
            "tp_margin_pips": args.tp_margin_pips,
            "min_sl_pips": args.min_sl_pips,
            "min_tp_pips": args.min_tp_pips,
        }
        _persist_summary(instrument, summary_entry)
        print(
            "Detailed results saved to "
            f"{detailed_path.name} (and legacy one_pip_results_detailed.csv)"
        )
        print(f"Unified summary updated: {SUMMARY_PATH.name}")
        return

    results: List[Dict[str, object]] = []
    swing_wins = plus1_wins = 0
    b = c = 0

    for trade in tqdm(trades, desc="Simulating", unit="trade"):
        swing_win, plus_win = evaluate_variants(
            tick_bids,
            tick_asks,
            trade["entry_idx"],
            trade["direction_sign"],
            [trade["stop_price_swing"], trade["stop_price_plus1"]],
            trade["tp_price"],
        )
        swing_wins += int(swing_win)
        plus1_wins += int(plus_win)
        if (not swing_win) and plus_win:
            b += 1
        elif swing_win and (not plus_win):
            c += 1

        results.append(
            {
                "instrument": instrument,
                "entry_time": trade["entry_time"],
                "direction": trade["direction"],
                "entry_price": trade["entry_price"],
                "protective_swing": trade["swing_extreme"],
                "stop_price_swing": trade["stop_price_swing"],
                "stop_price_plus1": trade["stop_price_plus1"],
                "tp_price": trade["tp_price"],
                "stop_pips_swing": trade["stop_pips_swing"],
                "stop_pips_plus1": trade["stop_pips_plus1"],
                "tp_pips": trade["tp_pips"],
                "won_swing": bool(swing_win),
                "won_plus1": bool(plus_win),
            }
        )

    trades_count = len(results)
    swing_wr = swing_wins / trades_count
    plus_wr = plus1_wins / trades_count
    delta_wr = plus_wr - swing_wr
    p_value = mcnemar_exact(b, c)

    print(f"Bullish BOS detected: {bos_bullish}")
    print(f"Bearish BOS detected: {bos_bearish}")
    print(f"Trades filtered by sessions: {filtered_session}")
    print(f"Trades filtered by min_sl_pips: {filtered_sl}")
    print(f"Trades filtered by min_tp_pips: {filtered_tp}")
    print(f"Trades skipped (no protective swing): {skipped_no_opposite}")

    print(f"Trades: {trades_count}")
    print(f"Swing win rate: {swing_wr:.6f}")
    print(f"Plus1 win rate: {plus_wr:.6f}")
    print(f"Delta win rate: {delta_wr:.6f}")
    print(f"McNemar b (swing loss, plus1 win): {b}")
    print(f"McNemar c (swing win, plus1 loss): {c}")
    print(f"McNemar exact p-value: {format_p_value(p_value)}")

    results_df = (
        pd.DataFrame(results)[
            [
                "instrument",
                "entry_time",
                "direction",
                "entry_price",
                "protective_swing",
                "stop_price_swing",
                "stop_pips_swing",
                "stop_price_plus1",
                "stop_pips_plus1",
                "tp_price",
                "tp_pips",
                "won_swing",
                "won_plus1",
            ]
        ]
    ).copy()
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    results_df.loc[:, numeric_cols] = results_df.loc[:, numeric_cols].round(6)
    detailed_path = Path(f"{RESULTS_PREFIX}{safe_instrument}.csv")
    results_df.to_csv(detailed_path, index=False)
    # Maintain the legacy file name for backward compatibility.
    results_df.to_csv("one_pip_results_detailed.csv", index=False)

    summary_entry = {
        "instrument": instrument,
        "trades": trades_count,
        "swing_wins": swing_wins,
        "plus1_wins": plus1_wins,
        "swing_win_rate": swing_wr,
        "plus1_win_rate": plus_wr,
        "delta_win_rate": delta_wr,
        "b": b,
        "c": c,
        "p_value": format_p_value(p_value),
        "last_run": pd.Timestamp.utcnow().isoformat(),
        "tp_mode": args.tp_mode,
        "rr": args.rr if args.tp_mode == "rr" else None,
        "sl_margin_pips": args.sl_margin_pips,
        "tp_margin_pips": args.tp_margin_pips,
        "min_sl_pips": args.min_sl_pips,
        "min_tp_pips": args.min_tp_pips,
    }
    _persist_summary(instrument, summary_entry)

    print(
        "Detailed results saved to "
        f"{detailed_path.name} (and legacy one_pip_results_detailed.csv)"
    )
    print(f"Unified summary updated: {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
