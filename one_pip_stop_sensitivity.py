#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-pip stop-loss sensitivity simulator (TP modes + RR filter)

Pipeline:
1) Load tick CSV: time (ISO8601), bid, ask.
2) Adapter (smc_adapter_josh.generate_entries_with_josh):
   - ticks -> M1,
   - BOS/CHOCH by smartmoneyconcepts,
   - first retest -> entry,
   - SL behind opposite minute swing; TP behind structural swing (adapter outputs).
3) Choose TP mode:
   - structure: use adapter TP (may yield RR<1).
   - symmetric: set TP at min_rr * risk from entry (guarantee RR>=min_rr).
4) Filter trades by RR (>= --min-rr).
5) Simulate tick-by-tick:
   - base: given stop,
   - plus1: stop widened by +1 pip (same TP as chosen by mode).
6) Save detailed CSV with entry_time, entry_price, RR fields.

Requires: pandas, tqdm, smartmoneyconcepts; local smc_adapter_josh.py
"""
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from smc_adapter_josh import generate_entries_with_josh


@dataclass
class Entry:
    tick_idx: int
    direction: int     # +1 long, -1 short
    stop_pips: float
    tp_pips: float
    stop_price: float
    tp_price: float
    meta: Dict[str, Any]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def simulate_until_exit(
    bids: np.ndarray,
    asks: np.ndarray,
    entry_idx: int,
    direction: int,
    stop_price: float,
    tp_price: float,
) -> bool:
    """
    Long: exit on BID (TP if bid>=tp, SL if bid<=sl)
    Short: exit on ASK (TP if ask<=tp, SL if ask>=sl)
    Returns True if TP is hit before SL, False otherwise.
    """
    if direction not in (+1, -1):
        raise ValueError("direction must be +1 or -1")

    start = entry_idx + 1
    if start >= len(bids):
        return False

    if direction == +1:
        bid_slice = bids[start:]
        tp_hits = np.flatnonzero(bid_slice >= tp_price)
        sl_hits = np.flatnonzero(bid_slice <= stop_price)
    else:
        ask_slice = asks[start:]
        tp_hits = np.flatnonzero(ask_slice <= tp_price)
        sl_hits = np.flatnonzero(ask_slice >= stop_price)

    tp_idx = tp_hits[0] if tp_hits.size else None
    sl_idx = sl_hits[0] if sl_hits.size else None

    if tp_idx is None and sl_idx is None:
        return False  # open at EOF -> count as loss
    if tp_idx is None:
        return False
    if sl_idx is None:
        return True
    return tp_idx <= sl_idx


def load_ticks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df = df.dropna(subset=["time", "bid", "ask"]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="One-pip SL sensitivity simulator")
    ap.add_argument("--csv", type=str, required=True, help="Tick CSV: time,bid,ask")
    ap.add_argument("--pip-size", type=float, required=True,
                    help="Pip size (EURUSD=0.0001, XAUUSD=0.1 etc.)")
    ap.add_argument("--smc-swing-lookback", type=int, default=50,
                    help="swing_length for SMC swings (default=50)")
    ap.add_argument("--retest-tolerance-pips", type=float, default=0.2,
                    help="Retest touch tolerance in pips")
    ap.add_argument("--sl-beyond-swing-pips", type=float, default=0.0,
                    help="Extra SL beyond swing in pips")
    ap.add_argument("--tp-beyond-swing-pips", type=float, default=0.0,
                    help="Extra TP beyond swing in pips (structure mode)")
    ap.add_argument("--allow-choch-fallback", action="store_true",
                    help="Allow CHOCH if no BOS on the bar")
    ap.add_argument("--no-choch-fallback", action="store_true",
                    help="Disable CHOCH; BOS only")
    ap.add_argument("--min-rr", type=float, default=1.0,
                    help="Minimum R/R (reward/risk). 1.0 = drop RR<1:1. 0.0 = disable filter.")
    ap.add_argument("--min-sl-pips", type=float, default=0.0,
                    help="Drop trades where risk (entry->SL) is below this value in pips.")
    ap.add_argument("--min-tp-pips", type=float, default=0.0,
                    help="Drop trades where reward (entry->TP) is below this value in pips.")
    ap.add_argument("--tp-mode", type=str, default="structure",
                    choices=["structure", "symmetric"],
                    help="TP mode: 'structure' (adapter TP) or 'symmetric' (TP at min_rr*risk).")
    ap.add_argument("--tf", type=int, default=1, help="Unused (CLI compatibility)")
    ap.add_argument("--smc-sessions", type=str, default="on",
                    help="SMC session filter (CLI compatibility; currently informational only).")
    ap.add_argument("--smc-fvg", type=float, default=0.0,
                    help="Fair value gap filter (CLI compatibility; currently informational only).")

    args = ap.parse_args()
    if args.allow_choch_fallback and args.no_choch_fallback:
        raise SystemExit("Use either --allow-choch-fallback or --no-choch-fallback, not both.")

    pip = float(args.pip_size)
    min_rr = max(0.0, float(args.min_rr))
    min_sl_pips = max(0.0, float(args.min_sl_pips))
    min_tp_pips = max(0.0, float(args.min_tp_pips))
    tp_mode = args.tp_mode

    ticks = load_ticks(args.csv)

    bid_arr = ticks["bid"].to_numpy(dtype=float)
    ask_arr = ticks["ask"].to_numpy(dtype=float)
    time_arr = ticks["time"].to_numpy()

    # 1) Entries from adapter (structure-based SL/TP)
    entries_raw = generate_entries_with_josh(
        ticks=ticks,
        pip=pip,
        swing_lookback=int(args.smc_swing_lookback),
        use_bos=True,
        allow_choch_fallback=(not args.no_choch_fallback),
        retest_tolerance_pips=float(args.retest_tolerance_pips),
        sl_margin_pips=float(args.sl_beyond_swing_pips),
        tp_margin_pips=float(args.tp_beyond_swing_pips),
        debug_log_path="smc_lib_debug.csv",
    )
    if not entries_raw:
        print("No entries produced by adapter. Check smc_lib_debug.csv for reasons.")
        return

    entries: List[Entry] = [Entry(*e) for e in entries_raw]

    # 2) Build per-trade parameters depending on TP mode + RR filter
    prepared: List[Tuple[Entry, float, float, float, float, float, float]] = []
    # (Entry, entry_price, risk_pips, reward_pips, rr_used, stop_price_used, tp_price_used)

    dropped_rr = 0
    dropped_sl = 0
    dropped_tp = 0
    for e in entries:
        # entry side price
        entry_side_price = float(ask_arr[e.tick_idx] if e.direction == +1 else bid_arr[e.tick_idx])

        # risk (pips) from entry to SL
        if e.direction == +1:
            risk_pips = max(0.0, (entry_side_price - e.stop_price) / pip)
        else:
            risk_pips = max(0.0, (e.stop_price - entry_side_price) / pip)

        if risk_pips + 1e-12 < min_sl_pips:
            dropped_sl += 1
            continue

        # Choose TP
        if tp_mode == "structure":
            tp_price_used = e.tp_price
            reward_pips = max(0.0, (tp_price_used - entry_side_price) / pip) if e.direction == +1 \
                          else max(0.0, (entry_side_price - tp_price_used) / pip)
            if reward_pips + 1e-12 < min_tp_pips:
                dropped_tp += 1
                continue
            rr = (float("inf") if risk_pips == 0 and reward_pips > 0
                  else (0.0 if risk_pips == 0 else reward_pips / risk_pips))
            # RR filter
            if rr + 1e-12 < min_rr:
                dropped_rr += 1
                continue
        else:  # symmetric
            # Set reward to at least min_rr * risk
            target_reward_pips = min_rr * risk_pips
            if target_reward_pips + 1e-12 < min_tp_pips:
                dropped_tp += 1
                continue
            if e.direction == +1:
                tp_price_used = entry_side_price + target_reward_pips * pip
            else:
                tp_price_used = entry_side_price - target_reward_pips * pip
            reward_pips = target_reward_pips
            rr = (float("inf") if risk_pips == 0 and reward_pips > 0 else (0.0 if risk_pips == 0 else reward_pips / risk_pips))
            # symmetric always meets rr >= min_rr by construction

        stop_price_used = e.stop_price
        prepared.append((e, float(entry_side_price), float(risk_pips), float(reward_pips), float(rr),
                         float(stop_price_used), float(tp_price_used)))

    if not prepared:
        print(f"All trades filtered out by RR (min_rr={min_rr}). Try --tp-mode symmetric or lower --min-rr.")
        return

    if dropped_rr > 0 and tp_mode == "structure":
        print(f"Filtered out by RR < {min_rr}: {dropped_rr} trades (kept {len(prepared)}).")
    if dropped_sl > 0:
        print(f"Filtered out by risk < {min_sl_pips} pips: {dropped_sl} trades.")
    if dropped_tp > 0:
        print(f"Filtered out by reward < {min_tp_pips} pips: {dropped_tp} trades.")

    # 3) Simulate outcomes
    base_wins = plus_wins = 0
    b = c = 0
    rows: List[Dict[str, Any]] = []

    for idx, (e, entry_price, risk_pips, reward_pips, rr_used, stop_price_used, tp_price_used) in enumerate(
        tqdm(prepared, desc="Simulating", unit="trade")
    ):
        # entry time string
        entry_ts = pd.Timestamp(time_arr[e.tick_idx])
        entry_time_str = entry_ts.isoformat() if not pd.isna(entry_ts) else ""

        # Base outcome (chosen TP mode)
        base_win = simulate_until_exit(
            bid_arr, ask_arr, e.tick_idx, e.direction, stop_price_used, tp_price_used
        )

        # +1 pip stop (same TP)
        stop_plus = stop_price_used - pip if e.direction == +1 else stop_price_used + pip
        plus_win = simulate_until_exit(
            bid_arr, ask_arr, e.tick_idx, e.direction, stop_plus, tp_price_used
        )

        base_wins += int(base_win)
        plus_wins += int(plus_win)
        if base_win and not plus_win: b += 1
        elif (not base_win) and plus_win: c += 1

        rows.append({
            "trade_id": idx,
            "entry_tick_idx": e.tick_idx,
            "entry_time": entry_time_str,
            "direction": e.direction,
            "entry_price": entry_price,
            "risk_pips": risk_pips,
            "reward_pips": reward_pips,
            "rr_used": rr_used,
            "tp_mode": tp_mode,
            "stop_price_base": stop_price_used,
            "tp_price_used": tp_price_used,
            "stop_price_plus1": stop_plus,
            "base_win": base_win,
            "plus1_win": plus_win,
            "i_bar": e.meta.get("i"),
            "entry_min_idx": e.meta.get("entry_min_idx"),
        })

    # 4) Metrics
    total = len(prepared)
    base_wr = base_wins / total
    plus_wr = plus_wins / total
    delta_wr = plus_wr - base_wr
    pval = mcnemar_exact(b, c)

    print(f"trades_kept: {total}")
    print(f"base_wins: {base_wins}")
    print(f"plus1_wins: {plus_wins}")
    print(f"base_wr: {base_wr:.6f}")
    print(f"plus_wr: {plus_wr:.6f}")
    print(f"delta_wr: {delta_wr:.6f}")
    print(f"mcnemar_b (base win, plus lose): {b}")
    print(f"mcnemar_c (base lose, plus win): {c}")
    print(f"mcnemar_exact_p: {pval:.6f}")

    pd.DataFrame(rows).to_csv("one_pip_results_detailed.csv", index=False)
    print("Detailed results saved to: one_pip_results_detailed.csv")
    print("Adapter debug (reasons) saved to: smc_lib_debug.csv")


if __name__ == "__main__":
    main()
