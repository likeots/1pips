# smc_adapter_josh.py
# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Any, Optional
from bisect import bisect_right
import pandas as pd
import numpy as np
from pathlib import Path
import csv
from smartmoneyconcepts import smc  # официальный импорт из README

Entry = Tuple[int, int, float, float, float, float, Dict[str, Any]]
# (entry_tick_idx, direction(+1/-1), stop_pips, tp_pips, stop_price, tp_price, meta)

# -------------------- utils --------------------

def _robust_time(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")
    bad = t.isna()
    if bad.any():
        t.loc[bad] = pd.to_datetime(s[bad], utc=True, errors="coerce")
    return t

def _ticks_to_m1(ticks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ожидаются колонки ticks: time, bid, ask
    Строим M1 OHLC по mid-цене (среднее bid/ask).
    """
    df = ticks.copy()
    df["time"] = _robust_time(df["time"])
    df["bid"]  = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"]  = pd.to_numeric(df["ask"], errors="coerce")
    df = df.dropna(subset=["time","bid","ask"]).sort_values("time").reset_index(drop=True)

    mid = (df["bid"] + df["ask"]) / 2.0
    grouped = (
        pd.DataFrame({"time": df["time"], "mid": mid})
        .set_index("time")
        .resample("1min")
    )

    m1 = grouped.agg(["first", "max", "min", "last"]).dropna()
    m1.columns = ["open", "high", "low", "close"]

    volume = (
        pd.DataFrame({"time": df["time"], "vol": 1.0})
        .set_index("time")
        .resample("1min")
        .sum()
    )

    m1 = m1.join(volume, how="left").rename(columns={"vol": "volume"}).fillna({"volume": 0.0})
    m1 = m1.reset_index(drop=False)
    return df, m1

def _dbg_init(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    write_header = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    if write_header:
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "reason",
                "i",
                "entry_min_idx",
                "dir",
                "entry_time",
                "hour",
                "level",
                "tol_pips",
                "sl_pips",
                "tp_pips",
                "ob_idx",
                "fvg_idx",
                "ob_top",
                "ob_bottom",
                "target_level",
            ])
    return p

def _dbg_row(p: Optional[Path], **kw) -> None:
    if not p:
        return
    with p.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            kw.get("reason"),
            kw.get("i"),
            kw.get("entry_min_idx"),
            kw.get("dir"),
            str(kw.get("entry_time")),
            kw.get("hour"),
            kw.get("level"),
            kw.get("tol_pips"),
            kw.get("sl_pips"),
            kw.get("tp_pips"),
            kw.get("ob_idx"),
            kw.get("fvg_idx"),
            kw.get("ob_top"),
            kw.get("ob_bottom"),
            kw.get("target_level"),
        ])

def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    m = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in m:
            return m[n.lower()]
    return None

# -------------------- core --------------------

def generate_entries_with_josh(
    ticks: pd.DataFrame,
    pip: float,
    swing_lookback: int = 50,          # README: swing_length default = 50
    use_bos: bool = True,              # открываем по BOS
    use_choch: bool = False,           # совместимость с твоим основным скриптом
    allow_choch_fallback: bool = True, # разрешить CHOCH если BOS нет на баре
    retest_tolerance_pips: float = 0.2,
    sl_margin_pips: float = 0.0,
    tp_margin_pips: float = 0.0,
    require_fvg_confirmation: bool = True,
    max_retest_minutes: int = 360,
    debug_log_path: Optional[str] = "smc_lib_debug.csv",
) -> List[Entry]:
    """
    Чистая связка под API smartmoneyconcepts (README):
      - swing_highs_lows(ohlc, swing_length)
      - bos_choch(ohlc, swing_highs_lows, close_break=True)
      - ob(...) + fvg(...) для подтверждения ликвидности
    Сигналы читаем из колонок BOS/CHOCH, ретесты ищем в ордер-блоке.
    SL/TP ставим за свингами/структурой (HighLow = ±1).
    """
    dbg = _dbg_init(debug_log_path)

    # 1) тики -> M1
    df_ticks, m1 = _ticks_to_m1(ticks)

    # 2) свинги библиотеки (HighLow: 1=свинг-хай, -1=свинг-лоу; Level=цена свинга)
    sw = smc.swing_highs_lows(m1, swing_length=int(swing_lookback))
    col_hl = _col(sw, "HighLow")
    if col_hl is None:
        # если вдруг иное имя колонки — fallback: пусть всё пусто (сделок не будет)
        return []
    swing_highs = set(sw.index[sw[col_hl] == 1])
    swing_lows  = set(sw.index[sw[col_hl] == -1])

    # 3) BOS/CHOCH (обязателен второй аргумент sw)
    st = smc.bos_choch(m1, sw, close_break=True)
    col_bos   = _col(st, "BOS")
    col_choch = _col(st, "CHOCH")
    col_lvl   = _col(st, "Level")
    if col_lvl is None:
        return []

    # 4) ордер-блоки и FVG для подтверждения
    price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in m1.columns]
    if not price_cols:
        return []
    price_df = m1[price_cols].copy()
    if "volume" not in price_df:
        price_df["volume"] = 1.0

    obs = smc.ob(price_df, sw, close_mitigation=False)
    fvgs = smc.fvg(price_df)

    col_ob = _col(obs, "OB")
    col_ob_top = _col(obs, "Top")
    col_ob_bottom = _col(obs, "Bottom")
    col_ob_mitigated = _col(obs, "MitigatedIndex")

    if col_ob is None or col_ob_top is None or col_ob_bottom is None:
        return []

    col_fvg = _col(fvgs, "FVG")
    col_fvg_top = _col(fvgs, "Top")
    col_fvg_bottom = _col(fvgs, "Bottom")
    col_fvg_mitigated = _col(fvgs, "MitigatedIndex")

    # 5) подготовка утилит
    # Работать с tz-aware Timestamp напрямую в numpy непросто: searchsorted
    # пытается сравнивать tz-aware и tz-naive представления и падает.
    # Поэтому приводим времена к числу наносекунд от эпохи (UTC).
    base_times = df_ticks["time"].apply(lambda x: x.value).to_numpy()
    tol = float(retest_tolerance_pips) * float(pip)
    max_retest_bars = max(1, int(max_retest_minutes))

    def tick_range_for_minute(minute_idx: int) -> Tuple[int, int]:
        start_ts = pd.to_datetime(m1.loc[minute_idx, "time"], utc=True)
        if minute_idx + 1 < len(m1):
            end_ts = pd.to_datetime(m1.loc[minute_idx + 1, "time"], utc=True)
        else:
            end_ts = start_ts + pd.Timedelta(minutes=1)
        start_val = start_ts.value
        end_val = end_ts.value
        start = int(np.searchsorted(base_times, start_val))
        end = int(np.searchsorted(base_times, end_val, side="left"))
        if start >= len(base_times):
            start = len(base_times) - 1
        start = max(0, start)
        end = max(start + 1, min(end if end > start else start + 1, len(base_times)))
        return start, end

    swing_high_list = sorted(swing_highs)
    swing_low_list = sorted(swing_lows)

    def next_swing(idx_list: List[int], current: int) -> Optional[int]:
        pos = bisect_right(idx_list, current)
        if pos < len(idx_list):
            return idx_list[pos]
        return None

    entries: List[Entry] = []
    last_H = last_L = None  # индексы последних свингов

    for i in range(len(m1)):
        if i in swing_highs:
            last_H = i
        if i in swing_lows:
            last_L = i
        if last_H is None or last_L is None:
            continue

        dir_ = 0

        if use_bos and col_bos is not None:
            v = st.loc[i, col_bos]
            if pd.notna(v) and v != 0:
                dir_ = +1 if v == 1 else -1

        if dir_ == 0 and (allow_choch_fallback or use_choch) and col_choch is not None:
            v = st.loc[i, col_choch]
            if pd.notna(v) and v != 0:
                dir_ = +1 if v == 1 else -1

        if dir_ == 0:
            _dbg_row(
                dbg,
                reason="skip_no_signal",
                i=i,
                entry_min_idx=None,
                dir=0,
                entry_time=m1.loc[i, "time"],
                hour=pd.to_datetime(m1.loc[i, "time"]).hour,
                level=None,
                tol_pips=retest_tolerance_pips,
                sl_pips=None,
                tp_pips=None,
            )
            continue

        ob_idx = None
        ob_top = ob_bottom = None

        if col_ob is not None:
            for k in range(i, -1, -1):
                ob_val = obs.loc[k, col_ob]
                if pd.isna(ob_val) or int(ob_val) != dir_:
                    continue
                top_val = obs.loc[k, col_ob_top] if col_ob_top is not None else np.nan
                bottom_val = obs.loc[k, col_ob_bottom] if col_ob_bottom is not None else np.nan
                if pd.isna(top_val) or pd.isna(bottom_val):
                    continue
                mitigated_val = obs.loc[k, col_ob_mitigated] if col_ob_mitigated is not None else np.nan
                if pd.isna(mitigated_val) or mitigated_val == 0 or mitigated_val > i:
                    ob_idx = k
                    ob_top = float(top_val)
                    ob_bottom = float(bottom_val)
                    break

        if ob_idx is None:
            _dbg_row(
                dbg,
                reason="skip_no_order_block",
                i=i,
                entry_min_idx=None,
                dir=dir_,
                entry_time=m1.loc[i, "time"],
                hour=pd.to_datetime(m1.loc[i, "time"]).hour,
                level=None,
                tol_pips=retest_tolerance_pips,
                sl_pips=None,
                tp_pips=None,
            )
            continue

        zone_low = min(ob_top, ob_bottom)
        zone_high = max(ob_top, ob_bottom)

        fvg_idx = None
        if col_fvg is not None:
            start_search = max(0, i - 5)
            for k in range(i, start_search - 1, -1):
                fvg_val = fvgs.loc[k, col_fvg]
                if pd.isna(fvg_val) or int(fvg_val) != dir_:
                    continue
                mitigated_val = fvgs.loc[k, col_fvg_mitigated] if col_fvg_mitigated is not None else np.nan
                if pd.isna(mitigated_val) or mitigated_val == 0 or mitigated_val > i:
                    fvg_idx = k
                    break
        if require_fvg_confirmation and fvg_idx is None:
            _dbg_row(
                dbg,
                reason="skip_no_fvg",
                i=i,
                entry_min_idx=None,
                dir=dir_,
                entry_time=m1.loc[i, "time"],
                hour=pd.to_datetime(m1.loc[i, "time"]).hour,
                level=None,
                tol_pips=retest_tolerance_pips,
                sl_pips=None,
                tp_pips=None,
                ob_idx=ob_idx,
                ob_top=ob_top,
                ob_bottom=ob_bottom,
            )
            continue

        entry_min_idx = None
        for j in range(i + 1, min(i + 1 + max_retest_bars, len(m1))):
            high_j = float(m1.loc[j, "high"])
            low_j = float(m1.loc[j, "low"])
            intersects = (high_j >= zone_low - tol) and (low_j <= zone_high + tol)
            if intersects:
                entry_min_idx = j
                break

        if entry_min_idx is None:
            _dbg_row(
                dbg,
                reason="skip_no_retest",
                i=i,
                entry_min_idx=None,
                dir=dir_,
                entry_time=m1.loc[i, "time"],
                hour=pd.to_datetime(m1.loc[i, "time"]).hour,
                level=None,
                tol_pips=retest_tolerance_pips,
                sl_pips=None,
                tp_pips=None,
                ob_idx=ob_idx,
                fvg_idx=fvg_idx,
                ob_top=ob_top,
                ob_bottom=ob_bottom,
            )
            continue

        tick_start, tick_end = tick_range_for_minute(entry_min_idx)
        chosen_tick = tick_start
        for idx in range(tick_start, tick_end):
            bid_price = df_ticks.iloc[idx]["bid"]
            ask_price = df_ticks.iloc[idx]["ask"]
            if dir_ == +1 and ask_price <= zone_high + tol:
                chosen_tick = idx
                break
            if dir_ == -1 and bid_price >= zone_low - tol:
                chosen_tick = idx
                break

        entry_time = m1.loc[entry_min_idx, "time"]
        entry_price = float(df_ticks.iloc[chosen_tick]["ask"] if dir_ == +1 else df_ticks.iloc[chosen_tick]["bid"])

        if dir_ == +1:
            swing_stop = float(m1.loc[last_L, "low"])
            stop_level = min(swing_stop, zone_low)
            stop_price = stop_level - sl_margin_pips * pip
            next_target_idx = next_swing(swing_high_list, i)
            if next_target_idx is None:
                next_target_idx = last_H
            target_level = float(m1.loc[next_target_idx, "high"])
            tp_base = max(target_level, entry_price)
            tp_price = max(entry_price, tp_base + tp_margin_pips * pip)
            stop_pips = max(0.0, (entry_price - stop_price) / pip)
            tp_pips = max(0.0, (tp_price - entry_price) / pip)
        else:
            swing_stop = float(m1.loc[last_H, "high"])
            stop_level = max(swing_stop, zone_high)
            stop_price = stop_level + sl_margin_pips * pip
            next_target_idx = next_swing(swing_low_list, i)
            if next_target_idx is None:
                next_target_idx = last_L
            target_level = float(m1.loc[next_target_idx, "low"])
            tp_base = min(target_level, entry_price)
            tp_price = min(entry_price, tp_base - tp_margin_pips * pip)
            stop_pips = max(0.0, (stop_price - entry_price) / pip)
            tp_pips = max(0.0, (entry_price - tp_price) / pip)

        _dbg_row(
            dbg,
            reason="accepted",
            i=i,
            entry_min_idx=entry_min_idx,
            dir=dir_,
            entry_time=entry_time,
            hour=pd.to_datetime(entry_time).hour,
            level=None,
            tol_pips=retest_tolerance_pips,
            sl_pips=round(stop_pips, 6),
            tp_pips=round(tp_pips, 6),
            ob_idx=ob_idx,
            fvg_idx=fvg_idx,
            ob_top=ob_top,
            ob_bottom=ob_bottom,
            target_level=target_level,
        )

        entries.append(
            (
                int(chosen_tick),
                int(dir_),
                float(stop_pips),
                float(tp_pips),
                float(stop_price),
                float(tp_price),
                {
                    "i": i,
                    "entry_min_idx": entry_min_idx,
                    "source": "smc.bos_choch",
                    "ob_idx": ob_idx,
                    "fvg_idx": fvg_idx,
                    "target_level": target_level,
                },
            )
        )

    return entries
