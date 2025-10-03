# smc_adapter_josh.py
# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Any, Optional
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
    m1 = (
        pd.DataFrame({"time": df["time"], "mid": mid})
        .set_index("time")
        .resample("1min")
        .agg(["first","max","min","last"])
        .dropna()
    )
    m1.columns = ["open","high","low","close"]
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
            w.writerow(["reason","i","entry_min_idx","dir","entry_time","hour",
                        "level","tol_pips","sl_pips","tp_pips"])
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
    debug_log_path: Optional[str] = "smc_lib_debug.csv",
) -> List[Entry]:
    """
    Чистая связка под API smartmoneyconcepts (README):
      - swing_highs_lows(ohlc, swing_length)
      - bos_choch(ohlc, swing_highs_lows, close_break=True)
    Сигналы читаем из колонок BOS/CHOCH, уровень ретеста — из Level.
    SL/TP — за ближайшими минутными свингами (HighLow = ±1).
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

    # 4) подготовка утилит
    base_times = df_ticks["time"].to_numpy()
    tol = float(retest_tolerance_pips) * float(pip)

    def nearest_tick_idx(ts) -> int:
        pos = np.searchsorted(base_times, pd.to_datetime(ts, utc=True))
        return int(min(max(pos, 0), len(base_times) - 1))

    entries: List[Entry] = []
    last_H = last_L = None  # индексы последних свингов

    for i in range(len(m1)):
        if i in swing_highs:
            last_H = i
        if i in swing_lows:
            last_L = i
        if last_H is None or last_L is None:
            # пока не знаем оба свинга — не можем ставить SL/TP «за свингом»
            continue

        # Направление: сначала BOS, потом (по флагу) CHOCH
        dir_ = 0
        lvl_i = None

        if use_bos and col_bos is not None:
            v = st.loc[i, col_bos]
            if pd.notna(v) and v != 0:
                dir_ = +1 if v == 1 else -1
                lvl_i = st.loc[i, col_lvl]

        if dir_ == 0 and (allow_choch_fallback or use_choch) and col_choch is not None:
            v = st.loc[i, col_choch]
            if pd.notna(v) and v != 0:
                dir_ = +1 if v == 1 else -1
                lvl_i = st.loc[i, col_lvl]

        if dir_ == 0 or pd.isna(lvl_i):
            _dbg_row(dbg, reason="skip_no_signal", i=i, entry_min_idx=None, dir=0,
                     entry_time=m1.loc[i, "time"], hour=pd.to_datetime(m1.loc[i, "time"]).hour,
                     level=None, tol_pips=retest_tolerance_pips, sl_pips=None, tp_pips=None)
            continue

        break_level = float(lvl_i)

        # 5) ищем ПЕРВЫЙ ретест уровня в ближайшие 2000 минут после сигнала
        entry_min_idx = None
        for j in range(i + 1, min(i + 2000, len(m1))):
            if dir_ == +1:
                # лонг: ретест — low <= level + tol
                if float(m1.loc[j, "low"]) <= break_level + tol:
                    entry_min_idx = j
                    break
            else:
                # шорт: ретест — high >= level - tol
                if float(m1.loc[j, "high"]) >= break_level - tol:
                    entry_min_idx = j
                    break

        if entry_min_idx is None:
            _dbg_row(dbg, reason="skip_no_retest", i=i, entry_min_idx=None, dir=dir_,
                     entry_time=m1.loc[i, "time"], hour=pd.to_datetime(m1.loc[i, "time"]).hour,
                     level=break_level, tol_pips=retest_tolerance_pips, sl_pips=None, tp_pips=None)
            continue

        # 6) Формируем сделку: вход — первый тик на минуте ретеста
        entry_time = m1.loc[entry_min_idx, "time"]
        eidx = nearest_tick_idx(entry_time)

        if dir_ == +1:
            # лонг: SL за последним swing low; TP за последним swing high
            protect = float(m1.loc[last_L, "low"])
            target  = float(m1.loc[last_H, "high"])
            entry_price = float(df_ticks.iloc[eidx]["ask"])
            stop_price  = protect - sl_margin_pips * pip
            tp_price    = target  + tp_margin_pips * pip
            stop_pips   = max(0.0, (entry_price - stop_price) / pip)
            tp_pips     = max(0.0, (tp_price - entry_price) / pip)
        else:
            # шорт: SL за последним swing high; TP за последним swing low
            protect = float(m1.loc[last_H, "high"])
            target  = float(m1.loc[last_L, "low"])
            entry_price = float(df_ticks.iloc[eidx]["bid"])
            stop_price  = protect + sl_margin_pips * pip
            tp_price    = target  - tp_margin_pips * pip
            stop_pips   = max(0.0, (stop_price - entry_price) / pip)
            tp_pips     = max(0.0, (entry_price - tp_price) / pip)

        _dbg_row(dbg, reason="accepted", i=i, entry_min_idx=entry_min_idx, dir=dir_,
                 entry_time=entry_time, hour=pd.to_datetime(entry_time).hour,
                 level=break_level, tol_pips=retest_tolerance_pips,
                 sl_pips=round(stop_pips, 6), tp_pips=round(tp_pips, 6))

        entries.append((
            eidx, int(dir_), float(stop_pips), float(tp_pips),
            float(stop_price), float(tp_price),
            {"i": i, "entry_min_idx": entry_min_idx, "source": "smc.bos_choch"}
        ))

    return entries
