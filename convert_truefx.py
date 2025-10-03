#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys
from tqdm import tqdm

if len(sys.argv) < 2:
    print("Usage: python convert_truefx.py EURUSD-2025-09.csv")
    sys.exit(1)

infile = sys.argv[1]
outfile = infile.replace(".csv", "_converted.csv")

print(f"Converting {infile} -> {outfile}")

# читаем truefx: EUR/USD,20250701 00:00:00.953,1.17862,1.17866
# используем tqdm для прогресса по чанкам и сразу пишем результат,
# чтобы не держать все данные в памяти
chunksize = 300000
reader = pd.read_csv(
    infile,
    header=None,
    names=["pair", "time_raw", "bid", "ask"],
    chunksize=chunksize,
)

first_chunk = True
for chunk in tqdm(reader, desc="Processing"):
    chunk["time"] = pd.to_datetime(
        chunk["time_raw"],
        format="%Y%m%d %H:%M:%S.%f",
        utc=True,
        errors="coerce",
    )
    converted = chunk[["time", "bid", "ask"]].dropna(subset=["time"])
    converted.to_csv(
        outfile,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
    )
    first_chunk = False

print(f"Done! Saved to {outfile}")
