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
# используем tqdm для прогресса по чанкам
chunksize = 300000
dfs = []
for chunk in tqdm(pd.read_csv(infile, header=None, names=["pair","time_raw","bid","ask"], chunksize=chunksize), desc="Processing"):
    chunk["time"] = pd.to_datetime(chunk["time_raw"], format="%Y%m%d %H:%M:%S.%f", utc=True)
    dfs.append(chunk[["time","bid","ask"]])

df_out = pd.concat(dfs, ignore_index=True)
df_out.to_csv(outfile, index=False)

print(f"Done! Saved to {outfile}")
