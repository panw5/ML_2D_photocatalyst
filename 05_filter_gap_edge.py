#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate from the c2db database:
  Table 2: Filtered only by HSE band gap range (1.63 < Eg < 3.0 eV)
  Table 3: Based on Table 2, further filtered by band edge positions
           (VBM < -5.87 eV, CBM > -4.24 eV)
"""

import pandas as pd
from ase.db import connect


def filter_gap_and_bandedges(db_path: str, out_gap_csv: str, out_band_csv: str) -> None:
    db = connect(db_path)
    data = []

    # Read all rows from the database
    for row in db.select():
        record = {k: row.get(k, None) for k in row._keys}
        data.append(record)
    df = pd.DataFrame(data)

    # Ensure key columns are numeric
    cols = ["gap_hse", "vbm_hse", "cbm_hse"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Table 2: Filter only by band gap
    df_gap = df[df["gap_hse"].notna()]
    df_gap = df_gap[(df_gap["gap_hse"] > 1.63) & (df_gap["gap_hse"] < 3.0)]
    df_gap.to_csv(out_gap_csv, index=False)
    print(f"Table 2 saved ({len(df_gap)} rows) -> {out_gap_csv}")

    # Table 3: Filter by band gap + band edge positions
    df_band = df_gap[
        df_gap["vbm_hse"].notna() &
        df_gap["cbm_hse"].notna() &
        (df_gap["vbm_hse"] < -5.87) &
        (df_gap["cbm_hse"] > -4.24)
    ]
    df_band.to_csv(out_band_csv, index=False)
    print(f"Table 3 saved ({len(df_band)} rows) -> {out_band_csv}")


if __name__ == "__main__":
    filter_gap_and_bandedges(
        "data/c2db-2.db",
        "output/filtered_by_gap.csv",         # Table 2
        "output/filtered_by_bandedge.csv"     # Table 3
    )
