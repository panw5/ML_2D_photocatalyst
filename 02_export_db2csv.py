# save as export_db_to_csv.py
import json
import numbers
from pathlib import Path

import pandas as pd
from ase.db import connect


def _to_scalar_or_json(x):
    """Convert complex objects to strings that can be written to CSV; keep scalars as-is."""
    if x is None or isinstance(x, (str, numbers.Number, bool)):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def flatten_row(row):
    """
    Flatten a single ASE db row into a dict with:
    - top-level fields (id, formula, etc.)
    - key_value_pairs
    - data (if present)
    """
    rec = {}

    # Top-level accessible fields
    rec["id"] = row.id
    # Some databases may not have `formula`, so add a guard
    try:
        rec["formula"] = row.formula
    except Exception:
        rec["formula"] = None

    # Merge key_value_pairs
    kv = getattr(row, "key_value_pairs", {}) or {}
    for k, v in kv.items():
        rec[k] = v

    # Merge data (if present)
    data = getattr(row, "data", {}) or {}
    if isinstance(data, dict):
        for k, v in data.items():
            # Avoid overwriting existing keys; if conflict, use data_ prefix
            if k in rec and rec[k] != v:
                rec[f"data_{k}"] = v
            else:
                rec[k] = v

    # Normalize values into types that can be written to CSV
    rec = {k: _to_scalar_or_json(v) for k, v in rec.items()}
    return rec


def export_db_to_csv(db_path: str, out_csv: str):
    """Export all records in an ASE .db file to a CSV file."""
    db_path = Path(db_path)
    out_csv = Path(out_csv)
    rows = []

    with connect(str(db_path)) as db:
        n = len(db)
        print(f"Reading database: {db_path}  (total {n} records)")
        for i, row in enumerate(db.select()):
            rows.append(flatten_row(row))
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{n}")

    df = pd.DataFrame(rows)

    # Move common key columns to the front; keep the remaining columns in original order
    preferred = [c for c in ["id", "formula", "uid", "olduid", "label"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in preferred]
    df = df[preferred + other_cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote CSV to: {out_csv}  (total {len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    # Example: modify the paths as needed
    export_db_to_csv("../data/c2db-2.db", "../output/c2db_export.csv")
