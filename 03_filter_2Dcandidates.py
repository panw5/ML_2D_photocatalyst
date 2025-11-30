import pandas as pd
from ase.db import connect
import re

# =====================================
# General Helper Functions
# =====================================

# Toxic / radioactive elements list
TOXIC_ELEMENTS = {
    "Hg", "Cd", "Pb", "As", "Tl", "Be", "Se", "Th", "U", "Po", "Ra"
}

def extract_elements_from_uid(uid: str):
    """Extract a list of element symbols from a UID."""
    return re.findall(r"[A-Z][a-z]?", uid)

def contains_toxic_elements(uid: str) -> bool:
    """Return True if UID contains toxic elements."""
    if not isinstance(uid, str):
        return False
    elements = extract_elements_from_uid(uid)
    return any(elem in TOXIC_ELEMENTS for elem in elements)


# =====================================
# UID Element Parsing + Solubility Screening
# =====================================

EL_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

def parse_elements(uid: str):
    """
    Parse element symbols from UID (without counts).
      Example: '2AgBrTe2-1' -> ['Ag', 'Br', 'Te']
    """
    if not isinstance(uid, str):
        return []
    head = uid.split('-', 1)[0]
    head = re.sub(r"^\d+", "", head)
    return [sym for sym, num in EL_TOKEN.findall(head)]

HALOGENS = {"Cl", "Br", "I"}
ALKALIS  = {"Li", "Na", "K", "Rb", "Cs", "NH4"}
HARDLY_SOLUBLE_PAIRS = {
    ("Ag", "Cl"), ("Ag", "Br"), ("Ag", "I"),
    ("Pb", "SO4"), ("Ba", "SO4")
}

def almost_certainly_soluble(uid: str) -> bool:
    """
    Returns True if the material is almost certainly soluble in water
    and should be excluded; otherwise False (uncertain or potentially stable).
    """
    els = set(parse_elements(uid))
    if not els:
        return False

    # Strongly soluble cations + halogens → high solubility
    if (els & ALKALIS) and (els & HALOGENS):
        # Exceptions: known poorly soluble pairs
        for a, b in HARDLY_SOLUBLE_PAIRS:
            if {a, b}.issubset(els):
                return False
        return True

    return False


# =====================================
# Main Filtering Function
# =====================================

def filter_candidates(db_path: str, output_csv: str) -> None:
    db = connect(db_path)
    data = []

    # Track number of materials after each step
    step_stats = []

    def record_step(name, df):
        step_stats.append((name, len(df)))
        print(f"{name}: {len(df)} materials remain")

    # Load all rows
    for row in db.select():
        record = {k: row.get(k, None) for k in row._keys}
        record["nspecies"] = len(set(row.symbols))
        data.append(record)

    df = pd.DataFrame(data)

    num_cols = ["ehull", "hform", "gap_hse", "vbm_hse", "cbm_hse",
                "emass_cbm", "emass_vbm"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Step 1: HSE band gap range
    df = df[df["gap_hse"].notna()]
    df = df[(df["gap_hse"] >= 1.63) & (df["gap_hse"] <= 3)]
    record_step("Step 1 - HSE band gap filter", df)

    # Step 2: Band edge positions (VBM, CBM)
    df = df[df["vbm_hse"].notna() & df["cbm_hse"].notna()]
    df = df[(df["vbm_hse"] < -5.87) & (df["cbm_hse"] > -4.24)]
    record_step("Step 2 - Band edge alignment filter", df)

    # Step 3: Quasi-direct band gap: gap_dir - gap ≤ 0.2 eV
    if {"gap_hse", "gap_dir_hse"}.issubset(df.columns):
        _mask_ok = df[["gap_hse", "gap_dir_hse"]].notna().all(axis=1)
        _mask_pos = (df["gap_hse"] >= 0) & (df["gap_dir_hse"] >= 0)
        _mask_quasi_direct = (df["gap_dir_hse"] - df["gap_hse"]) <= 0.2
        df = df[_mask_ok & _mask_pos & _mask_quasi_direct]
        record_step("Step 3 - Quasi-direct band gap filter", df)

    # Step 4: Remove materials with toxic elements
    df = df[~df["uid"].apply(contains_toxic_elements)]
    record_step("Step 4 - Toxic-element exclusion", df)

    # Step 5: Number of atomic species ≤ 3
    df = df[df["nspecies"] <= 3]
    record_step("Step 5 - Number of species ≤ 3", df)

    # Step 6: Thermodynamic stability (ehull < 0.2 eV)
    df = df[df["ehull"] < 0.2]
    record_step("Step 6 - Thermodynamic stability filter", df)

    # Step 7: Formation energy < 0 eV
    df = df[df["hform"] < 0]
    record_step("Step 7 - Formation energy < 0", df)

    # Step 8: Effective mass filter
    mcols = ["emass_cbm", "emass_vbm"]
    mmin = df[mcols].min(axis=1).clip(lower=1e-3)
    mmax = df[mcols].max(axis=1)
    ratio = mmax / mmin
    df = df[
        df[mcols].notna().all(axis=1) &
        (df["emass_cbm"] < 3) &
        (df["emass_vbm"] < 3) &
        (ratio > 1.5)
    ]
    record_step("Step 8 - Effective mass filter", df)

    # Step 9: Non-magnetic filter
    if "is_magnetic" in df.columns:
        def _to_bool_mag(x):
            if isinstance(x, bool):
                return x
            if isinstance(x, (int, float)):
                return bool(int(x))
            if isinstance(x, str):
                s = x.strip().lower()
                if s in {"true", "yes", "1", "magnetic"}:
                    return True
                if s in {"false", "no", "0", "non-magnetic", "nonmagnetic"}:
                    return False
            return None

        _mag = df["is_magnetic"].apply(_to_bool_mag)
        df = df[_mag == False]
        record_step("Step 9 - Non-magnetic filter", df)

    # Step 10: Water solubility filter (rule-based)
    if "uid" in df.columns:
        before = len(df)
        df = df[~df["uid"].apply(almost_certainly_soluble)]
        record_step("Step 10 - Water solubility filter", df)
        after = len(df)
        print(f"Removed {before - after} materials that are almost certainly soluble in water.")

    # Step 11: Remove specific known water-soluble materials
    easily_dissolved_uids = {
        "2AlI3-1",
        "2AlIS-2",
        "2P2S3-4",
        "4LiPS2-1",
        "2ZnP2S4-1",
        "2SbI3-1",
        "4NaSbS2-1",
        "4NaSbS2-2",
        "4SiP2-1",
    }

    df = df[~df["uid"].isin(easily_dissolved_uids)]
    record_step("Step 11 - Known water-soluble materials removed", df)

    print(f"After removing specific soluble materials, remaining count: {len(df)}")

    # Summary of all filtering steps
    print("\n============================")
    print("Summary of materials after each filtering step")
    print("============================")
    for name, count in step_stats:
        print(f"{name:<40} {count:>6}")
    print("============================\n")

    # Save filtered results
    df.to_csv(output_csv, index=False)
    print(f"Filtering complete. Final count: {len(df)} materials. Results saved to {output_csv}")


if __name__ == "__main__":
    filter_candidates("data/c2db-2.db", "output/filtered_by_all.csv")
