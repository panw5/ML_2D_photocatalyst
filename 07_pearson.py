import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ===== File paths =====
total_path   = "output1/c2db_export.csv"
filtered_path = "output/filtered_by_all.csv"

# ===== Load data =====
df_total = pd.read_csv(total_path)
df_after = pd.read_csv(filtered_path)

# ===== Compute averaged quantities (vacuum-independent) =====
def add_averages(df):
    df = df.copy()
    if {'alphax_el', 'alphay_el', 'alphaz_el'}.issubset(df.columns):
        df['alpha_avg'] = df[['alphax_el', 'alphay_el', 'alphaz_el']].mean(axis=1)
    if {'plasmafrequency_x', 'plasmafrequency_y'}.issubset(df.columns):
        df['plasma_avg'] = df[['plasmafrequency_x', 'plasmafrequency_y']].mean(axis=1)
    return df

df_total = add_averages(df_total)
df_after = add_averages(df_after)

# ===== Select 10 core features (vbm/cbm already referenced to vacuum) =====
cand = [
    'gap_hse', 'vbm_hse', 'cbm_hse',          # electronic structure / alignment
    'ehull', 'hform',                         # stability
    'emass_cbm', 'emass_vbm',                 # carrier properties
    'alpha_avg', 'dyn_stab',                  # dielectric / optical
    'thickness'                               # structural scale
]
features = [c for c in cand if c in df_total.columns and c in df_after.columns]

# ===== Normalize dyn_stab to numeric =====
def normalize_dyn_stab(df):
    if 'dyn_stab' in df.columns:
        df = df.copy()
        mapping = {'yes': 1.0, 'no': 0.0, 'unknown': np.nan}
        df['dyn_stab'] = df['dyn_stab'].map(mapping)
    return df

df_total = normalize_dyn_stab(df_total)
df_after = normalize_dyn_stab(df_after)

# ===== Improved version: automatic numeric conversion and tolerating missing values =====
def to_numeric_df(df, cols):
    df_num = df[cols].apply(pd.to_numeric, errors='coerce')
    # Drop a column if all values are NaN
    df_num = df_num.dropna(axis=1, how='all')
    return df_num

A = to_numeric_df(df_total, features)
B = to_numeric_df(df_after, features)

# Pearson correlation with pairwise deletion of missing values
corr_A = A.corr(method='pearson', min_periods=10)
corr_B = B.corr(method='pearson', min_periods=10)

print(f"N_before = {A.notna().any(axis=1).sum()}, N_after = {B.notna().any(axis=1).sum()}")


# ===== Plot heatmaps (full matrix + numbers) =====
sns.set_theme(style="white", font_scale=0.9)
fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

sns.heatmap(
    corr_A, ax=axes[0], cmap='RdBu_r', vmin=-1, vmax=1, center=0,
    square=True, annot=True, fmt=".2f", linewidths=0.4, cbar=False
)
axes[0].set_title(f"HSE core — Before screening (N={len(A)})")

sns.heatmap(
    corr_B, ax=axes[1], cmap='RdBu_r', vmin=-1, vmax=1, center=0,
    square=True, annot=True, fmt=".2f", linewidths=0.4,
    cbar_kws={'label': 'Pearson r', 'shrink': 0.82}
)
axes[1].set_title(f"HSE core — After screening (N={len(B)})")


# ===== Save figure =====
out_png = "output/figures/corr_heatmaps.png"

plt.savefig(out_png, dpi=600, bbox_inches="tight")

print(f"[OK] Saved: {out_png}")
plt.show()
