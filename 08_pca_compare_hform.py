import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ===== Paths =====
total_path   = "output1/c2db_export.csv"
filtered_path = "output/filtered_by_all.csv"

# ===== Load =====
df0 = pd.read_csv(total_path)
df1 = pd.read_csv(filtered_path)

# ===== Derived columns =====
def enrich(df):
    """Add averaged dielectric constant if available."""
    df = df.copy()
    if {'alphax_el', 'alphay_el', 'alphaz_el'}.issubset(df.columns):
        df['alpha_avg'] = df[['alphax_el', 'alphay_el', 'alphaz_el']].mean(axis=1)
    return df

df0 = enrich(df0)
df1 = enrich(df1)

# ===== Define feature set =====
base_features = [
    'gap', 'vbm', 'cbm', 'ehull', 'hform',
    'emass_cbm', 'emass_vbm', 'alpha_avg', 'thickness'
]
feats_common = [c for c in base_features if c in df0.columns and c in df1.columns]

def to_numeric(df, cols):
    """Convert to numeric and replace inf with NaN."""
    X = df[cols].apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    return X

# ===== Unified PCA workflow =====
def run_pca(df0, df1, feats):
    X0_raw = to_numeric(df0, feats)
    X1_raw = to_numeric(df1, feats)
    X_all = pd.concat([X0_raw, X1_raw], axis=0, ignore_index=True)

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imp.fit_transform(X_all)
    Z_all = scaler.fit_transform(X_imp)

    pca = PCA(n_components=2, random_state=42)
    P_all = pca.fit_transform(Z_all)
    var_ratio = pca.explained_variance_ratio_ * 100

    n0 = len(X0_raw)
    P0, P1 = P_all[:n0], P_all[n0:]
    return P0, P1, var_ratio, feats

# ===== PCA with two feature sets =====
P0_full, P1_full, var_full, feats_full = run_pca(df0, df1, feats_common)
P0_nh,  P1_nh,  var_nh,  feats_nh  = run_pca(df0, df1, [f for f in feats_common if f != 'hform'])

# ===== Visualization =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (P0, P1, var, title) in zip(
    axes,
    [
        (P0_full, P1_full, var_full, "With hform"),
        (P0_nh,  P1_nh,  var_nh,  "Without hform")
    ]
):
    ax.scatter(P0[:, 0], P0[:, 1], s=8,  alpha=0.25, label=f"Before (N={len(P0)})")
    ax.scatter(P1[:, 0], P1[:, 1], s=40, alpha=0.9,  label=f"After (N={len(P1)})")

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)")
    ax.set_title(f"PCA of feature space — {title}")
    ax.legend()

plt.tight_layout()

# ===== Save figure =====
out_png = "output/figures/pca_compare.png"
plt.savefig(out_png, dpi=600, bbox_inches="tight")

print(f"[OK] Saved: {out_png}")
plt.show()
