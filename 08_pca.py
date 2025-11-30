import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2

# ===== Paths =====
total_path   = "output1/c2db_export.csv"
filtered_path = "output/filtered_by_all.csv"

# ===== Load data =====
df0 = pd.read_csv(total_path)
df1 = pd.read_csv(filtered_path)

def enrich(df):
    """Add averaged dielectric constant if available."""
    if {'alphax_el', 'alphay_el', 'alphaz_el'}.issubset(df.columns):
        df = df.copy()
        df['alpha_avg'] = df[['alphax_el', 'alphay_el', 'alphaz_el']].mean(axis=1)
    return df

df0 = enrich(df0)
df1 = enrich(df1)

# Core features
cand = [
    'gap_hse', 'vbm_hse', 'cbm_hse',
    'ehull', 'emass_cbm', 'emass_vbm',
    'alpha_avg', 'thickness'
]
feats = [c for c in cand if c in df0.columns and c in df1.columns]

def to_numeric(df, cols):
    """Convert to numeric and remove inf values."""
    X = df[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return X

X0 = to_numeric(df0, feats)
X1 = to_numeric(df1, feats)

# ===== Combine, impute, scale, PCA =====
X_all = pd.concat([X0, X1], axis=0, ignore_index=True)
X_all = SimpleImputer(strategy="median").fit_transform(X_all)
X_all = StandardScaler().fit_transform(X_all)

pca = PCA(n_components=2, random_state=42)
P_all = pca.fit_transform(X_all)

n0 = len(X0)
P0, P1 = P_all[:n0], P_all[n0:]
var_ratio = pca.explained_variance_ratio_ * 100


# ===== Quantile-based confidence ellipse =====
def quantile_ellipse(x, y, ax, level=0.95, color='k', lw=2, alpha=0.2, zorder=3):
    """
    Draw a quantile ellipse using chi-square distribution.
    Equivalent to R-style confidence ellipses.
    """
    X = np.column_stack([x, y])
    cov = np.cov(X, rowvar=False)
    mean = X.mean(axis=0)

    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Angle of ellipse
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Chi-square scaling
    chi2_val = np.sqrt(chi2.ppf(level, df=2))
    width, height = 2 * chi2_val * np.sqrt(vals)

    from matplotlib.patches import Ellipse
    ell = Ellipse(
        mean, width, height, angle=angle,
        edgecolor=color, facecolor=color,
        lw=lw, alpha=alpha, zorder=zorder
    )
    ax.add_patch(ell)
    return ell


# ===== Plotting =====
fig, ax = plt.subplots(figsize=(8, 6))
color_before = "#4B8BBE"
color_after  = "#E69F00"

ax.scatter(P0[:, 0], P0[:, 1], s=10, alpha=0.25, color=color_before,
           label=f"Before (N={len(X0)})", zorder=2)
ax.scatter(P1[:, 0], P1[:, 1], s=45, alpha=0.85, color=color_after,
           label=f"After (N={len(X1)})", zorder=3)

# 95% confidence ellipses
quantile_ellipse(P0[:, 0], P0[:, 1], ax, level=0.95, color=color_before, alpha=0.25, lw=2.5)
quantile_ellipse(P1[:, 0], P1[:, 1], ax, level=0.95, color=color_after, alpha=0.25, lw=2.5)

# Aesthetic adjustments
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.axvline(0, color='gray', lw=0.8, ls='--')
ax.set_xlabel(f"PC1 ({var_ratio[0]:.1f}% var)")
ax.set_ylabel(f"PC2 ({var_ratio[1]:.1f}% var)")
ax.set_title("PCA of Feature Space (Before vs After)", fontsize=13)
ax.legend(frameon=True, loc="best")
ax.set_aspect('auto')
plt.tight_layout()

# ===== Save figure =====
out_png = "output/figures/pca.png"
plt.savefig(out_png, dpi=600, bbox_inches="tight")

print(f"[OK] Saved: {out_png}")
plt.show()
