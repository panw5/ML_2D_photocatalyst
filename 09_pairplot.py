import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# -------- Paths --------
total_path   = "output1/c2db_export.csv"
filtered_path = "output/filtered_by_all.csv"

# -------- Load and derive features --------
df0 = pd.read_csv(total_path)
df1 = pd.read_csv(filtered_path)

def enrich(df):
    """Add dielectric constant average if available."""
    df = df.copy()
    if {'alphax_el', 'alphay_el', 'alphaz_el'}.issubset(df.columns):
        df['alpha_avg'] = df[['alphax_el', 'alphay_el', 'alphaz_el']].mean(axis=1)
    return df

df0 = enrich(df0)
df1 = enrich(df1)

# -------- Select feature candidates (must exist in both) --------
cand = ['gap_hse', 'vbm_hse', 'cbm_hse', 'ehull', 'hform', 'emass_cbm', 'alpha_avg', 'thickness']
feats = [c for c in cand if c in df0.columns and c in df1.columns]

if len(feats) == 0:
    raise ValueError("None of the selected candidate features exist in both datasets. Please check column names.")

# -------- Convert to numeric + handle missing values --------
def to_numeric(df, cols):
    """Convert selected columns to numeric values and clean infinities."""
    X = df[cols].apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    return X

X0 = to_numeric(df0, feats)
X1 = to_numeric(df1, feats)

# Fit imputer on combined data (median strategy)
imp = SimpleImputer(strategy="median").fit(pd.concat([X0, X1], ignore_index=True))

X0 = pd.DataFrame(imp.transform(X0), columns=feats)
X1 = pd.DataFrame(imp.transform(X1), columns=feats)

X0['__set__'] = 'Before'
X1['__set__'] = 'After'
df_plot_all = pd.concat([X0, X1], ignore_index=True)

# -------- Split features into groups of 4 for plotting --------
def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

sns.set_theme(style="whitegrid")

for i, group in enumerate(chunks(feats, 4), start=1):
    g = sns.pairplot(
        df_plot_all,
        vars=group,
        hue="__set__",
        corner=True,
        diag_kind="kde",                         # KDE for diagonal density plots
        height=2.8,
        plot_kws={"alpha": 0.45, "s": 14, "edgecolor": None},
        diag_kws={"fill": True, "bw_adjust": 1.2}  # Smoother KDE
    )

    # ===== Fix axis labels (order & alignment) =====
    vars_now = group
    n = len(vars_now)

    # Left-most column: y-axis = vars_now[r]
    for r in range(n):
        ax = g.axes[r, 0] if g.axes.ndim == 2 else g.axes[r]
        if ax is not None:
            ax.set_ylabel(vars_now[r], fontsize=11, labelpad=8)
            ax.yaxis.set_label_position("left")
            ax.yaxis.set_visible(True)
            for tick in ax.get_yticklabels():
                tick.set_visible(True)

    # Bottom row: x-axis = vars_now[c]
    for c in range(n):
        ax = g.axes[n-1, c] if g.axes.ndim == 2 else g.axes[c]
        if ax is not None:
            ax.set_xlabel(vars_now[c], fontsize=11, labelpad=6)
            ax.xaxis.set_label_position("bottom")
            ax.xaxis.set_visible(True)
            for tick in ax.get_xticklabels():
                tick.set_visible(True)

    # Apply unified tick style & left alignment
    for ax in (g.axes.flat if g.axes.ndim == 2 else [*g.axes]):
        if ax is not None:
            ax.tick_params(axis='both', labelsize=9)
            ax.set_anchor('W')  # Align left

    # Force top-left subplot y-axis label to show
    ax00 = g.axes[0, 0]
    if ax00 is not None:
        ax00.set_ylabel(vars_now[0], fontsize=11, labelpad=8)
        ax00.yaxis.set_label_coords(-0.2, 0.5)

    # ===== Title and layout =====
    g.fig.suptitle(
        "Pairwise Relationships",
        y=0.97, fontsize=14
    )
    g.fig.subplots_adjust(left=0.10, bottom=0.12, right=0.98, top=0.93)

    out = "pairplot.png"
    g.fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {out}")
