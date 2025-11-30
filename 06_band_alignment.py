import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os, re

# ---------- Load and clean data ----------
candidates = ["output/filtered_by_all.csv", "output/tables/filtered_by_all.csv"]
csv_path = next((p for p in candidates if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError("filtered_by_all.csv not found in output/ or output/tables/")

df = pd.read_csv(csv_path).head(25).copy()

def clean_uid(u: str) -> str:
    s = str(u)
    s = re.sub(r"^\d+", "", s)      # remove leading numbers
    s = re.sub(r"-\d+$", "", s)     # remove ending “-number”
    return s

# ---------- Convert digits to chemical subscripts ----------
def to_subscript(s: str) -> str:
    """Convert digits to chemical formula subscripts."""
    sub_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return re.sub(r'\d+', lambda m: m.group(0).translate(sub_map), s)

labels = df["uid"].astype(str).map(clean_uid).map(to_subscript)

vbm = pd.to_numeric(df["vbm_hse"], errors="coerce").values
cbm = pd.to_numeric(df["cbm_hse"], errors="coerce").values

# Water redox potentials (vs vacuum)
E_red = -4.44  # H+/H2
E_ox  = -5.67  # O2/H2O

# ---------- Colors ----------
color_cbm = "#9ED9A3"   # slightly darker light green (CBM)
color_vbm = "#F3AFCF"   # slightly darker light pink (VBM)

y_min = -8
y_max = -2
x = np.arange(len(df))
bar_w = 0.8

fig, ax = plt.subplots(figsize=(16, 6))

# Bottom bars: CBM
ax.bar(x, cbm - y_min, bottom=y_min, width=bar_w, color=color_cbm, zorder=1)

# Top bars: VBM
ax.bar(x, 0 - vbm, bottom=vbm, width=bar_w, color=color_vbm, zorder=2)

# White masks to carve out the band gap (no border)
gap_heights = cbm - vbm
mask = gap_heights > 0
ax.bar(x[mask], gap_heights[mask], bottom=vbm[mask],
       width=bar_w * 0.9, color="white", edgecolor="none", zorder=3)

# Reference lines
ax.axhline(E_red, color="blue", linestyle="--", linewidth=1.5, zorder=4)
ax.axhline(E_ox,  color="black", linestyle="--", linewidth=1.5, zorder=4)

# Axis labels and limits
ax.set_ylim(y_min, y_max)
ax.set_xlim(-0.6, len(df)-0.4)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=10)

ax.set_ylabel("Band Edges (eV)", fontsize=12)
ax.yaxis.set_label_coords(-0.04, 0.45)

# ---------- Legend ----------
legend_handles = [
    Line2D([0], [0], color="blue", linestyle="--", linewidth=1.5, label="H⁺/H₂ (−4.44 eV)"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="O₂/H₂O (−5.67 eV)"),
    mpatches.Patch(facecolor=color_cbm, label="CBM (HSE)"),
    mpatches.Patch(facecolor=color_vbm, label="VBM (HSE)")
]
leg = ax.legend(handles=legend_handles, loc="upper right", frameon=True)

# Style legend box
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("#bbbbbb")
leg.get_frame().set_alpha(0.95)

# Adjust left spacing
fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.18)

# ---------- Save ----------
os.makedirs("output/figures", exist_ok=True)
out_path = "output/figures/band_alignment_selected_2D_materials.png"
plt.savefig(out_path, dpi=600)
plt.show()
print(f"Saved figure to: {out_path}")
