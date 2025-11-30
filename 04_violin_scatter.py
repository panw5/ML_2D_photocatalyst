#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script generates two main figures:
(a) HSE band gap violin plot (Table 1 vs Table 2)
(b) HSE CBM/VBM scatter plot (Table 1 vs Table 3, colored by bravais_type + three-level filtering)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ---------- File paths ----------
table1_path = "output1/c2db_export.csv"          # Full dataset
table2_path = "output/filtered_by_gap.csv"       # Filtered by gap conditions
table3_path = "output/filtered_by_bandedge.csv"  # Filtered by band edge conditions
os.makedirs("output/figures", exist_ok=True)

# ---------- Load data ----------
df_all = pd.read_csv(table1_path)
df_gap = pd.read_csv(table2_path)
df_band = pd.read_csv(table3_path)

# =====================================
# Figure (a): Violin plot — HSE band gap distribution
# (shape preserved + colored by bravais_type)
# =====================================

# Combine two tables into one with an additional “dataset” column
df_all2 = df_all.assign(dataset="All candidates")
df_gap2 = df_gap.assign(dataset="Filtered (1.63 < Eg < 3)")
df_plot = pd.concat([df_all2, df_gap2], ignore_index=True)

# Fix the order of bravais types
order = sorted(df_plot["bravais_type"].dropna().unique().tolist())
hue_order = ["All candidates", "Filtered (1.63 < Eg < 3)"]

# Parameters for consistent shape
common_kwargs = dict(
    data=df_plot,
    x="bravais_type",
    y="gap_hse",
    hue="dataset",
    hue_order=hue_order,
    order=order,
    dodge=False,
    cut=0,
    linewidth=1.0,
    inner="box",
    inner_kws={"box_width": 1.5},
    palette={"All candidates": "lightgray",
             "Filtered (1.63 < Eg < 3)": "lightgray"},  # temporary uniform color
)

plt.figure(figsize=(8, 5))
ver = tuple(int(p) for p in sns.__version__.split(".")[:2])
if ver >= (0, 15):
    ax = sns.violinplot(**common_kwargs, density_norm="count", common_norm=False)
else:
    ax = sns.violinplot(**common_kwargs, scale="count", scale_hue=True)

# Recolor each bravais_type
type_palette = dict(zip(order, sns.color_palette("muted", n_colors=len(order))))
n_hue = len(hue_order)
polys = [c for c in ax.collections if isinstance(c, mpl.collections.PolyCollection)]

for i, cat in enumerate(order):
    base = type_palette[cat]
    # Background (All candidates)
    poly_bg = polys[i * n_hue + 0]
    poly_bg.set_facecolor((*base, 0.35))
    poly_bg.set_edgecolor(base)
    # Foreground (Filtered)
    poly_fg = polys[i * n_hue + 1]
    poly_fg.set_facecolor((*base, 0.90))
    poly_fg.set_edgecolor(base)

# Draw red lines for the filtering range
plt.axhline(1.63, color='r', linestyle='--', lw=1)
plt.axhline(3, color='r', linestyle='--', lw=1)

# Legend
handles = [
    Patch(facecolor=(0.5, 0.5, 0.5, 0.35), edgecolor=(0.5, 0.5, 0.5),
          label="All candidates"),
    Patch(facecolor=(0.5, 0.5, 0.5, 0.90), edgecolor=(0.5, 0.5, 0.5),
          label="Filtered candidates (1.63 < Eg < 3)"),
]
plt.legend(handles=handles, frameon=False, fontsize=10)

# Axis formatting
plt.ylabel("HSE Band Gap (eV)", fontsize=12)
plt.xlabel("bravais_type", fontsize=12)

plt.tight_layout()
plt.savefig("output/figures/6violin_hse_gap.png", dpi=600)
plt.close()
print("violin_hse_gap.png saved.")

# =====================================
# Figure (b): HSE band edge scatter plot
# (colored by bravais_type + three-stage filtering)
# =====================================

E_HER_0, E_OER_0 = -4.24, -5.87  # vs vacuum

plt.figure(figsize=(6.5, 6.5))

# Color palette for each bravais_type
order = sorted(df_all["bravais_type"].dropna().unique().tolist())
type_palette = dict(zip(order, sns.color_palette("Set2", n_colors=len(order))))

# Layer 1: all candidates in gray
plt.scatter(df_all["vbm_hse"], df_all["cbm_hse"],
            c="lightgray", alpha=0.35, s=20, label="All candidates",
            edgecolors="none")

# Layer 2: materials passing the gap filter (light color)
for btype in order:
    sub = df_gap[df_gap["bravais_type"] == btype]
    if len(sub) > 0:
        color = type_palette[btype]
        plt.scatter(sub["vbm_hse"], sub["cbm_hse"],
                    color=(*color, 0.45), s=40, edgecolors="none")

# Layer 3: passing gap + band edge filter (strong color)
for btype in order:
    sub = df_band[df_band["bravais_type"] == btype]
    if len(sub) > 0:
        color = type_palette[btype]
        plt.scatter(sub["vbm_hse"], sub["cbm_hse"],
                    color=(*color, 0.9), s=55, edgecolors="k", linewidths=0.3)

# HER/OER levels
plt.axvline(E_OER_0, color='k', linestyle='--', lw=1, label='OER level (pH=0)')
plt.axhline(E_HER_0, color='k', linestyle='--', lw=1, label='HER level (pH=0)')

ax = plt.gca()

# Set axis limits
ax.set_xlim(-8, -3)
ax.set_ylim(-6, -2)

# Draw constant band gap lines Eg = 1.63 eV and Eg = 3.0 eV
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 500)
ax.plot(x, x + 1.63, 'r--', lw=1, zorder=2)
ax.plot(x, x + 3.0, 'r--', lw=1, zorder=2)

# Axis labels
plt.xlabel("VBM (eV)", fontsize=12)
plt.ylabel("CBM (eV)", fontsize=12)
plt.xlim(-8, -3)
plt.ylim(-6, -2)

# Legend
legend_elements = [
    Line2D([], [], marker='o', color='lightgray', markersize=6,
           linestyle='None', label='All candidates'),
    Line2D([], [], marker='o', color='gray', markersize=8,
           linestyle='None', markerfacecolor='gray', alpha=0.5,
           label='Filtered by gap'),
    Line2D([], [], marker='o', color='k', markersize=8,
           linestyle='None', markerfacecolor='k', alpha=0.9,
           label='Filtered by gap + band edge'),
]
plt.legend(handles=legend_elements, frameon=False, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig("output/figures/6scatter_band_edges_colored.png", dpi=600)
plt.close()
print("scatter_band_edges_colored.png saved.")
