#!/usr/bin/env python3
"""
CMD Isochrone Fitter (PARSEC, log age 6–8) — simplified version (no argparse)

Usage:
- Set the file paths and column names directly in the script below.
- Run with: python fit_isochrones_simple.py

Performs:
  • Isochrone filtering ((g - i) < 3, g > 0)
  • Loss calculation for each log(age)
  • Best-fit isochrone selection
  • Plots loss vs. age and CMD overlay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from scipy.spatial import cKDTree as KDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# === User Inputs ===
DATA_FILE = "ASCC19_cluster_members.csv"
ISO_FILE = "hehe.dat"

DATA_G_COL = "M_g"
DATA_I_COL = "M_i"
ISO_G_COL = "gmag"
ISO_I_COL = "imag"
ISO_AGE_COL = "logAge"

OUT_PREFIX = "results/fit"
GI_MAX = 3.0
G_MIN = 0.0

# === Functions ===
def coerce_log_age(age_values):
    vals = np.asarray(age_values, dtype=float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if 4 <= vmin <= 12 and 4 <= vmax <= 12:
        return vals
    else:
        return np.log10(vals)

def nearest_dist_squared(iso_xy, data_xy):
    if iso_xy.shape[0] == 0:
        return np.full(data_xy.shape[0], np.nan)
    if HAVE_SCIPY:
        tree = KDTree(iso_xy)
        dists, _ = tree.query(data_xy, k=1)
        return dists**2
    else:
        sq = np.empty(data_xy.shape[0])
        for i, pt in enumerate(data_xy):
            sq[i] = np.min(np.sum((iso_xy - pt)**2, axis=1))
        return sq

def compute_age_loss(data_g, data_i, iso_df, iso_g_col, iso_i_col, iso_age_col):
    data_color = data_g - data_i
    data_mag = data_g
    data_xy = np.column_stack([data_color, data_mag])

    iso_df = iso_df.copy()
    iso_df["logAge"] = coerce_log_age(iso_df[iso_age_col])
    iso_color = iso_df[iso_g_col] - iso_df[iso_i_col]
    mask = (iso_color < GI_MAX) & (iso_df[iso_g_col] > G_MIN)
    iso_df = iso_df.loc[mask]

    ages = np.sort(iso_df["logAge"].unique())
    rows = []
    for age in ages:
        iso_age_df = iso_df.loc[iso_df["logAge"] == age]
        iso_xy = np.column_stack([
            iso_age_df[iso_g_col] - iso_age_df[iso_i_col],
            iso_age_df[iso_g_col]
        ])
        d2 = nearest_dist_squared(iso_xy, data_xy)
        loss = np.nanmean(d2)
        rows.append((age, loss))
    return pd.DataFrame(rows, columns=["logAge", "loss"])

def select_best(loss_df):
    best = loss_df.loc[loss_df["loss"].idxmin()]
    return float(best["logAge"]), float(best["loss"])

def plot_loss(loss_df, out_path, show=True):
    plt.figure()
    plt.plot(loss_df["logAge"], loss_df["loss"], marker="o", lw=1.5)
    plt.xlabel("log10(Age/yr)")
    plt.ylabel("Mean squared distance (Loss)")
    plt.title("Loss vs Age")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_cmd(data_g, data_i, iso_df, best_age, out_path):
    iso_df = iso_df.copy()
    iso_df["logAge"] = coerce_log_age(iso_df[ISO_AGE_COL])
    iso_color = iso_df[ISO_G_COL] - iso_df[ISO_I_COL]
    mask = (iso_color < GI_MAX) & (iso_df[ISO_G_COL] > G_MIN)
    iso_df = iso_df.loc[mask]

    curve = iso_df.loc[np.isclose(iso_df["logAge"], best_age, atol=1e-6)]
    if curve.empty:
        nearest_age = iso_df["logAge"].iloc[np.argmin(np.abs(iso_df["logAge"] - best_age))]
        curve = iso_df.loc[iso_df["logAge"] == nearest_age]

    plt.figure()
    plt.scatter(data_g - data_i, data_g, s=10, alpha=0.6, label="Data")
    plt.plot(curve[ISO_G_COL] - curve[ISO_I_COL], curve[ISO_G_COL], lw=2, label=f"logAge={best_age:.3f}")
    plt.gca().invert_yaxis()
    plt.xlabel("g - i")
    plt.ylabel("g")
    plt.title("CMD Best Fit")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# === Main ===
data = pd.read_csv(DATA_FILE)
iso = pd.read_csv(ISO_FILE, delim_whitespace=True, comment="#", engine="python")

data_g = pd.to_numeric(data[DATA_G_COL], errors="coerce")
ldata_i = pd.to_numeric(data[DATA_I_COL], errors="coerce")
mask = np.isfinite(data_g) & np.isfinite(ldata_i)
data_g, data_i = data_g[mask].to_numpy(), ldata_i[mask].to_numpy()

loss_df = compute_age_loss(data_g, data_i, iso, ISO_G_COL, ISO_I_COL, ISO_AGE_COL)

best_age, best_loss = select_best(loss_df)
print(f"Best-fit logAge = {best_age:.4f}, loss = {best_loss:.4e}")

out_prefix = Path(OUT_PREFIX)
out_prefix.parent.mkdir(parents=True, exist_ok=True)

loss_df.to_csv(f"{out_prefix}_loss_table.csv", index=False)
plot_loss(loss_df, f"{out_prefix}_loss_vs_age.png", show=True)
plot_cmd(data_g, data_i, iso, best_age, f"{out_prefix}_cmd_bestfit.png")
