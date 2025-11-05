#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_EPOCH = Path("plots/epoch_stats_long.csv")  # adjust if needed
OUTDIR = Path("images")
OUTDIR.mkdir(parents=True, exist_ok=True)

# === Colors for each backbone (consistent + readable) ===
BB_ORDER = ["light", "medium", "heavy"]
BB_COLORS = {
    "light":  "#4C78A8",  # blue
    "medium": "#F58518",  # orange
    "heavy":  "#54A24B",  # green
}

def infer_backbone(name: str) -> str:
    n = str(name)
    if "light" in n: return "light"
    if "heavy" in n: return "heavy"
    return "medium"

def _ordered_group_mean(df_in: pd.DataFrame, colname: str) -> pd.DataFrame:
    """Return dataframe with mean(colname) grouped by backbone, ordered L→M→H."""
    d = (
        df_in.dropna(subset=[colname])
            .groupby("backbone", as_index=False)[colname]
            .mean()
    )
    # enforce order; keep only known backbones
    d = d[d["backbone"].isin(BB_ORDER)].copy()
    d["backbone"] = pd.Categorical(d["backbone"], categories=BB_ORDER, ordered=True)
    d = d.sort_values("backbone")
    return d

# --- Load data ---
df = pd.read_csv(CSV_EPOCH)

# keep only needed columns (ignore others)
need = [c for c in ["model", "epoch", "time", "cse", "backbone"] if c in df.columns]
missing = {"cse", "backbone"} - set(need)
if "cse" not in need:
    raise ValueError("epoch_stats_long.csv must contain a 'cse' column.")
df = df[need].copy()

# infer backbone if missing
if "backbone" not in df.columns or df["backbone"].isna().all():
    df["backbone"] = df["model"].apply(infer_backbone)

# enforce order
df["backbone"] = pd.Categorical(df["backbone"], categories=BB_ORDER, ordered=True)

# --- Mean CSE by backbone (ordered, colored) ---
g_cse = _ordered_group_mean(df, "cse")

plt.figure(figsize=(6.2, 4.2))
xs = np.arange(len(g_cse))
bar_colors = [BB_COLORS[str(b)] for b in g_cse["backbone"]]
plt.bar(xs, g_cse["cse"], color=bar_colors, edgecolor="#222222", linewidth=0.6)
plt.xticks(xs, [str(b).capitalize() for b in g_cse["backbone"]])
plt.ylabel("Mean CSE (minutes)")
plt.title("Mean CSE by Backbone (ordered light → medium → heavy)")
# annotate values
for x, y in zip(xs, g_cse["cse"]):
    plt.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
out_cse = OUTDIR / "mean_cse_per_epoch_per_backbone.png"
plt.savefig(out_cse, dpi=220)
plt.close()
print(f"Saved {out_cse}")

# --- Mean time/epoch by backbone (ordered, colored) ---
if "time" in df.columns and df["time"].notna().any():
    g_time = _ordered_group_mean(df, "time")
    plt.figure(figsize=(6.2, 4.2))
    xs = np.arange(len(g_time))
    bar_colors = [BB_COLORS[str(b)] for b in g_time["backbone"]]
    plt.bar(xs, g_time["time"], color=bar_colors, edgecolor="#222222", linewidth=0.6)
    plt.xticks(xs, [str(b).capitalize() for b in g_time["backbone"]])
    plt.ylabel("Avg. Time per Epoch (s)")
    plt.title("Average Training Time per Epoch by Backbone (ordered)")
    for x, y in zip(xs, g_time["time"]):
        plt.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out_time = OUTDIR / "mean_training_time_per_backbone.png"
    plt.savefig(out_time, dpi=220)
    plt.close()
    print(f"Saved {out_time}")
else:
    print("Column 'time' missing or all-NaN; skipping time/epoch plot.")
