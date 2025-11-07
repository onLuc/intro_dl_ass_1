from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# --------------------------
# CLI args
# --------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="./results/epoch_stats_long.csv", help="Path to per-epoch CSV")
    ap.add_argument("--summary", default="./results/results_summary.csv", help="Path to summary CSV")
    ap.add_argument("--outdir", default="./images", help="Output directory for results")
    return ap.parse_args()


# --------------------------
# Globals / styling
# --------------------------

# ======= Shared styling (top of file, after imports) =======
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 240,
    "figure.figsize": (8.5, 4.8),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Keep your existing color maps but make them a bit softer
BB_COLORS = {
    "light":  "#6F6AB1",  # muted indigo
    "medium": "#E67E76",  # muted coral
    "heavy":  "#4AA273",  # muted green
}
# Color-blind–safe, high-contrast palette (Okabe–Ito)
REP_COLORS = {
    "clf":               "#CC79A7",  # purple
    "reg":               "#009E73",  # green
    "multi-head":        "#D55E00",  # vermilion (far from green)
    "circle":            "#E69F00",  # orange
    "multi-head-circle": "#56B4E9",  # sky blue
}


# Small helper to add clean value labels without overlap
def _bar_labels(ax, bars, fmt="{:.2f}", pad=2):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h) and h > 0:
            ax.annotate(fmt.format(h),
                        xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, pad),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=9, rotation=0, clip_on=False)

# Optional: exponential moving average for smoothing epoch curves
def _ema(y, alpha=0.2):
    out = []
    s = None
    for v in y:
        s = v if s is None else alpha*v + (1-alpha)*s
        out.append(s)
    return np.array(out)

BB_ORDER = ["light", "medium", "heavy"]

# 5 output types (updated)
# - clf: any classifier_*bins (24/144/720); aggregated across bins as one category
# - reg: regression
# - multi-head: hour classification + minute regression (multi_regular)
# - circle: single circle (sin/cos) for 12h (minutes after midnight mod 720)
# - multi-head-circle: hour classification + minute circle (multi_circle)
REP_ORDER = ["multi-head-circle", "circle", "reg", "clf", "multi-head"]
REP_LABELS = {
    "clf": "clf (24/144/720)",
    "reg": "reg",
    "multi-head": "multi-head",
    "circle": "circle",
    "multi-head-circle": "multi-head-circle",
}
# backbone line styles for epoch results
BB_LINESTYLE = {"light": ":", "medium": "--", "heavy": "-"}


# --------------------------
# Helpers: parse model name
# --------------------------
MODEL_RE = re.compile(
    r"^(?P<backbone>light|medium|heavy)__"
    r"(?P<kind>classifier_(?P<bins>\d+)bins|regression|circle|multi_regular|multi_circle)"
    r"$"
)

def parse_model(model_name: str) -> Tuple[str, str, str]:
    """
    Returns (backbone, rep, subtype)
      rep in REP_ORDER:
        - clf
        - reg
        - multi-head
        - circle
        - multi-head-circle
      subtype: e.g., "24bins" for classifier; "" otherwise
    """
    m = MODEL_RE.match(model_name.strip())
    if not m:
        # Try to be lenient if naming differs slightly
        # Fallbacks:
        bb = "unknown"
        rep = "unknown"
        sub = ""
        if "__" in model_name:
            bb = model_name.split("__", 1)[0]
        if "classifier" in model_name:
            rep = "clf"
            bins = re.findall(r"(\d+)bins", model_name)
            sub = f"{bins[0]}bins" if bins else ""
        elif "multi_regular" in model_name:
            rep = "multi-head"
        elif "multi_circle" in model_name:
            rep = "multi-head-circle"
        elif "regression" in model_name:
            rep = "reg"
        elif "circle" in model_name:
            rep = "circle"
        return bb, rep, sub

    bb = m.group("backbone")
    kind = m.group("kind")
    bins = m.group("bins")

    if kind.startswith("classifier_"):
        rep = "clf"
        sub = f"{bins}bins"
    elif kind == "regression":
        rep = "reg"
        sub = ""
    elif kind == "multi_regular":
        rep = "multi-head"
        sub = ""
    elif kind == "circle":
        rep = "circle"
        sub = ""
    elif kind == "multi_circle":
        rep = "multi-head-circle"
        sub = ""
    else:
        rep = "unknown"
        sub = ""
    return bb, rep, sub


def ensure_columns_from_model(df: pd.DataFrame) -> pd.DataFrame:
    """Adds backbone, rep, subtype columns parsed from 'model'."""
    if "model" not in df.columns:
        return df.copy()
    d = df.copy()
    bbs, reps, subs = [], [], []
    for m in d["model"].astype(str):
        bb, rep, sub = parse_model(m)
        bbs.append(bb)
        reps.append(rep)
        subs.append(sub)
    d["backbone"] = bbs
    d["rep"] = reps
    d["rep_subtype"] = subs
    return d


# --------------------------
# Plots
# --------------------------
def plot_mean_cse_per_epoch(df_epoch: pd.DataFrame, outdir: Path):
    grp = (df_epoch.dropna(subset=["val_cse_mean"])
           .groupby(["backbone", "rep", "epoch"], as_index=False)["val_cse_mean"].mean())

    fig, ax = plt.subplots(figsize=(10.5, 4.9))

    for rep in REP_ORDER:
        rep_df = grp[grp["rep"] == rep]
        for bb in BB_ORDER:
            g = rep_df[rep_df["backbone"] == bb]
            if g.empty:
                continue
            ls = BB_LINESTYLE.get(bb, "-")
            y = _ema(g["val_cse_mean"].to_numpy(), alpha=0.2)
            ax.plot(g["epoch"], y,
                    label=f"{REP_LABELS[rep]} — {bb}",
                    color=REP_COLORS[rep], linestyle=ls, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean CSE (minutes)")
    # Use a suptitle so we can position legends *under* it
    fig.suptitle("Mean CSE per Epoch (aggregated by representation × backbone)",
                 y=0.99, fontsize=13, fontweight="semibold")
    # Make room under the title
    fig.subplots_adjust(top=0.78)
    ax.margins(x=0.02, y=0.06)

    # Build separate legend handles
    color_handles = [Line2D([0], [0], color=REP_COLORS[r], lw=3, label=REP_LABELS[r]) for r in REP_ORDER]
    ls_handles = [Line2D([0], [0], color="black", linestyle=BB_LINESTYLE[b], lw=2, label=b.capitalize())
                  for b in BB_ORDER]

    # # Place legends BELOW the title (stacked)
    # fig.legend(handles=color_handles, title="Representation",
    #            loc="upper center", bbox_to_anchor=(0.5, 0.90),
    #            ncol=len(color_handles), frameon=False)
    # fig.legend(handles=ls_handles, title="Backbone",
    #            loc="upper center", bbox_to_anchor=(0.5, 0.84),
    #            ncol=len(ls_handles), frameon=False)

    all_handles = color_handles + ls_handles
    fig.legend(all_handles, [h.get_label() for h in all_handles],
               loc="upper center", bbox_to_anchor=(0.5, 0.87),
               ncol=5, frameon=False)

    fig.savefig(outdir / "mean_cse_per_epoch.png", bbox_inches="tight")
    plt.close(fig)


def plot_mean_cse_per_epoch_per_backbone(df_epoch: pd.DataFrame, outdir: Path):
    sub = (df_epoch.dropna(subset=["val_cse_mean"])
           .groupby(["backbone", "rep", "epoch"], as_index=False)["val_cse_mean"].mean())

    fig, axes = plt.subplots(1, len(BB_ORDER), figsize=(12.5, 4.1), sharey=True, constrained_layout=True)
    for ax, bb in zip(axes, BB_ORDER):
        g = sub[sub["backbone"] == bb]
        for rep in REP_ORDER:
            gg = g[g["rep"] == rep]
            if gg.empty:
                continue
            y_raw = gg["val_cse_mean"].to_numpy()
            y_smooth = _ema(y_raw, alpha=0.2)   # comment out if you want raw only
            ax.plot(gg["epoch"], y_smooth, color=REP_COLORS[rep], lw=2, label=REP_LABELS[rep])
            ax.set_title(bb.capitalize())
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean CSE (minutes)")

    # Single legend on top, outside
    handles = [Line2D([0], [0], color=REP_COLORS[r], lw=3, label=REP_LABELS[r]) for r in REP_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Mean CSE per Epoch (representation × backbone)", y=1.12, fontsize=13, fontweight="semibold")
    fig.savefig(outdir / "mean_cse_per_epoch_per_backbone.png", bbox_inches="tight")
    plt.close(fig)


def plot_training_time_per_backbone(df_epoch: pd.DataFrame, outdir: Path):
    if "time_s" not in df_epoch.columns:
        print("Column 'time_s' missing; skipping training time plot.")
        return

    desired_order = ["heavy", "medium", "light"]  # choose your preferred order
    g = (df_epoch.dropna(subset=["time_s"])
         .groupby("backbone", as_index=False)["time_s"].mean())

    present = [b for b in desired_order if b in g["backbone"].unique()]
    if not present:
        print("No known backbones found; skipping training time plot.")
        return
    g = g[g["backbone"].isin(present)].copy()
    g["backbone"] = pd.Categorical(g["backbone"], categories=present, ordered=True)
    g = g.sort_values("backbone")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    xs = np.arange(len(g))
    colors = [BB_COLORS[str(b)] for b in g["backbone"]]
    bars = ax.bar(xs, g["time_s"], color=colors, edgecolor="white", linewidth=0.8)
    _bar_labels(ax, bars, fmt="{:.1f}", pad=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b).capitalize() for b in g["backbone"]])
    ax.set_ylabel("Avg. Time per Epoch (s)")
    ax.set_title("Average Training Time per Epoch by Backbone")
    fig.savefig(outdir / "mean_training_time_per_backbone.png")
    plt.close(fig)


def plot_final_mean_cses(df_summary: pd.DataFrame, outdir: Path):
    agg = (df_summary
           .dropna(subset=["mean_cse"])
           .groupby(["backbone", "rep"], as_index=False)["mean_cse"]
           .mean())

    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    x = np.arange(len(REP_ORDER))
    width = 0.22
    handles = []

    for j, bb in enumerate(BB_ORDER):
        xj = x + (j - (len(BB_ORDER)-1)/2) * width
        y = []
        for rep in REP_ORDER:
            row = agg[(agg["backbone"] == bb) & (agg["rep"] == rep)]
            y.append(float(row["mean_cse"].iloc[0]) if len(row) else np.nan)

        bars = ax.bar(xj, y, width=width, label=bb.capitalize(),
                      color=BB_COLORS.get(bb), edgecolor="white", linewidth=0.6)
        _bar_labels(ax, bars, fmt="{:.2f}", pad=3)

    ax.set_xticks(x)
    ax.set_xticklabels([REP_LABELS[r] for r in REP_ORDER], rotation=12, ha="right")
    ax.set_ylabel("Mean CSE (minutes)")
    ax.set_title("Mean CSE by Representation (aggregated) × Backbone")
    ax.legend(ncol=3, frameon=False, bbox_to_anchor=(1.0, 1.02), loc="lower right")
    fig.savefig(outdir / "final_mean_cses.png")
    plt.close(fig)


def plot_representation_mean_over_medium_heavy(df_summary: pd.DataFrame, outdir: Path):
    use = df_summary.copy()
    use = use[use["backbone"].isin(["medium", "heavy"])]

    # First: average per (backbone, rep) so classifier bins collapse
    per_bb_rep = (use.dropna(subset=["mean_cse"])
                    .groupby(["backbone","rep"], as_index=False)["mean_cse"]
                    .mean())
    # Then: average across the two backbones for each rep
    agg = (per_bb_rep.groupby("rep", as_index=False)["mean_cse"].mean())

    # ensure order/labels
    agg["rep"] = pd.Categorical(agg["rep"], categories=REP_ORDER, ordered=True)
    agg = agg.sort_values("rep")

    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    x = np.arange(len(agg))
    colors = [REP_COLORS[str(r)] for r in agg["rep"]]
    bars = ax.bar(x, agg["mean_cse"], color=colors, edgecolor="white", linewidth=0.6)
    _bar_labels(ax, bars, fmt="{:.2f}", pad=3)

    ax.set_xticks(x)
    ax.set_xticklabels([REP_LABELS[str(r)] for r in agg["rep"]], rotation=12, ha="right")
    ax.set_ylabel("Mean CSE (minutes)")
    ax.set_title("Mean CSE by Representation (averaged over Medium + Heavy)")
    fig.savefig(outdir / "mean_cse_representation_medium_heavy_avg.png")
    plt.close(fig)



# --------------------------
# Main
# --------------------------
def main():
    args = get_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    df_epoch = pd.read_csv(args.epoch)
    df_summary = pd.read_csv(args.summary)

    # Ensure parsed columns exist
    df_epoch = ensure_columns_from_model(df_epoch)
    df_summary = ensure_columns_from_model(df_summary)

    # Normalize/clean: keep only known reps/backbones
    df_epoch = df_epoch[df_epoch["backbone"].isin(BB_ORDER)]
    df_epoch = df_epoch[df_epoch["rep"].isin(REP_ORDER)]
    df_summary = df_summary[df_summary["backbone"].isin(BB_ORDER)]
    df_summary = df_summary[df_summary["rep"].isin(REP_ORDER) | df_summary["rep"].eq("clf")]  # keep clf too

    # 1) Mean CSE per epoch (aggregated by rep×bb)
    plot_mean_cse_per_epoch(df_epoch, outdir)

    # 2) Mean CSE per epoch per backbone
    plot_mean_cse_per_epoch_per_backbone(df_epoch, outdir)

    # 3) Avg training time per epoch by backbone
    plot_training_time_per_backbone(df_epoch, outdir)

    # 4) Mean CSE by representation (5 types) × backbone
    plot_final_mean_cses(df_summary, outdir)

    # 5) CSE by rep
    plot_representation_mean_over_medium_heavy(df_summary, outdir)



if __name__ == "__main__":
    main()


# Run - arguments
# Path to per-epoch CSV -> --epoch, default="./results/epoch_stats_long.csv"
# Path to summary CSV -> --summary", default="./results/results_summary.csv"
# Output directory for results -> --outdir", default="images"