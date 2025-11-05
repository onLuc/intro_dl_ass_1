# === Mean CSE per epoch with colors = representation, linestyles = backbone ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

CSV_EPOCH = Path("plots/epoch_stats_long.csv")   # adjust if needed
OUTDIR = Path("images")
OUTDIR.mkdir(exist_ok=True, parents=True)

# --- load ---
df = pd.read_csv(CSV_EPOCH)

# Keep only needed columns; val_loss intentionally ignored/excluded
need = [c for c in ["model", "epoch", "time", "cse", "type", "backbone"] if c in df.columns]
if "cse" not in need or "epoch" not in need:
    raise ValueError("epoch_stats_long.csv must contain at least 'epoch' and 'cse' columns.")
df = df[need].copy()

# --- enforce backbone order: light -> medium -> heavy ---
order_bb = ["light", "medium", "heavy"]
if "backbone" in df.columns:
    df["backbone"] = pd.Categorical(df["backbone"], categories=order_bb, ordered=True)
else:
    # Try to infer backbone from model name if missing
    def infer_bb(name: str) -> str:
        n = str(name)
        if "light" in n: return "light"
        if "heavy" in n: return "heavy"
        return "medium"
    df["backbone"] = df["model"].apply(infer_bb)
    df["backbone"] = pd.Categorical(df["backbone"], categories=order_bb, ordered=True)

# --- normalize representation/type ---
# Prefer the 'type' column if present; otherwise infer from model name
def infer_type_from_model(name: str) -> str:
    n = str(name)
    if "multi_circle" in n: return "multi-circle"
    if "circle" in n and "multi" not in n: return "circle"
    if "reg" in n: return "regression"
    if "clf" in n: return "classification"
    return "other"

if "type" not in df.columns:
    df["type"] = df["model"].apply(infer_type_from_model)

# Pretty labels for legend
pretty = {
    "classification": "Classification",
    "regression": "Regression",
    "circle": "Circle (sin/cos)",
    "multi-circle": "Multi-Circle (Hr+Angle)",
    "other": "Other"
}
df["type_pretty"] = df["type"].map(pretty).fillna(df["type"])

# Limit to known reps to keep the plot tidy
rep_order = ["classification", "regression", "circle", "multi-circle"]
present_types = [t for t in rep_order if (df["type"] == t).any()]
if not present_types:
    present_types = sorted(df["type"].dropna().unique().tolist())

# --- aggregate mean CSE per epoch for each (type, backbone) ---
agg = (
    df.dropna(subset=["cse"])
      .groupby(["type", "backbone", "epoch"], as_index=False)["cse"]
      .mean()
)

# --- styling: colors by representation, linestyles by backbone ---
# Pick a simple, readable palette
base_colors = {
    "classification": "#1f77b4",  # blue
    "regression":     "#2ca02c",  # green
    "circle":         "#ff7f0e",  # orange
    "multi-circle":   "#d62728",  # red
    "other":          "#7f7f7f"   # gray
}
# Fallback: any missing types get auto-colors
for t in present_types:
    base_colors.setdefault(t, None)

linestyles = {
    "light": "-",
    "medium": "--",
    "heavy": ":"
}

# --- plot ---
plt.figure(figsize=(9, 5.2))

# To keep consistent ordering in the plot:
for t in present_types:
    for bb in order_bb:
        d = agg[(agg["type"] == t) & (agg["backbone"] == bb)]
        if d.empty:
            continue
        label = f"{pretty.get(t, t)} / {bb.capitalize()}"
        plt.plot(
            d["epoch"], d["cse"],
            color=base_colors.get(t, None),
            linestyle=linestyles[bb],
            linewidth=1.6,
            alpha=0.95,
            label=label
        )

plt.xlabel("Epoch")
plt.ylabel("Mean CSE (minutes)")
plt.title("Mean CSE per Epoch")
plt.grid(alpha=0.15)

# Build two separate legends (color = rep, linestyle = backbone)
# 1) Color legend (representations)
color_handles = [
    Line2D([0], [0], color=base_colors.get(t, "k"), lw=2.4, label=pretty.get(t, t))
    for t in present_types
    if agg[agg["type"] == t].shape[0] > 0
]
# 2) Linestyle legend (backbones) — use neutral color
ls_handles = [
    Line2D([0], [0], color="#333333", linestyle=linestyles[bb], lw=2.4, label=bb.capitalize())
    for bb in order_bb
    if agg[agg["backbone"] == bb].shape[0] > 0
]

# Place legends without overlap
first_legend = plt.legend(handles=color_handles, title="Representation", loc="upper right", frameon=False)
plt.gca().add_artist(first_legend)
plt.legend(handles=ls_handles, title="Backbone", loc="upper center", ncol=len(ls_handles), frameon=False)

plt.tight_layout()
out_path = OUTDIR / "mean_cse_per_epoch_subset.png"
plt.savefig(out_path, dpi=220)
plt.close()
print(f"Saved {out_path}")
