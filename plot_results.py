"""
plot_results.py
---------------
Figures produced:
  fig_main_comparison.png      – bar chart of all methods at N=200
  fig_density_Eb_Pd.png        – E_b and P_d vs node count
  fig_ablation.png             – ablation radar/bar chart
  fig_hw_stress.png            – hardware stress line plots
  fig_pi_sensitivity.png       – sensitivity bar chart
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = "results"
FIGS_DIR    = "results"
os.makedirs(FIGS_DIR, exist_ok=True)

METHOD_COLORS = {
    "Glossy":   "#4C72B0",
    "LWB":      "#DD8452",
    "Crystal":  "#55A868",
    "Chaos":    "#C44E52",
    "Splash":   "#8172B2",
    "Proposed": "#D62728",
}


# ────────────────────────────────────────────────────────────────────────────
# Fig 1 – Main comparison bar chart (N=200)
# ────────────────────────────────────────────────────────────────────────────

def plot_main_comparison():
    path = f"{RESULTS_DIR}/results_main.csv"
    if not os.path.exists(path):
        print(f"[plot] Skipping fig_main_comparison (missing {path})")
        return

    df = pd.read_csv(path)
    metrics = ["E_b (mJ/KB)", "T_c (ms)", "P_d (%)", "G_p (KB/s)"]
    titles  = ["Energy per delivered byte\n$E_b$ (mJ/KB)",
               "Completion time\n$T_c$ (ms)",
               "Delivery probability\n$P_d$ (%)",
               "Goodput\n$G_p$ (KB/s)"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    x = np.arange(len(df))
    colors = [METHOD_COLORS.get(m, "#999999") for m in df["Method"]]

    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.bar(x, df[metric], color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Method"], rotation=35, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        # annotate proposed bar
        prop_idx = df.index[df["Method"] == "Proposed"][0]
        ax.bar(prop_idx, df[metric].iloc[prop_idx],
               color=METHOD_COLORS["Proposed"], edgecolor="black",
               linewidth=1.2, zorder=3)

    plt.suptitle("Performance comparison at $N=200$, $L=64$ bytes",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = f"{FIGS_DIR}/fig_main_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {out}")


# ────────────────────────────────────────────────────────────────────────────
# Fig 2 – Density comparison (E_b and P_d)
# ────────────────────────────────────────────────────────────────────────────

def plot_density():
    path = f"{RESULTS_DIR}/results_density.csv"
    if not os.path.exists(path):
        print(f"[plot] Skipping fig_density (missing {path})")
        return

    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for method, grp in df.groupby("Method"):
        color = METHOD_COLORS.get(method, "#999")
        ls    = "--" if method == "Glossy" else "-"
        marker = "o" if method == "Proposed" else "s"
        for ax, metric, ylabel in zip(
            axes,
            ["E_b (mJ/KB)", "P_d (%)"],
            ["$E_b$ (mJ/KB)", "$P_d$ (%)"]
        ):
            ax.plot(grp["Node count"], grp[metric],
                    color=color, linestyle=ls, marker=marker,
                    linewidth=2, markersize=7, label=method)
            ax.set_xlabel("Node count $N$")
            ax.set_ylabel(ylabel)
            ax.grid(linestyle="--", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_title("Energy per delivered byte vs density")
    axes[1].set_title("Delivery probability vs density")
    axes[0].legend(fontsize=8, framealpha=0.7)

    plt.tight_layout()
    out = f"{FIGS_DIR}/fig_density_Eb_Pd.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {out}")


# ────────────────────────────────────────────────────────────────────────────
# Fig 3 – Ablation study
# ────────────────────────────────────────────────────────────────────────────

def plot_ablation():
    path = f"{RESULTS_DIR}/results_ablation.csv"
    if not os.path.exists(path):
        print(f"[plot] Skipping fig_ablation (missing {path})")
        return

    df = pd.read_csv(path)
    metrics = ["E_b (mJ/KB)", "T_c (ms)", "P_d (%)", "G_p (KB/s)", "ATC (ms)"]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    colors = ["#6baed6", "#74c476", "#fd8d3c", "#D62728"]   # last = full method

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(x, df[metric], color=colors, edgecolor="white", linewidth=0.7)
        ax.set_title(metric, fontsize=9)
        ax.set_xticks(x)
        short_labels = [c.replace("Without ", "w/o ").replace(" (", "\n(")
                        for c in df["Configuration"]]
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Ablation study at $N=200$", fontsize=11, y=1.01)
    plt.tight_layout()
    out = f"{FIGS_DIR}/fig_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {out}")


# ────────────────────────────────────────────────────────────────────────────
# Fig 4 – Hardware stress
# ────────────────────────────────────────────────────────────────────────────

def plot_hw_stress():
    path = f"{RESULTS_DIR}/results_hw_stress.csv"
    if not os.path.exists(path):
        print(f"[plot] Skipping fig_hw_stress (missing {path})")
        return

    df = pd.read_csv(path)
    metrics = ["E_b (mJ/KB)", "T_c (ms)", "P_d (%)", "G_p (KB/s)"]
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    for ax, metric in zip(axes, metrics):
        ax.bar(x, df[metric], color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(metric, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Profile"], rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Hardware-aware stress results", fontsize=11, y=1.01)
    plt.tight_layout()
    out = f"{FIGS_DIR}/fig_hw_stress.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {out}")


# ────────────────────────────────────────────────────────────────────────────
# Fig 5 – Forwarder-selection sensitivity
# ────────────────────────────────────────────────────────────────────────────

def plot_pi_sensitivity():
    path = f"{RESULTS_DIR}/results_pi_sensitivity.csv"
    if not os.path.exists(path):
        print(f"[plot] Skipping fig_pi_sensitivity (missing {path})")
        return

    df = pd.read_csv(path)
    metrics = ["E_b (mJ/KB)", "P_d (%)", "G_p (KB/s)"]
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    colors = ["#9467bd", "#D62728", "#17becf"]

    for ax, metric in zip(axes, metrics):
        ax.bar(x, df[metric], color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(metric, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Profile"], rotation=18, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Forwarder-selection sensitivity at $N=200$", fontsize=11, y=1.01)
    plt.tight_layout()
    out = f"{FIGS_DIR}/fig_pi_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {out}")


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_main_comparison()
    plot_density()
    plot_ablation()
    plot_hw_stress()
    plot_pi_sensitivity()
    print("\n✓  All plots generated.")
