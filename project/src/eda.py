"""EDA plots for the O*NET bipartite graph."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EDGES_PATH = Path("data/processed/onet_edges.csv")
PLOTS_DIR = Path("outputs/plots")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(EDGES_PATH)

    # 1. Degree distribution per node type (unweighted).
    occ_deg = df.groupby("occ_code").size()
    sk_deg = df.groupby("skill").size()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(occ_deg.values, bins=30, color="#3B7DDD", edgecolor="black")
    axes[0].set_title("Occupation degree distribution")
    axes[0].set_xlabel("# skills per occupation")
    axes[0].set_ylabel("count")
    axes[1].hist(sk_deg.values, bins=30, color="#DD7F3B", edgecolor="black")
    axes[1].set_title("Skill degree distribution")
    axes[1].set_xlabel("# occupations per skill")
    axes[1].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "degree_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Top-20 skills by weighted degree.
    top = (
        df.groupby("skill")["weight"].sum().sort_values(ascending=False).head(20)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index[::-1], top.values[::-1], color="#4CA64C")
    ax.set_title("Top 20 skills by weighted degree")
    ax.set_xlabel("sum of edge weights")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "top20_skills.png", dpi=150)
    plt.close(fig)

    # 3. Edge-weight histogram.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["weight"].values, bins=40, color="#884EA0", edgecolor="black")
    ax.set_title("Edge-weight distribution")
    ax.set_xlabel("weight = (IM_norm + LV_norm) / 2")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "edge_weight_hist.png", dpi=150)
    plt.close(fig)

    print(f"[eda] plots -> {PLOTS_DIR}")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parents[1])
    main()
