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


def plot_occ_embeddings(occ_emb: np.ndarray, idx2occ: dict, out_path: Path):
    """t-SNE of occupation embeddings colored by SOC major group."""
    from sklearn.manifold import TSNE

    n_occ = len(idx2occ)
    families = [idx2occ[i][:2] for i in range(n_occ)]
    unique_fams = sorted(set(families))
    fam2int = {f: i for i, f in enumerate(unique_fams)}
    color_ids = np.array([fam2int[f] for f in families])

    print("[viz] running t-SNE on occupation embeddings...")
    emb_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(occ_emb)

    cmap = plt.get_cmap("tab20", len(unique_fams))
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
               c=color_ids, cmap=cmap, vmin=-0.5, vmax=len(unique_fams) - 0.5,
               alpha=0.65, s=25, linewidths=0)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(fam2int[f]), markersize=7, label=f)
        for f in unique_fams
    ]
    ax.legend(handles=handles, title="SOC major group",
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    ax.set_title("t-SNE of Learned Occupation Embeddings (GraphSAGE)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved -> {out_path}")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parents[1])
    main()
