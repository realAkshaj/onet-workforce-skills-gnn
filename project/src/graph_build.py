"""Build a PyG HeteroData object for the occupation-skill bipartite graph."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from data_utils import load_maps

EDGES_PATH = Path("data/processed/onet_edges.csv")
GRAPH_PATH = Path("data/processed/hetero_graph.pt")


def build_hetero(edges_df: pd.DataFrame, maps: dict) -> HeteroData:
    occ2idx = maps["occ2idx"]
    skill2idx = maps["skill2idx"]
    n_occ, n_skill = len(occ2idx), len(skill2idx)

    src = edges_df["occ_code"].map(occ2idx).to_numpy()
    dst = edges_df["skill"].map(skill2idx).to_numpy()
    w = edges_df["weight"].to_numpy(dtype=np.float32)

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)

    data = HeteroData()

    # Placeholder features: identity (one-hot) for skills (small set),
    # and a degree-aware feature for occupations: row-wise weighted profile
    # over skills (n_skill dims). This gives each occupation a content vector
    # that still works for cold-start via GraphSAGE message passing.
    occ_feat = torch.zeros(n_occ, n_skill, dtype=torch.float32)
    occ_feat[src, dst] = edge_weight
    skill_feat = torch.eye(n_skill, dtype=torch.float32)

    data["occupation"].x = occ_feat
    data["occupation"].num_nodes = n_occ
    data["skill"].x = skill_feat
    data["skill"].num_nodes = n_skill

    data["occupation", "requires", "skill"].edge_index = edge_index
    data["occupation", "requires", "skill"].edge_weight = edge_weight

    rev = edge_index.flip(0)
    data["skill", "rev_requires", "occupation"].edge_index = rev
    data["skill", "rev_requires", "occupation"].edge_weight = edge_weight
    return data


def main() -> HeteroData:
    edges = pd.read_csv(EDGES_PATH)
    maps = load_maps()
    data = build_hetero(edges, maps)
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, GRAPH_PATH)
    print(f"[graph] {data}")
    print(f"[graph] saved -> {GRAPH_PATH}")
    return data


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parents[1])
    main()
