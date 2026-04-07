"""Cosine, Jaccard, and Node2Vec baselines for occupation->skill link prediction."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from evaluate import evaluate_scores, print_table

EDGES_PATH = Path("data/processed/onet_edges.csv")
NODE_MAPS_PATH = Path("data/processed/node_maps.pkl")
STD_PATH = Path("data/processed/split_standard.pkl")


def _load_maps():
    with open(NODE_MAPS_PATH, "rb") as f:
        return pickle.load(f)


def _to_matrix(df: pd.DataFrame, maps: dict, binary: bool = False) -> np.ndarray:
    n_occ = len(maps["occ2idx"])
    n_skill = len(maps["skill2idx"])
    M = np.zeros((n_occ, n_skill), dtype=np.float32)
    oi = df["occ_code"].map(maps["occ2idx"]).to_numpy()
    si = df["skill"].map(maps["skill2idx"]).to_numpy()
    w = df["weight"].to_numpy(dtype=np.float32)
    if binary:
        M[oi, si] = 1.0
    else:
        M[oi, si] = w
    return M


def _test_pairs(df: pd.DataFrame, maps: dict) -> Dict[int, np.ndarray]:
    oi = df["occ_code"].map(maps["occ2idx"]).to_numpy()
    si = df["skill"].map(maps["skill2idx"]).to_numpy()
    out: Dict[int, list] = {}
    for o, s in zip(oi, si):
        out.setdefault(int(o), []).append(int(s))
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def cosine_baseline(train: pd.DataFrame, test: pd.DataFrame, maps: dict):
    """Represent each occupation as its weighted skill vector; score pairs by
    cosine similarity between the occ vector and each skill's one-hot column.
    Because skills are one-hot, cosine simplifies to
        sim(o, s) = w_{o,s} / ||w_o||.
    That already yields a useful per-occupation ranking, but is not very
    discriminative across skills for held-out edges (all zeros). So we instead
    measure similarity between occupations via their train vectors and predict
    skills using a neighbor-weighted average — the classic user-based CF form
    of a "cosine" baseline.
    """
    M = _to_matrix(train, maps)  # (n_occ, n_skill)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    Mn = M / norms
    sim = Mn @ Mn.T  # (n_occ, n_occ)
    np.fill_diagonal(sim, 0.0)
    scores = sim @ M  # (n_occ, n_skill)
    return scores


def jaccard_baseline(train: pd.DataFrame, test: pd.DataFrame, maps: dict):
    B = _to_matrix(train, maps, binary=True)  # (n_occ, n_skill)
    inter = B @ B.T
    row_sums = B.sum(axis=1, keepdims=True)
    union = row_sums + row_sums.T - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    np.fill_diagonal(sim, 0.0)
    scores = sim @ B
    return scores


def node2vec_baseline(train: pd.DataFrame, test: pd.DataFrame, maps: dict,
                       device: str = "cpu", epochs: int = 50) -> np.ndarray:
    """Node2Vec transductive baseline on a homogeneous bipartite projection.

    Skill node indices are offset by n_occ so all nodes share one embedding
    space. After training, the score matrix is occ_emb @ skill_emb.T.
    Node2Vec is transductive and cannot score cold-start occupations.
    """
    from torch_geometric.nn import Node2Vec

    n_occ = len(maps["occ2idx"])
    n_skill = len(maps["skill2idx"])
    n_total = n_occ + n_skill

    occ_idx = torch.tensor(
        train["occ_code"].map(maps["occ2idx"]).to_numpy(), dtype=torch.long
    )
    skill_idx = torch.tensor(
        train["skill"].map(maps["skill2idx"]).to_numpy(), dtype=torch.long
    ) + n_occ  # offset so skills live in a separate range

    src = torch.cat([occ_idx, skill_idx])
    dst = torch.cat([skill_idx, occ_idx])
    edge_index = torch.stack([src, dst], dim=0)

    model = Node2Vec(
        edge_index,
        embedding_dim=64,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        num_nodes=n_total,
        sparse=True,
    ).to(device)

    loader = model.loader(batch_size=256, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  [node2vec] epoch {epoch}/{epochs}  loss={total_loss / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        emb = model()  # (n_total, 64)
    occ_emb = emb[:n_occ].cpu().numpy()
    skill_emb = emb[n_occ:].cpu().numpy()
    return (occ_emb @ skill_emb.T).astype(np.float32)


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    maps = _load_maps()
    with open(STD_PATH, "rb") as f:
        split = pickle.load(f)
    train, test = split["train"], split["test"]

    n_skill = len(maps["skill2idx"])
    train_mask = _to_matrix(train, maps, binary=True).astype(bool)
    test_pairs = _test_pairs(test, maps)

    results = {}
    for name, fn in [
        ("cosine", cosine_baseline),
        ("jaccard", jaccard_baseline),
        ("node2vec", lambda tr, te, m: node2vec_baseline(tr, te, m, device=device)),
    ]:
        print(f"\n[baselines] running {name}...")
        scores = fn(train, test, maps)
        results[name] = evaluate_scores(scores, test_pairs, train_mask, n_skill)

    print_table("Baselines (standard split)", results)
    return results


if __name__ == "__main__":
    import os, sys
    os.chdir(Path(__file__).resolve().parents[1])
    sys.path.insert(0, "src")
    run()
