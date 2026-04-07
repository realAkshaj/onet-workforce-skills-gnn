"""Training loop for BipartiteSAGE on occupation->skill link prediction."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from model import BipartiteSAGE
from evaluate import evaluate_scores

NODE_MAPS_PATH = Path("data/processed/node_maps.pkl")
EDGES_PATH = Path("data/processed/onet_edges.csv")
STD_PATH = Path("data/processed/split_standard.pkl")
COLD_PATH = Path("data/processed/split_coldstart.pkl")
CKPT_DIR = Path("checkpoints")


def _load_maps():
    with open(NODE_MAPS_PATH, "rb") as f:
        return pickle.load(f)


def _edge_tensors(df: pd.DataFrame, maps: dict):
    oi = torch.tensor(df["occ_code"].map(maps["occ2idx"]).to_numpy(), dtype=torch.long)
    si = torch.tensor(df["skill"].map(maps["skill2idx"]).to_numpy(), dtype=torch.long)
    return oi, si


def _build_graph_from_train(train_df: pd.DataFrame, maps: dict) -> HeteroData:
    n_occ = len(maps["occ2idx"])
    n_skill = len(maps["skill2idx"])
    oi, si = _edge_tensors(train_df, maps)
    w = torch.tensor(train_df["weight"].to_numpy(), dtype=torch.float32)

    data = HeteroData()
    occ_feat = torch.zeros(n_occ, n_skill)
    occ_feat[oi, si] = w
    data["occupation"].x = occ_feat
    data["skill"].x = torch.eye(n_skill)
    data["occupation", "requires", "skill"].edge_index = torch.stack([oi, si])
    data["skill", "rev_requires", "occupation"].edge_index = torch.stack([si, oi])
    return data


def _test_pairs(df: pd.DataFrame, maps: dict) -> Dict[int, np.ndarray]:
    oi = df["occ_code"].map(maps["occ2idx"]).to_numpy()
    si = df["skill"].map(maps["skill2idx"]).to_numpy()
    out: Dict[int, list] = {}
    for o, s in zip(oi, si):
        out.setdefault(int(o), []).append(int(s))
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def _train_mask(df: pd.DataFrame, maps: dict) -> np.ndarray:
    n_occ = len(maps["occ2idx"])
    n_skill = len(maps["skill2idx"])
    M = np.zeros((n_occ, n_skill), dtype=bool)
    oi = df["occ_code"].map(maps["occ2idx"]).to_numpy()
    si = df["skill"].map(maps["skill2idx"]).to_numpy()
    M[oi, si] = True
    return M


def sample_negatives(pos_occ: torch.Tensor, n_skill: int, device) -> torch.Tensor:
    return torch.randint(0, n_skill, (pos_occ.shape[0],), device=device)


def train_one(train_df, test_df, maps, *, epochs=100, lr=0.01, hidden=128,
              out_dim=64, device="cpu", label="standard", ckpt_name=None):
    n_skill = len(maps["skill2idx"])
    data = _build_graph_from_train(train_df, maps).to(device)

    model = BipartiteSAGE(
        in_dims={"occupation": n_skill, "skill": n_skill},
        hidden_dim=hidden, out_dim=out_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pos_occ, pos_skill = _edge_tensors(train_df, maps)
    pos_occ = pos_occ.to(device)
    pos_skill = pos_skill.to(device)

    test_pairs = _test_pairs(test_df, maps)
    train_mask_np = _train_mask(train_df, maps)

    best_recall10 = -1.0
    best_metrics = None
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / (ckpt_name or f"sage_{label}.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        h = model(data.x_dict, data.edge_index_dict)

        pos_logit = model.decode(h["occupation"], h["skill"], pos_occ, pos_skill)
        neg_skill = sample_negatives(pos_occ, n_skill, device)
        neg_logit = model.decode(h["occupation"], h["skill"], pos_occ, neg_skill)

        logits = torch.cat([pos_logit, neg_logit])
        labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                h_eval = model(data.x_dict, data.edge_index_dict)
                scores = model.score_all(h_eval["occupation"], h_eval["skill"]).cpu().numpy()
            metrics = evaluate_scores(scores, test_pairs, train_mask_np, n_skill)
            print(
                f"[{label}] epoch {epoch:3d}  loss={loss.item():.4f}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )
            if metrics["Recall@10"] > best_recall10:
                best_recall10 = metrics["Recall@10"]
                best_metrics = metrics
                torch.save(model.state_dict(), ckpt_path)

    print(f"[{label}] best checkpoint -> {ckpt_path}")
    return best_metrics


def load_embeddings(ckpt_name: str, train_df: pd.DataFrame, maps: dict,
                    device: str = "cpu"):
    """Load a saved checkpoint and return (occ_emb, skill_emb) as numpy arrays.

    Rebuilds the training graph (no test leakage) then runs a single forward
    pass through the saved model to produce node embeddings.
    """
    n_skill = len(maps["skill2idx"])
    data = _build_graph_from_train(train_df, maps).to(device)

    model = BipartiteSAGE(
        in_dims={"occupation": n_skill, "skill": n_skill},
        hidden_dim=128, out_dim=64,
    ).to(device)
    ckpt_path = CKPT_DIR / ckpt_name
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        h = model(data.x_dict, data.edge_index_dict)

    return h["occupation"].cpu().numpy(), h["skill"].cpu().numpy()
