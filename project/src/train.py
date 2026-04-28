"""Training loop for BipartiteSAGE on occupation->skill link prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from model import BipartiteSAGE
from evaluate import evaluate_scores, test_pairs, train_mask

STD_PATH = Path("data/processed/split_standard.pkl")
COLD_PATH = Path("data/processed/split_coldstart.pkl")
CKPT_DIR = Path("checkpoints")


def _edge_tensors(df: pd.DataFrame, maps: dict):
    oi = torch.tensor(df["occ_code"].map(maps["occ2idx"]).to_numpy(), dtype=torch.long)
    si = torch.tensor(df["skill"].map(maps["skill2idx"]).to_numpy(), dtype=torch.long)
    return oi, si


def build_tfidf_features(train_df: pd.DataFrame, maps: dict,
                          task_texts: dict[str, str],
                          max_features: int = 500) -> torch.Tensor:
    """Fit TF-IDF on training occupation texts; transform all occupations.

    Returns a (n_occ, max_features) float tensor. Cold-start occupations get
    non-zero features derived from their O*NET task descriptions, giving the
    model occupation-specific signal even with no training edges.

    Critically, the vectorizer is fit on training occupation texts only so
    no cold-start information leaks into the feature space.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_occ = len(maps["occ2idx"])
    all_occ_codes = [maps["idx2occ"][i] for i in range(n_occ)]
    train_occ_set = set(train_df["occ_code"].unique())

    train_texts = [task_texts.get(c, "") for c in all_occ_codes if c in train_occ_set]
    vec = TfidfVectorizer(max_features=max_features, stop_words="english",
                          min_df=2, sublinear_tf=True)
    vec.fit(train_texts)

    all_texts = [task_texts.get(c, "") for c in all_occ_codes]
    matrix = vec.transform(all_texts).toarray().astype(np.float32)
    return torch.tensor(matrix)


def _build_graph_from_train(train_df: pd.DataFrame, maps: dict,
                             tfidf: torch.Tensor | None = None) -> HeteroData:
    n_occ = len(maps["occ2idx"])
    n_skill = len(maps["skill2idx"])
    oi, si = _edge_tensors(train_df, maps)
    w = torch.tensor(train_df["weight"].to_numpy(), dtype=torch.float32)

    data = HeteroData()
    occ_feat = torch.zeros(n_occ, n_skill)
    occ_feat[oi, si] = w
    # Concatenate TF-IDF features if provided; this gives cold-start occupations
    # a non-zero, occupation-specific input even when they have no training edges.
    data["occupation"].x = torch.cat([occ_feat, tfidf], dim=1) if tfidf is not None else occ_feat
    data["skill"].x = torch.eye(n_skill)
    data["occupation", "requires", "skill"].edge_index = torch.stack([oi, si])
    data["occupation", "requires", "skill"].edge_weight = w
    data["skill", "rev_requires", "occupation"].edge_index = torch.stack([si, oi])
    data["skill", "rev_requires", "occupation"].edge_weight = w
    return data


def sample_negatives(pos_occ: torch.Tensor, n_skill: int, device) -> torch.Tensor:
    return torch.randint(0, n_skill, (pos_occ.shape[0],), device=device)


def _edge_weight_dict(data) -> dict:
    return {
        ("occupation", "requires", "skill"):
            data["occupation", "requires", "skill"].edge_weight,
        ("skill", "rev_requires", "occupation"):
            data["skill", "rev_requires", "occupation"].edge_weight,
    }


def train_one(train_df, test_df, maps, *, epochs=100, lr=0.01, hidden=128,
              out_dim=64, device="cpu", label="standard", ckpt_name=None,
              task_texts=None, use_edge_weights=False):
    n_skill = len(maps["skill2idx"])
    tfidf = build_tfidf_features(train_df, maps, task_texts) if task_texts else None
    tfidf_dim = tfidf.shape[1] if tfidf is not None else 0
    data = _build_graph_from_train(train_df, maps, tfidf=tfidf).to(device)

    model = BipartiteSAGE(
        in_dims={"occupation": n_skill + tfidf_dim, "skill": n_skill},
        hidden_dim=hidden, out_dim=out_dim,
        use_edge_weights=use_edge_weights,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pos_occ, pos_skill = _edge_tensors(train_df, maps)
    pos_occ = pos_occ.to(device)
    pos_skill = pos_skill.to(device)

    test_pairs_dict = test_pairs(test_df, maps)
    train_mask_np = train_mask(train_df, maps)

    best_recall10 = -1.0
    best_metrics = None
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / (ckpt_name or f"sage_{label}.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        ew = _edge_weight_dict(data) if use_edge_weights else None
        h = model(data.x_dict, data.edge_index_dict, ew)

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
                h_eval = model(data.x_dict, data.edge_index_dict, ew)
                scores = model.score_all(h_eval["occupation"], h_eval["skill"]).cpu().numpy()
            metrics = evaluate_scores(scores, test_pairs_dict, train_mask_np, n_skill)
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
                    device: str = "cpu", task_texts=None, use_edge_weights=False):
    """Load a saved checkpoint and return (occ_emb, skill_emb) as numpy arrays."""
    n_skill = len(maps["skill2idx"])
    tfidf = build_tfidf_features(train_df, maps, task_texts) if task_texts else None
    tfidf_dim = tfidf.shape[1] if tfidf is not None else 0
    data = _build_graph_from_train(train_df, maps, tfidf=tfidf).to(device)

    model = BipartiteSAGE(
        in_dims={"occupation": n_skill + tfidf_dim, "skill": n_skill},
        hidden_dim=128, out_dim=64,
        use_edge_weights=use_edge_weights,
    ).to(device)
    ckpt_path = CKPT_DIR / ckpt_name
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    ew = _edge_weight_dict(data) if use_edge_weights else None
    with torch.no_grad():
        h = model(data.x_dict, data.edge_index_dict, ew)

    return h["occupation"].cpu().numpy(), h["skill"].cpu().numpy()
