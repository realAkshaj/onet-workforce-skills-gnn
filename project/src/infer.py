"""Inference utilities for the Streamlit dashboard.

Loads trained artifacts once and exposes two functions:
  predict_skills  -- occupation (existing or new) -> ranked skill list
  recommend_roles -- selected skills -> ranked occupation list
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import BipartiteSAGE
from train import _build_graph_from_train, build_tfidf_features

CKPT_DIR = ROOT / "checkpoints"
DATA_DIR = ROOT / "data" / "processed"

# Dimension color mapping for the dashboard
DIM_COLORS = {
    "Skills":         "#4ecdc4",
    "Abilities":      "#ff6b6b",
    "WorkActivities": "#ffd93d",
    "Knowledge":      "#6bcb77",
}
DIM_LABELS = {
    "Skills": "Skills",
    "Abilities": "Abilities",
    "WorkActivities": "Work Activities",
    "Knowledge": "Knowledge",
}


def _skill_dimension(skill_name: str) -> str:
    """Extract the O*NET dimension prefix from a skill name like 'Skills/Active Listening'."""
    if "/" in skill_name:
        return skill_name.split("/")[0]
    return "Skills"


def _skill_display(skill_name: str) -> str:
    """Strip the dimension prefix for display: 'Skills/Active Listening' -> 'Active Listening'."""
    if "/" in skill_name:
        return skill_name.split("/", 1)[1]
    return skill_name


def load_artifacts(device: str = "cpu") -> dict:
    """Load and return all artifacts needed for inference (call once, cache the result).

    Returns a dict with:
      maps, train_df, task_texts, tfidf_vec,
      model, occ_emb (n_occ x 64), skill_emb (n_skill x 64),
      occ_titles (list of str), skill_names (list of str),
      cosine_score_matrix (n_occ x n_skill)
    """
    maps = pickle.load(open(DATA_DIR / "node_maps.pkl", "rb"))

    # Use the standard split training data for feature construction
    std = pickle.load(open(DATA_DIR / "split_standard.pkl", "rb"))
    train_df = std["train"]

    # Load O*NET task descriptions for TF-IDF
    from data_utils import load_task_descriptions
    task_texts = load_task_descriptions()

    # Build TF-IDF vectorizer (same as training — fitted on training occs only)
    tfidf_tensor = build_tfidf_features(train_df, maps, task_texts, max_features=500)
    n_occ = len(maps["occ2idx"])
    n_skill = len(maps["skill2idx"])

    # Refit a sklearn vectorizer so we can transform new occupation descriptions
    train_occ_codes = set(train_df["occ_code"].unique())
    all_codes = [maps["idx2occ"][i] for i in range(n_occ)]
    train_texts = [task_texts.get(c, "") for c in all_codes if c in train_occ_codes]
    tfidf_vec = TfidfVectorizer(max_features=500, stop_words="english",
                                min_df=2, sublinear_tf=True)
    tfidf_vec.fit(train_texts)

    # Load the best model: full (TF-IDF + weighted MP) if it exists, else standard
    ckpt = CKPT_DIR / "sage_full.pt"
    use_ew = True
    if not ckpt.exists():
        ckpt = CKPT_DIR / "sage_standard.pt"
        use_ew = False

    data = _build_graph_from_train(train_df, maps, tfidf=tfidf_tensor).to(device)
    model = BipartiteSAGE(
        in_dims={"occupation": n_skill + 500, "skill": n_skill},
        hidden_dim=128, out_dim=64,
        use_edge_weights=use_ew,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    ew = None
    if use_ew:
        ew = {
            ("occupation", "requires", "skill"):
                data["occupation", "requires", "skill"].edge_weight,
            ("skill", "rev_requires", "occupation"):
                data["skill", "rev_requires", "occupation"].edge_weight,
        }

    with torch.no_grad():
        h = model(data.x_dict, data.edge_index_dict, ew)

    occ_emb = h["occupation"].cpu().numpy()   # (n_occ, 64)
    skill_emb = h["skill"].cpu().numpy()      # (n_skill, 64)

    # Precompute full score matrix for existing occupations
    score_matrix = occ_emb @ skill_emb.T      # (n_occ, n_skill)

    # Mask training edges so we don't re-recommend known skills for existing occs
    train_oi = train_df["occ_code"].map(maps["occ2idx"]).to_numpy()
    train_si = train_df["skill"].map(maps["skill2idx"]).to_numpy()
    train_mask = np.zeros((n_occ, n_skill), dtype=bool)
    train_mask[train_oi, train_si] = True

    occ_titles = [maps["idx2occ"].get(i, "") for i in range(n_occ)]
    skill_names = [maps["idx2skill"].get(i, "") for i in range(n_skill)]

    # Build title -> index lookup (case-insensitive)
    title2idx = {maps["occ_title"].get(maps["idx2occ"][i], "").lower(): i
                 for i in range(n_occ)}

    return dict(
        maps=maps, train_df=train_df, task_texts=task_texts,
        tfidf_vec=tfidf_vec, tfidf_tensor=tfidf_tensor,
        model=model, device=device, data=data, ew=ew,
        occ_emb=occ_emb, skill_emb=skill_emb,
        score_matrix=score_matrix, train_mask=train_mask,
        occ_titles=occ_titles, skill_names=skill_names,
        title2idx=title2idx, n_occ=n_occ, n_skill=n_skill,
    )


def predict_skills(occ_input: str, description: str, top_k: int,
                   arts: dict, existing_occ_idx: int | None = None
                   ) -> tuple[list[dict], bool]:
    """Return top_k predicted skills for an occupation.

    If existing_occ_idx is given, uses the precomputed score matrix.
    Otherwise runs cold-start inference using TF-IDF on description.

    Returns (skill_list, is_cold_start) where skill_list is a list of dicts:
      {name, display_name, dimension, color, score}
    """
    arts_sm = arts["score_matrix"]
    arts_tm = arts["train_mask"]

    if existing_occ_idx is not None:
        # Existing occupation: mask out training edges, rank the rest
        scores = arts_sm[existing_occ_idx].copy()
        scores[arts_tm[existing_occ_idx]] = -np.inf
        is_cold = False
    else:
        # Cold-start: add the new occupation as an extra node with TF-IDF features
        # and no edges, then run a full forward pass so message-passing behaves
        # identically to how the model was trained on cold-start occupations.
        tfidf_vec: TfidfVectorizer = arts["tfidf_vec"]
        device = arts["device"]
        n_skill = arts["n_skill"]
        n_occ = arts["n_occ"]

        text = f"{occ_input} {description}" if description else occ_input
        tfidf_row = torch.tensor(
            tfidf_vec.transform([text]).toarray().astype(np.float32), device=device
        )
        new_occ_feat = torch.cat([
            torch.zeros(1, n_skill, device=device), tfidf_row
        ], dim=1)  # (1, n_skill+500)

        # Extend occupation features; new node sits at index n_occ with no edges
        data = arts["data"]
        occ_feat_ext = torch.cat([data["occupation"].x, new_occ_feat], dim=0)
        x_dict = {"occupation": occ_feat_ext, "skill": data["skill"].x}

        arts["model"].eval()
        with torch.no_grad():
            h = arts["model"](x_dict, data.edge_index_dict, arts["ew"])

        scores = (h["occupation"][n_occ] @ h["skill"].T).cpu().numpy()
        is_cold = True

    order = np.argsort(-scores)[:top_k * 3]  # over-fetch, filter -inf
    results = []
    for idx in order:
        if len(results) >= top_k:
            break
        if not np.isfinite(scores[idx]):
            continue
        name = arts["skill_names"][idx]
        dim = _skill_dimension(name)
        results.append({
            "name": name,
            "display_name": _skill_display(name),
            "dimension": dim,
            "dim_label": DIM_LABELS.get(dim, dim),
            "color": DIM_COLORS.get(dim, "#aaaaaa"),
            "score": float(scores[idx]),
        })
    return results, is_cold


def recommend_roles(selected_skill_names: list[str], top_k: int,
                    arts: dict) -> list[dict]:
    """Recommend occupations most aligned with a set of skill names.

    Computes the average skill embedding for selected skills, then ranks
    occupations by dot-product with that query vector.

    Returns list of {code, title, score} dicts.
    """
    maps = arts["maps"]
    skill_emb = arts["skill_emb"]
    occ_emb = arts["occ_emb"]

    idxs = [maps["skill2idx"][s] for s in selected_skill_names
            if s in maps["skill2idx"]]
    if not idxs:
        return []

    query = skill_emb[idxs].mean(axis=0, keepdims=True)  # (1, 64)
    scores = (occ_emb @ query.T).squeeze(1)               # (n_occ,)

    order = np.argsort(-scores)[:top_k]
    occ_title_map = maps.get("occ_title", {})
    results = []
    for idx in order:
        code = maps["idx2occ"][idx]
        title = occ_title_map.get(code, code)
        results.append({"code": code, "title": title,
                        "score": float(scores[idx]), "occ_idx": int(idx)})
    return results
