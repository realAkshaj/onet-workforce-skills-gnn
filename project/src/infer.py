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
import pandas as pd
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

    # Precompute full score matrix for existing occupations (used by Skill Explorer)
    score_matrix = occ_emb @ skill_emb.T      # (n_occ, n_skill)

    # Mask training edges so we don't re-recommend known skills for existing occs
    train_oi = train_df["occ_code"].map(maps["occ2idx"]).to_numpy()
    train_si = train_df["skill"].map(maps["skill2idx"]).to_numpy()
    train_mask = np.zeros((n_occ, n_skill), dtype=bool)
    train_mask[train_oi, train_si] = True

    # Build actual O*NET edge-weight matrix for Career Advisor (skills -> roles).
    # Using real edge weights gives semantically correct reverse lookup:
    # "which occupations genuinely require this skill at high importance/level?"
    edges_df = pd.read_csv(DATA_DIR / "onet_edges.csv")
    edge_weight_matrix = np.zeros((n_occ, n_skill), dtype=np.float32)
    oi = edges_df["occ_code"].map(maps["occ2idx"]).to_numpy()
    si = edges_df["skill"].map(maps["skill2idx"]).to_numpy()
    w  = edges_df["weight"].to_numpy(dtype=np.float32)
    valid = ~(np.isnan(oi) | np.isnan(si))
    oi, si, w = oi[valid].astype(int), si[valid].astype(int), w[valid]
    edge_weight_matrix[oi, si] = w

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
        edge_weight_matrix=edge_weight_matrix,
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
        cold_text = text

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

    return results, is_cold, (cold_text if is_cold else None)


def predict_skills_with_update(occ_idx: int, new_skill_names: list[str],
                                top_k: int, arts: dict) -> tuple[list[dict], float]:
    """Inject new skills into an existing occupation and re-run inference.

    Adds edges to the graph and updates the occupation's feature vector, then
    runs a single forward pass. Model parameters are not changed — this
    demonstrates inductive updating without retraining.

    Returns (updated_skill_list, elapsed_ms).
    """
    import time
    maps = arts["maps"]
    data = arts["data"]
    model = arts["model"]
    device = arts["device"]

    new_idxs = [maps["skill2idx"][s] for s in new_skill_names
                if s in maps["skill2idx"]]
    if not new_idxs:
        return [], 0.0

    # Update occupation feature vector — add injected skills with weight 1.0
    occ_feat = data["occupation"].x.clone()
    for si in new_idxs:
        occ_feat[occ_idx, si] = 1.0

    # Add new edges in both directions
    new_oi = torch.tensor([occ_idx] * len(new_idxs), dtype=torch.long, device=device)
    new_si = torch.tensor(new_idxs, dtype=torch.long, device=device)
    new_ew = torch.ones(len(new_idxs), dtype=torch.float32, device=device)

    ei_fwd = data["occupation", "requires", "skill"].edge_index
    ew_fwd = data["occupation", "requires", "skill"].edge_weight
    ei_rev = data["skill", "rev_requires", "occupation"].edge_index
    ew_rev = data["skill", "rev_requires", "occupation"].edge_weight

    edge_index_dict = {
        ("occupation", "requires", "skill"):
            torch.cat([ei_fwd, torch.stack([new_oi, new_si])], dim=1),
        ("skill", "rev_requires", "occupation"):
            torch.cat([ei_rev, torch.stack([new_si, new_oi])], dim=1),
    }
    ew_dict = None
    if arts["ew"] is not None:
        ew_dict = {
            ("occupation", "requires", "skill"):  torch.cat([ew_fwd, new_ew]),
            ("skill", "rev_requires", "occupation"): torch.cat([ew_rev, new_ew]),
        }

    x_dict = {"occupation": occ_feat, "skill": data["skill"].x}

    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        h = model(x_dict, edge_index_dict, ew_dict)
        scores = (h["occupation"][occ_idx] @ h["skill"].T).cpu().numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Mask known training skills AND newly injected skills from results
    mask = arts["train_mask"][occ_idx].copy()
    for si in new_idxs:
        mask[si] = True
    scores[mask] = -np.inf

    order = np.argsort(-scores)
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
    return results, elapsed_ms


def find_similar_roles(query_text: str, top_k: int, arts: dict) -> list[dict]:
    """Find existing O*NET occupations most similar to a new role description.

    Uses TF-IDF cosine similarity between the query text and existing occupation
    task descriptions. Works best when the new role's vocabulary overlaps with
    O*NET (e.g. policy, compliance, management roles).

    Returns list of {title, code, similarity} dicts.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf_vec: TfidfVectorizer = arts["tfidf_vec"]
    maps = arts["maps"]
    occ_title_map = maps.get("occ_title", {})
    task_texts = arts["task_texts"]
    n_occ = arts["n_occ"]

    query_vec = tfidf_vec.transform([query_text])
    all_texts = [task_texts.get(maps["idx2occ"][i], "") for i in range(n_occ)]
    corpus_vecs = tfidf_vec.transform(all_texts)
    sims = cosine_similarity(query_vec, corpus_vecs).squeeze(0)

    order = np.argsort(-sims)[:top_k]
    results = []
    for idx in order:
        code = maps["idx2occ"][idx]
        title = occ_title_map.get(code, code)
        results.append({"title": title, "code": code, "similarity": float(sims[idx])})
    return results


def recommend_roles(selected_skill_names: list[str], top_k: int,
                    arts: dict) -> list[dict]:
    """Recommend occupations most aligned with a set of skill names.

    Sums the model's predicted link scores for each selected skill across all
    occupations, then ranks. This directly uses what the model was trained to
    predict, so the results are semantically grounded.

    Returns list of {code, title, score, pct} dicts where pct is 0-100.
    """
    maps = arts["maps"]
    # Use actual O*NET edge weights — semantically correct for reverse lookup:
    # rank occupations by how strongly they require the selected skills.
    ewm = arts["edge_weight_matrix"]  # (n_occ, n_skill), real edge weights

    idxs = [maps["skill2idx"][s] for s in selected_skill_names
            if s in maps["skill2idx"]]
    if not idxs:
        return []

    # Mean edge weight across selected skills; zero means the skill is absent.
    occ_scores = ewm[:, idxs].mean(axis=1)  # (n_occ,)

    order = np.argsort(-occ_scores)[:top_k]
    occ_title_map = maps.get("occ_title", {})

    top_scores = occ_scores[order]
    max_s = top_scores.max() if top_scores.max() > top_scores.min() else 1.0
    min_s = top_scores.min()

    results = []
    for rank_i, idx in enumerate(order):
        code = maps["idx2occ"][idx]
        title = occ_title_map.get(code, code)
        s = float(occ_scores[idx])
        pct = int(100 * (s - min_s) / max(max_s - min_s, 1e-8))
        results.append({"code": code, "title": title,
                        "score": s, "pct": pct, "occ_idx": int(idx)})
    return results
