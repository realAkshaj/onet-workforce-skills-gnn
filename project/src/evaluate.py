"""Link-prediction ranking metrics shared by baselines and GraphSAGE."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def _ranking_metrics(score_matrix: np.ndarray,
                     test_pairs_by_occ: Dict[int, np.ndarray],
                     train_mask: np.ndarray | None,
                     ks=(5, 10)) -> Dict[str, float]:
    """score_matrix: (n_occ, n_skill); higher = more likely positive.

    test_pairs_by_occ: occ_idx -> array of held-out skill indices.
    train_mask: boolean (n_occ, n_skill); True where training edges live
        (those are masked out of ranking to avoid trivially re-recommending).
    """
    n_occ, n_skill = score_matrix.shape
    if train_mask is not None:
        score_matrix = score_matrix.copy()
        score_matrix[train_mask] = -np.inf

    recall = {k: [] for k in ks}
    precision = {k: [] for k in ks}

    for occ, gold in test_pairs_by_occ.items():
        if len(gold) == 0:
            continue
        scores = score_matrix[occ]
        order = np.argsort(-scores)
        gold_set = set(gold.tolist())
        for k in ks:
            topk = order[:k]
            hits = sum(1 for s in topk if s in gold_set)
            recall[k].append(hits / len(gold_set))
            precision[k].append(hits / k)

    out = {}
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean(recall[k])) if recall[k] else 0.0
    out[f"Precision@{max(ks)}"] = float(np.mean(precision[max(ks)])) if precision[max(ks)] else 0.0
    return out


def auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    s = np.concatenate([pos_scores, neg_scores])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def evaluate_scores(score_matrix: np.ndarray,
                    test_pairs_by_occ: Dict[int, np.ndarray],
                    train_mask: np.ndarray | None,
                    n_skill: int,
                    seed: int = 0) -> Dict[str, float]:
    metrics = _ranking_metrics(score_matrix, test_pairs_by_occ, train_mask)

    # AUC over per-occupation positive edges vs one random negative each.
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    # Reload original scores for AUC (we masked train to -inf above for ranking;
    # for AUC use the raw score_matrix passed in).
    raw = score_matrix.copy()
    raw[~np.isfinite(raw)] = 0.0
    for occ, gold in test_pairs_by_occ.items():
        if len(gold) == 0:
            continue
        forbidden = set(gold.tolist())
        if train_mask is not None:
            forbidden |= set(np.where(train_mask[occ])[0].tolist())
        for g in gold:
            # sample a negative skill outside forbidden set
            for _ in range(20):
                cand = int(rng.integers(0, n_skill))
                if cand not in forbidden:
                    pos.append(raw[occ, g])
                    neg.append(raw[occ, cand])
                    break
    metrics["AUC"] = auc_from_scores(np.asarray(pos), np.asarray(neg))
    return metrics


def evaluate_occupation_similarity(occ_emb: np.ndarray, idx2occ: dict,
                                    ks=(5, 10)) -> Dict[str, float]:
    """Measure coherence of occupation embeddings against O*NET SOC family labels.

    For each occupation, computes the fraction of its top-k nearest neighbors
    (cosine similarity in embedding space) that share the same SOC major group
    (first 2 chars of the code, e.g. "11" from "11-1011.00").
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n_occ = len(idx2occ)
    families = [idx2occ[i][:2] for i in range(n_occ)]

    sim = cosine_similarity(occ_emb)  # (n_occ, n_occ)
    np.fill_diagonal(sim, -1.0)

    results = {}
    for k in ks:
        prec = []
        for i in range(n_occ):
            top_k = np.argsort(sim[i])[::-1][:k]
            same = sum(1 for j in top_k if families[j] == families[i])
            prec.append(same / k)
        results[f"FamilyP@{k}"] = float(np.mean(prec))
    return results


def print_table(title: str, results: Dict[str, Dict[str, float]]):
    print(f"\n=== {title} ===")
    cols = ["Recall@5", "Recall@10", "Precision@10", "AUC"]
    header = f"{'method':<28}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = f"{name:<28}"
        for c in cols:
            v = m.get(c, float('nan'))
            row += f"{v:>12.4f}"
        print(row)
