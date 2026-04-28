"""End-to-end runner for the O*NET workforce skill recommendation project."""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from data_utils import (download_all_dimensions, download_tasks, load_and_clean,
                        load_task_descriptions, save_edges, build_node_maps)
from graph_build import main as build_graph
from splits import standard_split, coldstart_split, STD_PATH, COLD_PATH
from baselines import run as run_baselines
from train import train_one, load_embeddings
from evaluate import print_table, evaluate_occupation_similarity
from eda import plot_occ_embeddings
import eda

RESULTS_PATH = Path("outputs/results.json")
WEIGHT_THRESHOLD = 0.5

# Ablation variants on the standard split.
# Each row: (label, ckpt_name, use_tfidf, use_edge_weights)
ABLATION_VARIANTS = [
    ("Base SAGE",          "sage_ablation_base.pt",     False, False),
    ("+ TF-IDF",           "sage_standard.pt",          True,  False),
    ("+ Weighted MP",      "sage_ablation_weighted.pt", False, True),
    ("Full (TF-IDF + WMP)","sage_full.pt",              True,  True),
]


def ensure_data():
    download_all_dimensions()
    download_tasks()
    edges = load_and_clean()
    before = len(edges)
    edges = edges[edges["weight"] >= WEIGHT_THRESHOLD].reset_index(drop=True)
    print(f"[data] threshold >={WEIGHT_THRESHOLD}: {before:,} -> {len(edges):,} edges")
    save_edges(edges)
    maps = build_node_maps(edges)
    build_graph()

    tr, te = standard_split(edges)
    with open(STD_PATH, "wb") as f:
        pickle.dump({"train": tr, "test": te}, f)
    tr_c, te_c, cold_occs = coldstart_split(edges)
    with open(COLD_PATH, "wb") as f:
        pickle.dump({"train": tr_c, "test": te_c, "cold_occs": cold_occs}, f)
    return edges, maps


def main():
    edges, maps = ensure_data()
    eda.main()

    task_texts = load_task_descriptions()

    baseline_results = run_baselines()

    with open(STD_PATH, "rb") as f:
        std = pickle.load(f)
    with open(COLD_PATH, "rb") as f:
        cold = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[train] device = {device}")

    # --- Ablation: 4 variants on standard split ---
    ablation_results = {}
    for label, ckpt, use_tfidf, use_ew in ABLATION_VARIANTS:
        print(f"\n[ablation] {label}  tfidf={use_tfidf}  weighted={use_ew}")
        res = train_one(
            std["train"], std["test"], maps,
            epochs=100, lr=0.01, device=device,
            label=label, ckpt_name=ckpt,
            task_texts=task_texts if use_tfidf else None,
            use_edge_weights=use_ew,
        )
        ablation_results[label] = res

    # --- Cold-start: full model (TF-IDF + weighted MP) ---
    sage_cold = train_one(
        cold["train"], cold["test"], maps,
        epochs=100, lr=0.01, device=device,
        label="cold-start (full)", ckpt_name="sage_coldstart_full.pt",
        task_texts=task_texts,
        use_edge_weights=True,
    )

    # Also keep the TF-IDF-only cold-start for comparison
    sage_cold_base = train_one(
        cold["train"], cold["test"], maps,
        epochs=100, lr=0.01, device=device,
        label="cold-start (tfidf)", ckpt_name="sage_coldstart.pt",
        task_texts=task_texts,
        use_edge_weights=False,
    )

    # --- Task 2: occupation similarity ---
    occ_emb, _ = load_embeddings(
        "sage_full.pt", std["train"], maps, device,
        task_texts=task_texts, use_edge_weights=True,
    )
    sim_metrics = evaluate_occupation_similarity(occ_emb, maps["idx2occ"])
    print_table("Task 2 - Occupation similarity (SOC family coherence)",
                {"GraphSAGE (full)": sim_metrics})
    plot_occ_embeddings(
        occ_emb, maps["idx2occ"],
        Path("outputs/plots/occ_embeddings_tsne.png"),
    )

    # --- Final comparison table ---
    task1_final = {
        "Cosine (standard)":         baseline_results["cosine"],
        "Jaccard (standard)":        baseline_results["jaccard"],
        "Node2Vec (standard)":       baseline_results["node2vec"],
        **{f"SAGE {k} (std)": v for k, v in ablation_results.items()},
        "GraphSAGE cold (TF-IDF)":   sage_cold_base,
        "GraphSAGE cold (full)":     sage_cold,
    }
    print_table("Final comparison - Task 1 (skill recommendation)", task1_final)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {
                "note": "Cosine/Jaccard/Node2Vec cannot score unseen occupations.",
                "task1_baselines": {
                    "Cosine (standard)":   baseline_results["cosine"],
                    "Jaccard (standard)":  baseline_results["jaccard"],
                    "Node2Vec (standard)": baseline_results["node2vec"],
                },
                "task1_ablation_standard": ablation_results,
                "task1_coldstart": {
                    "GraphSAGE (TF-IDF only)": sage_cold_base,
                    "GraphSAGE (full)":         sage_cold,
                },
                "task2_occupation_similarity": {"GraphSAGE (full)": sim_metrics},
            },
            f,
            indent=2,
        )
    print(f"\n[results] saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
