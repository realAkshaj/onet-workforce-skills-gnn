"""Create standard and cold-start train/test edge splits."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

EDGES_PATH = Path("data/processed/onet_edges.csv")
STD_PATH = Path("data/processed/split_standard.pkl")
COLD_PATH = Path("data/processed/split_coldstart.pkl")


def standard_split(edges: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Hold out `test_frac` of edges per occupation at random."""
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for occ, grp in edges.groupby("occ_code", sort=False):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        # Guarantee at least one training edge per occupation.
        if n_test >= len(idx):
            n_test = len(idx) - 1
        test_rows.append(idx[:n_test])
        train_rows.append(idx[n_test:])
    test_idx = np.concatenate(test_rows)
    train_idx = np.concatenate(train_rows)
    return edges.loc[train_idx].reset_index(drop=True), edges.loc[test_idx].reset_index(drop=True)


def coldstart_split(edges: pd.DataFrame, occ_test_frac: float = 0.1, seed: int = 42):
    """Hold out whole occupations from training."""
    rng = np.random.default_rng(seed)
    occs = edges["occ_code"].unique()
    rng.shuffle(occs)
    n_hold = max(1, int(round(len(occs) * occ_test_frac)))
    test_occs = set(occs[:n_hold].tolist())
    mask_test = edges["occ_code"].isin(test_occs)
    return (
        edges[~mask_test].reset_index(drop=True),
        edges[mask_test].reset_index(drop=True),
        test_occs,
    )


def main():
    edges = pd.read_csv(EDGES_PATH)

    tr, te = standard_split(edges)
    with open(STD_PATH, "wb") as f:
        pickle.dump({"train": tr, "test": te}, f)
    print(f"[split] standard: train={len(tr):,} test={len(te):,} -> {STD_PATH}")

    tr_c, te_c, cold_occs = coldstart_split(edges)
    with open(COLD_PATH, "wb") as f:
        pickle.dump({"train": tr_c, "test": te_c, "cold_occs": cold_occs}, f)
    print(
        f"[split] cold-start: train={len(tr_c):,} test={len(te_c):,} "
        f"held_occs={len(cold_occs)} -> {COLD_PATH}"
    )


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parents[1])
    main()
