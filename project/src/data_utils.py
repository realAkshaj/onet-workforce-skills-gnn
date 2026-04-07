"""Download O*NET Skills dataset and build the occupation-skill edge table."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ONET_URL = "https://www.onetcenter.org/dl_files/database/db_30_1_excel/Skills.xlsx"
RAW_PATH = Path("data/raw/Skills.xlsx")
EDGES_PATH = Path("data/processed/onet_edges.csv")
NODE_MAPS_PATH = Path("data/processed/node_maps.pkl")


def download_skills(url: str = ONET_URL, dest: Path = RAW_PATH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[data] {dest} already present ({dest.stat().st_size/1024:.1f} KB)")
        return dest
    print(f"[data] downloading {url}")
    headers = {"User-Agent": "Mozilla/5.0 (cs519-project)"}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"[data] saved -> {dest} ({len(r.content)/1024:.1f} KB)")
    return dest


def load_and_clean(xlsx_path: Path = RAW_PATH) -> pd.DataFrame:
    """Return a dataframe with one row per (occupation, skill) containing IM, LV, weight."""
    df = pd.read_excel(xlsx_path)
    # Normalize column names: the O*NET file uses things like "O*NET-SOC Code",
    # "Element Name", "Scale ID", "Data Value", "Title".
    df.columns = [c.strip() for c in df.columns]

    needed = ["O*NET-SOC Code", "Title", "Element Name", "Scale ID", "Data Value"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected columns: {missing}; got {list(df.columns)}")

    df = df[needed].copy()
    # Keep only Importance (IM) and Level (LV) rows.
    df = df[df["Scale ID"].isin(["IM", "LV"])]

    # Pivot so each (occ, skill) becomes one row with IM and LV columns.
    wide = df.pivot_table(
        index=["O*NET-SOC Code", "Title", "Element Name"],
        columns="Scale ID",
        values="Data Value",
        aggfunc="mean",
    ).reset_index()
    wide = wide.dropna(subset=["IM", "LV"])

    # O*NET scales: Importance 1..5, Level 0..7. Normalize each to [0,1] then average.
    im_norm = (wide["IM"] - 1.0) / 4.0
    lv_norm = wide["LV"] / 7.0
    wide["weight"] = ((im_norm + lv_norm) / 2.0).clip(0.0, 1.0)

    # Drop zero-weight edges: they carry no signal.
    wide = wide[wide["weight"] > 0.0].reset_index(drop=True)
    wide = wide.rename(
        columns={
            "O*NET-SOC Code": "occ_code",
            "Title": "occ_title",
            "Element Name": "skill",
        }
    )
    return wide[["occ_code", "occ_title", "skill", "IM", "LV", "weight"]]


def save_edges(df: pd.DataFrame, path: Path = EDGES_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data] wrote {len(df):,} edges -> {path}")
    return path


def build_node_maps(df: pd.DataFrame, path: Path = NODE_MAPS_PATH) -> dict:
    occs = sorted(df["occ_code"].unique())
    skills = sorted(df["skill"].unique())
    occ2idx = {o: i for i, o in enumerate(occs)}
    skill2idx = {s: i for i, s in enumerate(skills)}
    title_map = (
        df.drop_duplicates("occ_code").set_index("occ_code")["occ_title"].to_dict()
    )
    maps = {
        "occ2idx": occ2idx,
        "skill2idx": skill2idx,
        "idx2occ": {i: o for o, i in occ2idx.items()},
        "idx2skill": {i: s for s, i in skill2idx.items()},
        "occ_title": title_map,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(maps, f)
    print(f"[data] {len(occs)} occupations, {len(skills)} skills -> {path}")
    return maps


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    download_skills()
    edges = load_and_clean()
    save_edges(edges)
    build_node_maps(edges)
