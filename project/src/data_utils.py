"""Download O*NET dimension files and build the occupation-feature edge table."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Four O*NET dimensions. Each uses Importance (IM) and Level (LV) scales.
# Prefixing element names keeps them distinct across dimensions in the node map.
ONET_BASE = "https://www.onetcenter.org/dl_files/database/db_30_1_excel"
ONET_DIMENSIONS = [
    ("Skills",          f"{ONET_BASE}/Skills.xlsx",           Path("data/raw/Skills.xlsx")),
    ("Abilities",       f"{ONET_BASE}/Abilities.xlsx",        Path("data/raw/Abilities.xlsx")),
    ("WorkActivities",  f"{ONET_BASE}/Work%20Activities.xlsx", Path("data/raw/Work_Activities.xlsx")),
    ("Knowledge",       f"{ONET_BASE}/Knowledge.xlsx",        Path("data/raw/Knowledge.xlsx")),
]

TASKS_URL     = f"{ONET_BASE}/Task%20Statements.xlsx"
TASKS_RAW_PATH = Path("data/raw/Task_Statements.xlsx")
EDGES_PATH    = Path("data/processed/onet_edges.csv")
NODE_MAPS_PATH = Path("data/processed/node_maps.pkl")


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[data] {dest.name} already present ({dest.stat().st_size/1024:.1f} KB)")
        return dest
    print(f"[data] downloading {url}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (cs519-project)"}, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"[data] saved -> {dest} ({len(r.content)/1024:.1f} KB)")
    return dest


def download_all_dimensions():
    for _, url, path in ONET_DIMENSIONS:
        _download(url, path)


def download_tasks() -> Path:
    return _download(TASKS_URL, TASKS_RAW_PATH)


def _load_dimension(xlsx_path: Path, prefix: str) -> pd.DataFrame:
    """Parse one O*NET rating file into (occ_code, occ_title, skill, IM, LV, weight) rows."""
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]

    needed = ["O*NET-SOC Code", "Title", "Element Name", "Scale ID", "Data Value"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{xlsx_path.name}: missing columns {missing}")

    df = df[needed][df["Scale ID"].isin(["IM", "LV"])].copy()

    wide = df.pivot_table(
        index=["O*NET-SOC Code", "Title", "Element Name"],
        columns="Scale ID",
        values="Data Value",
        aggfunc="mean",
    ).reset_index().dropna(subset=["IM", "LV"])

    im_norm = (wide["IM"] - 1.0) / 4.0
    lv_norm = wide["LV"] / 7.0
    wide["weight"] = ((im_norm + lv_norm) / 2.0).clip(0.0, 1.0)
    wide = wide[wide["weight"] > 0.0].reset_index(drop=True)

    wide = wide.rename(columns={
        "O*NET-SOC Code": "occ_code",
        "Title": "occ_title",
        "Element Name": "skill",
    })
    wide["skill"] = prefix + "/" + wide["skill"]
    return wide[["occ_code", "occ_title", "skill", "IM", "LV", "weight"]]


def load_and_clean() -> pd.DataFrame:
    """Load all O*NET dimensions and return a unified (occ, feature, weight) edge table."""
    frames = [_load_dimension(path, prefix) for prefix, _, path in ONET_DIMENSIONS]
    return pd.concat(frames, ignore_index=True)


def load_task_descriptions(xlsx_path: Path = TASKS_RAW_PATH) -> dict[str, str]:
    """Return {occ_code: concatenated task statement text} for all occupations."""
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]
    code_col = next(c for c in df.columns if "O*NET" in c and "Code" in c)
    task_col = next(c for c in df.columns if c.strip() == "Task")
    out: dict[str, str] = {}
    for code, grp in df.groupby(code_col):
        out[str(code)] = " ".join(grp[task_col].dropna().astype(str).tolist())
    return out


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
    title_map = df.drop_duplicates("occ_code").set_index("occ_code")["occ_title"].to_dict()
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
    print(f"[data] {len(occs)} occupations, {len(skills)} features -> {path}")
    return maps


def load_maps(path: Path = NODE_MAPS_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    download_all_dimensions()
    download_tasks()
    edges = load_and_clean()
    save_edges(edges)
    build_node_maps(edges)
