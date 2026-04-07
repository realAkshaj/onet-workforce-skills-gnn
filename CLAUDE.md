# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS 519 (ML on Graphs) project: bipartite occupation–skill link prediction on the
O*NET Skills dataset (Release 30.1). Compares cosine / Jaccard collaborative-
filtering baselines against a heterogeneous GraphSAGE model, including a
cold-start split that the non-GNN baselines cannot handle.

All code lives under `project/`. The repo root only holds this file and `.git`.

## Commands

Run everything from inside `project/`:

```bash
cd project
pip install -r requirements.txt

# End-to-end: download data, build graph, run EDA, baselines, train both
# GraphSAGE models, write outputs/results.json
python main.py

# Individual stages (each script is runnable standalone and chdirs to project/)
python src/data_utils.py    # download + parse Skills.xlsx -> onet_edges.csv + node_maps.pkl
python src/graph_build.py   # build HeteroData -> data/processed/hetero_graph.pt
python src/splits.py        # standard + cold-start pickle splits
python src/eda.py           # plots -> outputs/plots/

# Baselines only (needs PYTHONPATH for the sibling import of evaluate.py)
PYTHONPATH=src python src/baselines.py
```

There is no test suite, linter, or formatter configured. Do not invent one.

## Architecture

The pipeline is linear and artifact-driven — each stage writes a file that the
next stage reads. Understanding that data flow is the fastest way to navigate
the code:

```
Skills.xlsx
  -> data_utils.load_and_clean   (pivot IM/LV, compute weight = avg of normalized IM & LV)
  -> data/processed/onet_edges.csv     (occ_code, occ_title, skill, IM, LV, weight)
  -> data/processed/node_maps.pkl      (occ2idx / skill2idx / idx2* / occ_title)

onet_edges.csv + node_maps.pkl
  -> graph_build.build_hetero    (HeteroData with 'occupation' and 'skill' node types)
  -> splits.standard_split       (20% edges per occupation held out)
  -> splits.coldstart_split      (10% of occupations fully held out)

split_standard.pkl, split_coldstart.pkl
  -> baselines.run               (cosine + jaccard, standard split only)
  -> train.train_one             (BipartiteSAGE, both splits)
  -> outputs/results.json        (final comparison table)
```

Key design decisions that span multiple files:

- **Node features are intentionally simple placeholders.** Occupations get a
  dense (n_occ × n_skill) weighted skill profile built from the *training*
  edges only; skills get an identity matrix. This is defined in two places
  that must stay consistent: `graph_build.build_hetero` (for the saved full
  graph) and `train._build_graph_from_train` (rebuilt per split so test edges
  don't leak through the message-passing graph).

- **The hetero graph has both directions.** `('occupation','requires','skill')`
  plus a mirrored `('skill','rev_requires','occupation')`. `BipartiteSAGE`
  (`src/model.py`) wraps each in a `SAGEConv` inside a `HeteroConv`, so both
  relations must exist in `edge_index_dict` or message passing fails. Two
  layers, hidden 128, output 64, dot-product link decoder.

- **Evaluation is shared between baselines and the GNN** via
  `evaluate.evaluate_scores`, which takes a dense `(n_occ, n_skill)` score
  matrix. Training edges are masked to `-inf` for ranking metrics
  (Recall@5/10, Precision@10) so the model can't trivially re-recommend
  training edges, but the raw scores are reused for per-edge AUC (one random
  negative per positive, sampled outside the forbidden set). When adding new
  models, return a score matrix in this same shape and reuse this function —
  do not reimplement metrics.

- **Cold-start is GraphSAGE-only by design.** `cosine`/`jaccard` represent
  each occupation purely from its training edges, so held-out occupations have
  all-zero rows and cannot be scored. `main.py` only runs the GNN on
  `split_coldstart.pkl` and explicitly notes this in `results.json`.

- **Sibling imports inside `src/`.** Modules like `baselines.py` and
  `train.py` import `evaluate` and `model` as top-level names. Either run them
  via `main.py` (which inserts `src/` on `sys.path`) or set
  `PYTHONPATH=src` when running them directly.

- **Checkpointing keeps the best Recall@10** observed during eval (every 10
  epochs) and writes to `checkpoints/sage_<label>.pt`. Callers pass `label`
  and `ckpt_name` to `train_one` to distinguish standard vs cold-start runs.

## Git conventions

- Remote: `origin` -> `https://github.com/realAkshaj/onet-workforce-skills-gnn`
  (branch `main`).
- Commits so far follow `feat(stepN): ...` / `chore: ...`. Keep that style
  when adding new stages.
- Do not add Claude / AI co-author trailers to commit messages, and do not
  add collaborators to the GitHub repo.
- `data/raw/*.xlsx`, `data/processed/*`, and `checkpoints/*.pt` are
  git-ignored — regenerate them by running `python main.py`.
