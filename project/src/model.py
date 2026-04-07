"""Bipartite heterogeneous GraphSAGE for occupation-skill link prediction."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class BipartiteSAGE(nn.Module):
    def __init__(self, in_dims: dict[str, int], hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.proj = nn.ModuleDict(
            {ntype: nn.Linear(d, hidden_dim) for ntype, d in in_dims.items()}
        )
        self.conv1 = HeteroConv(
            {
                ("occupation", "requires", "skill"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("skill", "rev_requires", "occupation"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("occupation", "requires", "skill"): SAGEConv((hidden_dim, hidden_dim), out_dim),
                ("skill", "rev_requires", "occupation"): SAGEConv((hidden_dim, hidden_dim), out_dim),
            },
            aggr="sum",
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        h = {nt: F.relu(self.proj[nt](x)) for nt, x in x_dict.items()}
        h = self.conv1(h, edge_index_dict)
        h = {nt: F.relu(v) for nt, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        return h

    @staticmethod
    def decode(occ_emb: torch.Tensor, skill_emb: torch.Tensor,
               occ_idx: torch.Tensor, skill_idx: torch.Tensor) -> torch.Tensor:
        return (occ_emb[occ_idx] * skill_emb[skill_idx]).sum(dim=-1)

    @staticmethod
    def score_all(occ_emb: torch.Tensor, skill_emb: torch.Tensor) -> torch.Tensor:
        return occ_emb @ skill_emb.t()
