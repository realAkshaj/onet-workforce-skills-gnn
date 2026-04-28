"""Bipartite heterogeneous GraphSAGE for occupation-skill link prediction."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class WeightedSAGEConv(nn.Module):
    """SAGEConv variant with edge-weight-aware neighbor aggregation.

    Computes weighted mean of neighbor features using normalized edge weights,
    then concatenates with self features and projects:
        h_v = W · [h_v || sum_u(w_uv / sum_u' w_u'v) · h_u]

    Uses plain scatter ops instead of MessagePassing to avoid bipartite
    size inference issues in heterogeneous graphs.
    Falls back to unweighted mean when edge_weight is None.
    """

    def __init__(self, in_channels: tuple[int, int] | int, out_channels: int):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin = nn.Linear(in_channels[0] + in_channels[1], out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        x_src, x_dst = x
        n_dst = size[1] if size is not None else x_dst.size(0)
        hidden = x_src.size(-1)

        msgs = x_src[edge_index[0]]                      # (E, hidden)

        if edge_weight is not None:
            w_sum = torch.zeros(n_dst, dtype=edge_weight.dtype, device=edge_weight.device)
            w_sum.scatter_add_(0, edge_index[1], edge_weight)
            w_norm = edge_weight / w_sum[edge_index[1]].clamp(min=1e-8)
            msgs = w_norm.unsqueeze(-1) * msgs

        agg = torch.zeros(n_dst, hidden, dtype=x_src.dtype, device=x_src.device)
        agg.scatter_add_(0, edge_index[1].unsqueeze(-1).expand_as(msgs), msgs)

        if edge_weight is None:
            deg = torch.zeros(n_dst, device=x_src.device)
            deg.scatter_add_(0, edge_index[1],
                             torch.ones(edge_index.size(1), device=x_src.device))
            agg = agg / deg.clamp(min=1).unsqueeze(-1)

        return self.lin(torch.cat([x_dst, agg], dim=-1))


class BipartiteSAGE(nn.Module):
    def __init__(self, in_dims: dict[str, int], hidden_dim: int = 128,
                 out_dim: int = 64, use_edge_weights: bool = False):
        super().__init__()
        self.use_edge_weights = use_edge_weights
        self.proj = nn.ModuleDict(
            {ntype: nn.Linear(d, hidden_dim) for ntype, d in in_dims.items()}
        )
        ConvCls = WeightedSAGEConv if use_edge_weights else SAGEConv
        self.conv1 = HeteroConv(
            {
                ("occupation", "requires", "skill"): ConvCls((hidden_dim, hidden_dim), hidden_dim),
                ("skill", "rev_requires", "occupation"): ConvCls((hidden_dim, hidden_dim), hidden_dim),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("occupation", "requires", "skill"): ConvCls((hidden_dim, hidden_dim), out_dim),
                ("skill", "rev_requires", "occupation"): ConvCls((hidden_dim, hidden_dim), out_dim),
            },
            aggr="sum",
        )

    def forward(self, x_dict: dict, edge_index_dict: dict,
                edge_weight_dict: dict | None = None) -> dict:
        h = {nt: F.relu(self.proj[nt](x)) for nt, x in x_dict.items()}
        if self.use_edge_weights and edge_weight_dict is not None:
            # HeteroConv requires kwargs ending with '_dict'; it strips the suffix
            # and passes edge_weight=edge_weight_dict[edge_type] to each conv.
            h = self.conv1(h, edge_index_dict, edge_weight_dict=edge_weight_dict)
            h = {nt: F.relu(v) for nt, v in h.items()}
            h = self.conv2(h, edge_index_dict, edge_weight_dict=edge_weight_dict)
        else:
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
