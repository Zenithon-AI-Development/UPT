"""
Grid-structure conditioning utilities for Stage 2.

Provides:
- GridStructureSummary: summarize per-supernode slot layout (mask, levels, positions)
- GridStructureMLP: map summary vector to grid embedding g[b,t,m]
- GridConditionCombiner: combine base (global) conditioner with grid embedding to per-token conditioner
- GridToLatentCondition: pool/attend grid embeddings to latent-token conditioners
"""

from typing import Optional, Tuple

import torch
from torch import nn
import einops


class GridStructureSummary(nn.Module):
    """
    Compute per-supernode grid summary vectors from slot masks, optional AMR levels, and canonical positions.

    Inputs:
        - subnode_mask: [B, T, M, N] (bool)
        - slot_positions: [M, N, dx] (canonical positions, normalized per grid)
        - subnode_pos: [B, T, M, N, dx] (optional real positions)
        - subnode_level: [B, T, M, N] (long) or None

    Output:
        - s: [B, T, M, D_s]
          where D_s = (L_max+1 if levels) + (1 for count) + (dx for pos mean) + (dx for pos var if use_pos_var)
    """

    def __init__(
        self,
        L_max: int = 0,
        ndim: int = 2,
        use_pos_mean: bool = True,
        use_pos_var: bool = True,
        normalize_hist: bool = True,
    ):
        super().__init__()
        assert ndim in (2, 3)
        self.L_max = int(L_max)
        self.ndim = int(ndim)
        self.use_pos_mean = bool(use_pos_mean)
        self.use_pos_var = bool(use_pos_var)
        self.normalize_hist = bool(normalize_hist)

    @property
    def output_dim(self) -> int:
        level_dim = (self.L_max + 1) if self.L_max >= 0 else 0
        count_dim = 1
        pos_mean_dim = self.ndim if self.use_pos_mean else 0
        pos_var_dim = self.ndim if self.use_pos_var else 0
        return level_dim + count_dim + pos_mean_dim + pos_var_dim

    def forward(
        self,
        subnode_mask: torch.Tensor,          # [B, T, M, N], bool
        slot_positions: torch.Tensor,        # [M, N, dx]
        subnode_pos: Optional[torch.Tensor] = None,  # [B, T, M, N, dx]
        subnode_level: Optional[torch.Tensor] = None,  # [B, T, M, N], long
    ) -> torch.Tensor:
        assert subnode_mask.ndim == 4
        B, T, M, N = subnode_mask.shape
        dx = slot_positions.shape[-1]
        assert slot_positions.shape == (M, N, dx)
        assert dx == self.ndim
        device = subnode_mask.device

        mask_f = subnode_mask.to(torch.float32)
        # counts per-supernode
        count = mask_f.sum(dim=-1, keepdim=False)  # [B, T, M]

        # level histogram
        if subnode_level is not None:
            assert subnode_level.shape == (B, T, M, N)
            level_hist = []
            for lvl in range(self.L_max + 1):
                lvl_mask = (subnode_level == lvl).to(torch.float32) * mask_f  # mask inactive
                lvl_count = lvl_mask.sum(dim=-1)  # [B, T, M]
                level_hist.append(lvl_count)
            level_hist = torch.stack(level_hist, dim=-1)  # [B, T, M, L_max+1]
            if self.normalize_hist:
                denom = count.clamp_min(1.0).unsqueeze(-1)
                level_hist = level_hist / denom
        else:
            # No levels provided -> use zeros histogram to keep summary dim consistent
            if self.L_max >= 0:
                level_hist = torch.zeros(B, T, M, self.L_max + 1, device=device, dtype=mask_f.dtype)
            else:
                level_hist = None

        if subnode_pos is not None:
            assert subnode_pos.shape == (B, T, M, N, dx)
            pos = subnode_pos.to(torch.float32)
        else:
            pos = slot_positions.unsqueeze(0).unsqueeze(0).expand(B, T, M, N, dx)
        masked_pos = pos * mask_f.unsqueeze(-1)  # [B, T, M, N, dx]
        # mean
        pos_mean = masked_pos.sum(dim=-2) / count.clamp_min(1.0).unsqueeze(-1)  # [B, T, M, dx]
        # var
        pos_var = (masked_pos - pos_mean.unsqueeze(-2)) ** 2
        pos_var = pos_var.sum(dim=-2) / count.clamp_min(1.0).unsqueeze(-1)  # [B, T, M, dx]

        features = []
        if level_hist is not None:
            features.append(level_hist)  # [B, T, M, L_max+1]
        features.append(count.unsqueeze(-1))  # [B, T, M, 1]
        if self.use_pos_mean:
            features.append(pos_mean)  # [B, T, M, dx]
        if self.use_pos_var:
            features.append(pos_var)  # [B, T, M, dx]

        s = torch.cat(features, dim=-1)  # [B, T, M, D_s]
        return s


class GridStructureMLP(nn.Module):
    """
    Map grid summary vectors to grid embeddings g[b,t,m] in R^{d_grid}.
    """

    def __init__(
        self,
        input_dim: int,
        d_grid: int,
        hidden_mul: int = 4,
    ):
        super().__init__()
        hidden = max(d_grid * hidden_mul, d_grid)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_grid),
            nn.GELU(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: [B, T, M, D_s]
        return self.net(s)


class GridConditionCombiner(nn.Module):
    """
    Combine base global condition c_base[b*T, d_base] with per-supernode grid embedding g[b,T,M,d_grid]
    to produce per-supernode token condition c_token[b*T,M,d_token].
    """

    def __init__(
        self,
        d_base: int,
        d_grid: int,
        d_token: int,
        hidden_mul: int = 2,
    ):
        super().__init__()
        hidden = max((d_base + d_grid) * hidden_mul, d_token)
        self.net = nn.Sequential(
            nn.Linear(d_base + d_grid, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_token),
        )

    def forward(self, c_base: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        c_base: [B*T, d_base]
        g:      [B, T, M, d_grid]
        returns:
            c_token: [B*T, M, d_token]
        """
        B, T, M, d_grid = g.shape
        BT = B * T
        g_bt_m = einops.rearrange(g, "b t m d -> (b t) m d")  # [BT, M, d_grid]
        c_base_bt = c_base.unsqueeze(1).expand(BT, M, -1)     # [BT, M, d_base]
        u = torch.cat([c_base_bt, g_bt_m], dim=-1)            # [BT, M, d_base+d_grid]
        c_token = self.net(u)                                 # [BT, M, d_token]
        return c_token


class GridToLatentCondition(nn.Module):
    """
    Pool/attend grid embeddings g[b,T,M,d_grid] to latent-token conditioners
    c_latent[b*T, n_latent, d_latent].

    Modes:
        - attn: learnable queries (n_latent) attend over M grid embeddings
        - mean: mean pool g, then project and repeat to n_latent
    """

    def __init__(
        self,
        d_grid: int,
        d_latent: int,
        n_latent: int,
        mode: str = "attn",
        attn_dim: Optional[int] = None,
        num_heads: int = 4,
    ):
        super().__init__()
        assert mode in ("attn", "mean")
        self.mode = mode
        self.n_latent = n_latent
        self.d_latent = d_latent
        if mode == "attn":
            d_attn = attn_dim or max(d_grid, d_latent)
            self.q = nn.Parameter(torch.randn(n_latent, d_attn) * 0.02)
            self.q_proj = nn.Linear(d_attn, d_attn)
            self.k_proj = nn.Linear(d_grid, d_attn)
            self.v_proj = nn.Linear(d_grid, d_latent)
            self.num_heads = num_heads
            self.scale = (d_attn // num_heads) ** -0.5
        else:
            self.reduce = nn.Linear(d_grid, d_latent)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """
        g: [B, T, M, d_grid]
        returns:
            c_latent: [B*T, n_latent, d_latent]
        """
        B, T, M, d_grid = g.shape
        BT = B * T
        if self.mode == "mean":
            pooled = g.mean(dim=2)  # [B, T, d_grid]
            pooled = einops.rearrange(pooled, "b t d -> (b t) d")  # [BT, d_grid]
            cond = self.reduce(pooled)  # [BT, d_latent]
            cond = cond.unsqueeze(1).expand(BT, self.n_latent, self.d_latent)
            return cond

        # attention mode
        # K,V from g; Q are learned queries
        K = self.k_proj(einops.rearrange(g, "b t m d -> (b t) m d"))  # [BT, M, d_attn]
        V = self.v_proj(einops.rearrange(g, "b t m d -> (b t) m d"))  # [BT, M, d_latent]
        Q = self.q_proj(self.q).unsqueeze(0).expand(BT, -1, -1)       # [BT, n_latent, d_attn]

        # multi-head attention (manual)
        h = self.num_heads
        def split_heads(x, d):  # [BT, L, d] -> [BT, h, L, d//h]
            return einops.rearrange(x, "bt l (h dh) -> bt h l dh", h=h)
        Qh = split_heads(Q, Q.size(-1))
        Kh = split_heads(K, K.size(-1))
        Vh = einops.rearrange(V, "bt m d -> bt 1 m d").expand(-1, h, -1, -1)

        attn = torch.matmul(Qh, Kh.transpose(-2, -1)) * self.scale  # [BT, h, n_latent, M]
        attn = attn.softmax(dim=-1)
        Oh = torch.matmul(attn, Vh)  # [BT, h, n_latent, d_latent]
        # merge heads by averaging (keeps d_latent unchanged)
        O = Oh.mean(dim=1)  # [BT, n_latent, d_latent]
        return O


