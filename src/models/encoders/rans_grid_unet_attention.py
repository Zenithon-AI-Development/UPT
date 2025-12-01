import math

import einops
import torch
import torch.nn.functional as F
from torch import nn

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier
from kappamodules.transformer import PerceiverPoolingBlock

# High level story:
# - A 3D U-Net backbone augmented with linear attention at every scale encodes SDF grids.
# - The grid output is flattened and pooled with a Perceiver block to a fixed token count.
# - A learned type token tags the output for downstream consumers.

# Bug risk: GroupNorm throws if channels are not divisible by num_groups; choose divisor defensively.
def _group_norm(num_channels: int) -> nn.GroupNorm:
    # Pick a divisor of num_channels to avoid GroupNorm shape issues (GroupNorm requires divisible channels).
    num_groups = math.gcd(num_channels, 8) or 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ConvBlock(nn.Module):
    # Bug risk: mismatched in/out channels will surface here; keep channel plan consistent with UNet setup.
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Two 3D convs with GELU and GroupNorm; the basic local feature block.
        self.block = nn.Sequential(
            _group_norm(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            _group_norm(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LinearAttention3d(nn.Module):
    # Bug risk: huge spatial sizes can make attention memory heavy despite linear trick; watch h*w*d.
    def __init__(self, dim: int, num_heads: int, dim_head: int = 32, dropout: float = 0.0):
        super().__init__()
        inner_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head
        # 1x1 conv to produce qkv over channel dimension (spatial dims stay intact until flatten).
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, kernel_size=1, bias=False)
        # Project heads back to model dim.
        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim, h, w, d)
        b, c, h, w, d = x.shape
        n = h * w * d
        x_flat = x.reshape(b, c, n)  # flatten spatial dims to sequence length n

        qkv = self.to_qkv(x_flat).chunk(3, dim=1)
        q, k, v = qkv
        q = q.view(b, self.num_heads, self.dim_head, n).transpose(2, 3)  # (b, h, n, d_h)
        k = k.view(b, self.num_heads, self.dim_head, n).transpose(2, 3)
        v = v.view(b, self.num_heads, self.dim_head, n).transpose(2, 3)

        # Linear attention: softmax over tokens for q and over channels for k (kernel trick).
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        # Context: sum over tokens -> (b, heads, dim_head, dim_head)
        context = torch.einsum("b h n d, b h n e -> b h d e", k, v)  # (b, h, d_h, d_h)
        # Attend: mix context back into q -> (b, heads, tokens, dim_head)
        attn_out = torch.einsum("b h n d, b h d e -> b h n e", q, context)  # (b, h, n, d_h)

        # Merge heads and reshape to original spatial grid.
        attn_out = attn_out.transpose(2, 3).reshape(b, -1, n)  # (b, h*d_h, n)
        out = self.to_out(attn_out)
        return out.reshape(b, c, h, w, d)


class LinearAttentionUnet(nn.Module):
    # Bug risk: resolution must be divisible by pooling schedule; enforced at caller but keep in mind for config.
    def __init__(
            self,
            input_dim: int,
            dim: int,
            depth: int,
            attn_heads: int,
            attn_dim_head: int = 32,
            attn_dropout: float = 0.0,
    ):
        super().__init__()
        assert depth >= 2, "depth must be at least 2 for down + up path"

        # schedule channel widths, increasing until clamped at dim
        channels = []
        width = max(1, dim // depth)
        for i in range(depth):
            if i > 0:
                width = min(dim, width * 2)
            channels.append(width)
        channels[-1] = dim  # ensure bottleneck width hits target dim
        self.channels = channels

        self.downs = nn.ModuleList()
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            block = ConvBlock(in_ch, out_ch)
            attn = LinearAttention3d(out_ch, num_heads=attn_heads, dim_head=attn_dim_head, dropout=attn_dropout)
            # no downsample after last level
            downsample = nn.MaxPool3d(kernel_size=2, stride=2) if i < depth - 1 else nn.Identity()
            self.downs.append(nn.ModuleDict({
                "block": block,
                "attn": attn,
                "down": downsample,
            }))
            in_ch = out_ch

        self.ups = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            skip_ch = channels[i]
            in_ch = channels[i + 1]
            upsample = nn.ConvTranspose3d(in_ch, skip_ch, kernel_size=2, stride=2)
            block = ConvBlock(in_ch + skip_ch, skip_ch)
            attn = LinearAttention3d(skip_ch, num_heads=attn_heads, dim_head=attn_dim_head, dropout=attn_dropout)
            self.ups.append(nn.ModuleDict({
                "up": upsample,
                "block": block,
                "attn": attn,
            }))

        self.proj_out = nn.Conv3d(channels[0], dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for stage in self.downs:
            x = stage["block"](x)
            x = stage["attn"](x)
            skips.append(x)
            x = stage["down"](x)

        x = skips.pop()  # start decode from bottleneck
        for stage in self.ups:
            x = stage["up"](x)
            skip = skips.pop()
            # pad if shapes mismatch due to odd sizes
            if x.shape[-3:] != skip.shape[-3:]:
                target = skip.shape[-3:]
                x = F.interpolate(x, size=target, mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = stage["block"](x)
            x = stage["attn"](x)

        return self.proj_out(x)


class RansGridUnetAttention(SingleModelBase):
    # Bug risk: assumes cubic grids with equal sides and divisibility by 2**(depth-1); assertions guard this.
    def __init__(
            self,
            dim,
            num_attn_heads,
            num_output_tokens,
            depth=4,
            attn_dim_head=32,
            attn_dropout=0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.num_output_tokens = num_output_tokens
        self.attn_dim_head = attn_dim_head
        self.attn_dropout = attn_dropout
        self.resolution = self.data_container.get_dataset().grid_resolution
        self.ndim = len(self.resolution)

        # sdf + grid_pos
        if self.data_container.get_dataset().concat_pos_to_sdf:
            input_dim = 4
        else:
            input_dim = 1

        assert all(self.resolution[0] == r for r in self.resolution[1:])
        assert self.resolution[0] % (2 ** (depth - 1)) == 0, "resolution must be divisible by pooling steps"

        self.unet = LinearAttentionUnet(
            input_dim=input_dim,
            dim=dim,
            depth=depth,
            attn_heads=num_attn_heads,
            attn_dim_head=attn_dim_head,
            attn_dropout=attn_dropout,
        )

        self.perceiver = PerceiverPoolingBlock(
            dim=dim,
            num_heads=num_attn_heads,
            num_query_tokens=num_output_tokens,
            perceiver_kwargs=dict(init_weights="truncnormal"),
        )

        self.type_token = nn.Parameter(torch.empty(size=(1, 1, dim,)))

        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = self.ndim
        self.output_shape = (num_output_tokens, dim)

    def model_specific_initialization(self):
        nn.init.trunc_normal_(self.type_token)

    def get_model_specific_param_group_modifiers(self):
        return [ExcludeFromWdByNameModifier(name="type_token")]

    def forward(self, x):
        # Inputs arrive as dim-last grids (B, H, W, D, C); switch to channels-first for convs.
        x = einops.rearrange(x, "batch_size height width depth dim -> batch_size dim height width depth")
        # Encode with linear-attention U-Net to produce dense feature grid.
        x = self.unet(x)
        # Flatten spatial grid into tokens.
        x = einops.rearrange(x, "batch_size dim height width depth -> batch_size (height width depth) dim")
        # Pool to fixed-length token set with Perceiver pooling.
        x = self.perceiver(x)
        # Add a learned type token for downstream conditioning.
        x = x + self.type_token
        return x
