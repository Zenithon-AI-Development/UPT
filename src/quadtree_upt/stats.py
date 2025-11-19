import torch

def compute_supernode_stats(
    *,
    super_mask: torch.Tensor,
    sub_mask: torch.Tensor,
    sub_depths: torch.Tensor,
    super_depths: torch.Tensor,
    sub_rel_centers: torch.Tensor,
    super_half_sizes: torch.Tensor,
    sub_distances: torch.Tensor,
    max_children: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a compact geometry-aware statistics vector per supernode."""

    super_mask_f = super_mask.unsqueeze(-1).to(dtype=torch.float32)
    sub_mask_f = sub_mask.to(dtype=torch.float32)
    counts = sub_mask_f.sum(dim=-1, keepdim=True)
    safe_counts = counts.clamp_min(1.0)

    occupancy = counts / max(max_children, 1)

    super_depths_f = super_depths.float().unsqueeze(-1).clamp_min(1.0)
    sub_depths_f = sub_depths.float()
    norm_depths = sub_depths_f / super_depths_f
    mean_depth = (norm_depths * sub_mask_f).sum(dim=-1, keepdim=True) / safe_counts
    depth_var = ((norm_depths - mean_depth) ** 2 * sub_mask_f).sum(dim=-1, keepdim=True) / safe_counts
    depth_std = torch.sqrt(depth_var.clamp_min(eps))

    norm_half = torch.norm(super_half_sizes.float(), dim=-1, keepdim=True).clamp_min(eps)
    mean_radius = (sub_distances * sub_mask_f).sum(dim=-1, keepdim=True) / safe_counts
    mean_radius = mean_radius / norm_half
    radius_var = (((sub_distances / norm_half) - mean_radius) ** 2 * sub_mask_f).sum(dim=-1, keepdim=True) / safe_counts
    radius_std = torch.sqrt(radius_var.clamp_min(eps))

    half_sizes = super_half_sizes.float().unsqueeze(-2).clamp_min(eps)
    rel_norm = sub_rel_centers / half_sizes

    right = ((rel_norm[..., 0] >= 0).float() * sub_mask_f).sum(dim=-1, keepdim=True)
    left = ((rel_norm[..., 0] < 0).float() * sub_mask_f).sum(dim=-1, keepdim=True)
    top = ((rel_norm[..., 1] >= 0).float() * sub_mask_f).sum(dim=-1, keepdim=True)
    bottom = ((rel_norm[..., 1] < 0).float() * sub_mask_f).sum(dim=-1, keepdim=True)
    lr_imbalance = (right - left) / safe_counts
    tb_imbalance = (top - bottom) / safe_counts

    components = [
        occupancy,
        mean_depth,
        depth_std,
        mean_radius,
        radius_std,
        lr_imbalance,
        tb_imbalance,
    ]
    stats = torch.cat(components, dim=-1)
    stats = stats * super_mask_f
    return stats


__all__ = ["compute_supernode_stats"]
