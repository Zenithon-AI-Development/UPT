from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def pad_to_power_of_k(inputs: torch.Tensor, fill_value: float, k: int) -> torch.Tensor:
    def _next_power_of_k(x: int, base: int) -> int:
        return base ** math.ceil(math.log(x, base))

    if inputs.ndim != 4:
        raise ValueError(f"pad_to_power_of_k expects tensor of shape [B,H,W,C]; got {inputs.shape}")

    b, h, w, c = inputs.shape
    target_h = _next_power_of_k(h, k)
    target_w = _next_power_of_k(w, k)
    max_tgt = max(target_h, target_w)
    pad_h = max_tgt - h
    pad_w = max_tgt - w
    padding = (0, 0, 0, pad_w, 0, pad_h)  # (w_left, w_right, h_top, h_bottom)
    return F.pad(inputs, padding, "constant", fill_value)


def unified_condition_patches(
    regions: torch.Tensor,
    masks: torch.Tensor,
    nonzero_ratio_threshold: float,
    min_size: int,
    max_size: int,
    common_refine_threshold: float = 0.4,
    integral_refine_threshold: float = 0.1,
    vorticity_threshold: float = 0.5,
    momentum_threshold: float = 0.5,
    shear_threshold: float = 0.5,
    condition_type: str = "grad",
) -> torch.Tensor:
    """
    Replica of the AMR_Transformer heuristic that decides whether to refine or keep a region.
    """
    num_of_patches, height, width, _ = regions.shape

    non_zero_mask = torch.any(regions != 0, dim=-1)
    non_zero_counts = non_zero_mask.sum(dim=(1, 2))
    total_elements = height * width

    results = torch.full((num_of_patches,), 2, dtype=torch.int64, device=regions.device)

    if height > max_size:
        results[:] = -1
    if height == max_size:
        results[:] = 1

    zero_or_small = (height < min_size) | (non_zero_counts == 0)
    results[zero_or_small] = 0

    valid_mask = results == 2

    if valid_mask.any():
        nonzero_ratio = non_zero_counts[valid_mask] / total_elements

        u = regions[valid_mask, :, :, 0]
        v = regions[valid_mask, :, :, 1]

        du_dx = torch.gradient(u, dim=2)[0]
        du_dy = torch.gradient(u, dim=1)[0]
        dv_dx = torch.gradient(v, dim=2)[0]
        dv_dy = torch.gradient(v, dim=1)[0]

        gradients_u = torch.sqrt(du_dx ** 2 + du_dy ** 2)
        gradients_v = torch.sqrt(dv_dx ** 2 + dv_dy ** 2)
        gradients_norm = torch.sqrt(gradients_u ** 2 + gradients_v ** 2)
        gradients = gradients_norm.mean(dim=(1, 2))
        avg_gradient = gradients.mean()
        common_condition = gradients > (avg_gradient * common_refine_threshold)

        sorted_gradients = torch.sort(gradients).values
        threshold_index = int(len(sorted_gradients) * (1 - integral_refine_threshold))
        threshold_index = max(min(threshold_index, sorted_gradients.numel() - 1), 0)
        integral_condition = gradients >= sorted_gradients[threshold_index]

        mask_condition = torch.any(masks[valid_mask] != 1, dim=(1, 2, 3))
        ratio_condition = nonzero_ratio <= nonzero_ratio_threshold

        if condition_type == "grad":
            condition_result = mask_condition | ratio_condition | integral_condition | common_condition
        else:
            raise ValueError("Invalid condition type specified")

        result_true = torch.ones(condition_result.shape, dtype=torch.int64, device=regions.device)
        result_false = torch.full(condition_result.shape, 2, dtype=torch.int64, device=regions.device)
        results[valid_mask] = torch.where(condition_result, result_true, result_false)

    return results


def quadtree_partition_parallel(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    feature_field: torch.Tensor,
    nonzero_ratio_threshold: float = 0.99,
    k: int = 2,
    min_size: int = 1,
    max_size: int = 4,
    common_refine_threshold: float = 0.4,
    integral_refine_threshold: float = 0.1,
    vorticity_threshold: float = 0.5,
    momentum_threshold: float = 0.5,
    shear_threshold: float = 0.5,
    condition_type: str = "grad",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quadtree (or k-ary) partitioning implemented in the original AMR_Transformer.
    Returns (regions_final, patches_final, labels_final), matching original semantics.
    """
    device = inputs.device
    inputs = inputs.to(device)
    labels = labels.to(device)
    mask = mask.to(device)
    feature_field = feature_field.to(device)

    h_, w_, _ = inputs.shape
    cur_regions_index = torch.tensor([[0, 0, h_ - 1, w_ - 1]], dtype=torch.long, device=device)

    regions_final_ls = []
    patches_final_ls = []
    labels_final_ls = []
    depth = 0

    while cur_regions_index.shape[0] > 0:
        def _get_data(data_ls, x1, y1, x2, y2):
            h_cur = (x2[0] - x1[0] + 1).item()
            w_cur = (y2[0] - y1[0] + 1).item()
            n = x1.shape[0]
            rows = torch.arange(h_cur, device=device).unsqueeze(0).expand(n, -1) + x1.unsqueeze(1)
            cols = torch.arange(w_cur, device=device).unsqueeze(0).expand(n, -1) + y1.unsqueeze(1)
            grid_rows = rows.unsqueeze(2).expand(n, h_cur, w_cur)
            grid_cols = cols.unsqueeze(1).expand(n, h_cur, w_cur)
            if isinstance(data_ls, (list, tuple)):
                return [data[grid_rows, grid_cols] for data in data_ls]
            return data_ls[grid_rows, grid_cols]

        x1s, y1s, x2s, y2s = (
            cur_regions_index[:, 0],
            cur_regions_index[:, 1],
            cur_regions_index[:, 2],
            cur_regions_index[:, 3],
        )
        h_cur = (x2s[0] - x1s[0] + 1).item()
        w_cur = (y2s[0] - y1s[0] + 1).item()
        h_sub = h_cur // k
        w_sub = w_cur // k
        if h_sub <= 0 or w_sub <= 0 or h_sub < max(1, min_size // k):
            break

        cur_regions, cur_labels, cur_masks, cur_features = _get_data(
            [inputs, labels, mask, feature_field], x1s, y1s, x2s, y2s
        )

        con_res = unified_condition_patches(
            cur_regions,
            cur_masks,
            nonzero_ratio_threshold,
            min_size,
            max_size,
            common_refine_threshold=common_refine_threshold,
            integral_refine_threshold=integral_refine_threshold,
            vorticity_threshold=vorticity_threshold,
            momentum_threshold=momentum_threshold,
            shear_threshold=shear_threshold,
            condition_type=condition_type,
        )

        subdivide_index = torch.where((con_res == 1) | (con_res == -1) | (con_res == 2))[0]
        con_res = con_res[subdivide_index]
        if con_res.numel() == 0:
            break

        to_subdivide_regions_index = cur_regions_index[subdivide_index]
        subdivide_num = to_subdivide_regions_index.shape[0]

        row_offsets = torch.arange(k, device=device).view(-1, 1) * h_sub
        col_offsets = torch.arange(k, device=device).view(1, -1) * w_sub
        x1_sub = to_subdivide_regions_index[:, 0].view(-1, 1, 1) + row_offsets
        y1_sub = to_subdivide_regions_index[:, 1].view(-1, 1, 1) + col_offsets
        x2_sub = x1_sub + h_sub - 1
        y2_sub = y1_sub + w_sub - 1
        x1_sub = x1_sub.expand(-1, k, k).reshape(subdivide_num, k * k)
        y1_sub = y1_sub.expand(-1, k, k).reshape(subdivide_num, k * k)
        x2_sub = x2_sub.expand(-1, k, k).reshape(subdivide_num, k * k)
        y2_sub = y2_sub.expand(-1, k, k).reshape(subdivide_num, k * k)
        subdivided_regions_index = torch.stack([x1_sub, y1_sub, x2_sub, y2_sub], dim=-1)

        next_loop_regions_index = subdivided_regions_index[(con_res == 1) | (con_res == -1)]
        cur_regions_index = next_loop_regions_index.reshape(-1, 4)

        subdivide_save_regions_index = subdivided_regions_index[(con_res == 1) | (con_res == 2)]
        subdivide_save_num = subdivide_save_regions_index.shape[0]
        if subdivide_save_num == 0:
            depth += 1
            continue
        subdivide_save_regions_index = subdivide_save_regions_index.reshape(-1, 4)
        regions_final_ls.append(subdivide_save_regions_index)

        subdivide_save_regions, subdivide_save_labels = _get_data(
            [inputs, labels],
            subdivide_save_regions_index[:, 0],
            subdivide_save_regions_index[:, 1],
            subdivide_save_regions_index[:, 2],
            subdivide_save_regions_index[:, 3],
        )
        subdivide_save_regions = subdivide_save_regions.reshape(subdivide_save_num, k * k, h_sub, w_sub, -1)
        subdivide_save_labels = subdivide_save_labels.reshape(subdivide_save_num, k * k, h_sub, w_sub, -1)
        subdivide_save_patches = subdivide_save_regions.mean(dim=(2, 3))
        subdivide_save_labels = subdivide_save_labels.mean(dim=(2, 3))

        subdivide_save_regions_index = subdivide_save_regions_index.reshape(subdivide_save_num, k * k, 4)
        x1 = subdivide_save_regions_index[:, :, 0].unsqueeze(-1)
        y1 = subdivide_save_regions_index[:, :, 1].unsqueeze(-1)
        depth_tensor = torch.full((subdivide_save_num, k * k, 1), depth, device=device)
        subdivide_save_patches = torch.cat([subdivide_save_patches, depth_tensor, x1, y1], dim=-1)
        subdivide_save_labels = torch.cat([subdivide_save_labels, depth_tensor, x1, y1], dim=-1)

        patches_final_ls.append(subdivide_save_patches)
        labels_final_ls.append(subdivide_save_labels)
        depth += 1

    regions_final = torch.cat(regions_final_ls, dim=0) if regions_final_ls else torch.empty(0, 4, device=device, dtype=torch.long)
    patches_final = torch.cat(patches_final_ls, dim=0) if patches_final_ls else torch.empty(0, k * k, inputs.shape[-1] + 3, device=device)
    labels_final = torch.cat(labels_final_ls, dim=0) if labels_final_ls else torch.empty(0, k * k, labels.shape[-1] + 3, device=device)

    return regions_final, patches_final, labels_final

