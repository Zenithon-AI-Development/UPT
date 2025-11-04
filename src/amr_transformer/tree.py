import math
import time

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

import torch.nn as nn
from einops import rearrange

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def pad_to_power_of_k(inputs, fill_value, k):
    def next_power_of_k(x, k):
        return k ** math.ceil(math.log(x, k))

    b, h, w, c = inputs.shape
    target_h = next_power_of_k(h, k)
    target_w = next_power_of_k(w, k)
    max_tgt = max(target_h, target_w)
    pad_h = max_tgt - h
    pad_w = max_tgt - w

    # Calculate padding size
    padding = (0, 0, 0, pad_w, pad_h, 0)  # Corrected padding order
    padded_inputs = F.pad(inputs, padding, "constant", fill_value)
    return padded_inputs

def unified_condition_patches(regions,
                              masks,
                              nonzero_ratio_threshold,
                              min_size,
                              max_size,
                              common_refine_threshold=4.0,
                              integral_refine_threshold=0.1,
                              vorticity_threshold=0.5,
                              momentum_threshold=0.5,
                              shear_threshold=0.5,
                              condition_type='diff'):
    '''
    regions shape: num_of_patches, h, w, c
    masks shape: num_of_patches, h, w, 1
    nonzero_ratio_threshold: float
    min_size: int
    max_size: int
    common_refine_threshold: float
    integral_refine_threshold: float
    '''
    num_of_patches, height, width, _ = regions.shape

    # Calculate mean, absolute difference, and non-zero region for each patch
    # region_means = regions.mean(dim=(1, 2, 3), keepdim=True)
    # abs_diffs = torch.abs(regions - region_means)

    non_zero_mask = torch.any(regions != 0, dim=-1)
    non_zero_counts = non_zero_mask.sum(dim=(1, 2))
    total_elements = regions.shape[1] * regions.shape[2]

    results = torch.full((num_of_patches,), 2, dtype=torch.int64, device=regions.device)

    # Condition 1: height > max_size, return -1
    results[height > max_size] = -1

    # Condition 2: height == max_size, return 1
    results[height == max_size] = 1

    # Condition 3: height < min_size or region is all zeros, return 0
    zero_or_small = (height < min_size) | (non_zero_counts == 0)
    # results[zero_or_small] = 0

    valid_mask = (results == 2)

    if valid_mask.any():
        # Calculate variance and non-zero ratio
        # variance = regions[valid_mask].var(dim=(1, 2, 3))
        nonzero_ratio = non_zero_counts[valid_mask] / total_elements

        # **Unified gradient calculation**
        # For the two components of the velocity field (assumed to be u and v, corresponding to positions 0 and 1 in the last dimension of regions)
        u = regions[valid_mask, :, :, 0]
        v = regions[valid_mask, :, :, 1]

        # Calculate gradients of u and v in x and y directions
        du_dx = torch.gradient(u, dim=2)[0]
        du_dy = torch.gradient(u, dim=1)[0]
        dv_dx = torch.gradient(v, dim=2)[0]
        dv_dy = torch.gradient(v, dim=1)[0]

        # **Calculate gradient norm, used for common_condition and integral_condition**
        gradients_u = torch.sqrt(du_dx ** 2 + du_dy ** 2)
        gradients_v = torch.sqrt(dv_dx ** 2 + dv_dy ** 2)
        gradients_norm = torch.sqrt(gradients_u ** 2 + gradients_v ** 2)
        gradients = gradients_norm.mean(dim=(1, 2))

        # Calculate common_refine_condition
        avg_gradient = gradients.mean()

        # print(gradients.shape, avg_gradient.shape,avg_gradient * common_refine_threshold)

        common_condition = gradients > (avg_gradient * common_refine_threshold)

        # Calculate integral_refine_condition
        sorted_gradients = torch.sort(gradients).values
        threshold_index = int(len(sorted_gradients) * (1 - integral_refine_threshold))
        integral_condition = gradients >= sorted_gradients[threshold_index]

        # **Method 1 and Method 4: Calculate vorticity and curl field conditions**
        # Calculate vorticity (curl)
        vorticity = dv_dx - du_dy  # dv/dx - du/dy
        vorticity_magnitude = torch.abs(vorticity)
        vorticity_mean = vorticity_magnitude.mean(dim=(1, 2))

        # Vorticity refinement condition (Method 1)
        vorticity_condition = vorticity_mean > vorticity_threshold * 0.5

        # **Method 3: Momentum conservation deviation refinement**
        # Calculate total momentum for each patch
        momentum_x = u.sum(dim=(1, 2))
        momentum_y = v.sum(dim=(1, 2))
        momentum_magnitude = torch.sqrt(momentum_x ** 2 + momentum_y ** 2)
        avg_momentum = momentum_magnitude.mean()
        momentum_condition = momentum_magnitude > avg_momentum * momentum_threshold * 9

        # **Method 7: Kelvin-Helmholtz instability refinement**
        # Kelvin-Helmholtz instability detection: based on velocity field shear layer strength
        shear_strength = torch.abs(du_dy - dv_dx)  # Shear strength
        kh_instability_condition = torch.any(shear_strength > shear_threshold, dim=(1, 2))

        mask_condition = torch.any(masks[valid_mask] != 1, dim=(1, 2, 3))
        ratio_condition = nonzero_ratio <= nonzero_ratio_threshold

        # if condition_type == 'diff':
        #     condition_result = diff_condition | mask_condition | ratio_condition | vorticity_condition | curl_condition | momentum_condition | instability_condition
        # elif condition_type == 'var':
        #     condition_result = var_condition | mask_condition | ratio_condition
        # elif condition_type == 'both':
        #     condition_result = (diff_condition & var_condition) | mask_condition | ratio_condition
        # elif condition_type == 'either':
        #     condition_result = (diff_condition | var_condition) | mask_condition | ratio_condition
        if condition_type == 'grad':
            condition_result = mask_condition | ratio_condition | integral_condition | common_condition
                    # condition_result = vorticity_condition
        # vorticity_condition | momentum_condition | kh_instability_condition
        else:
            raise ValueError("Invalid condition type specified")

        results[valid_mask] = torch.where(condition_result, 1, 2)

    return results

def quadtree_partition_parallel(
    inputs: torch.tensor, labels: torch.tensor, mask: torch.tensor,
    feature_field: torch.tensor,
    nonzero_ratio_threshold: float = 0.99, k: int = 2,
    min_size: int = 1, max_size: int = 4,
    common_refine_threshold=0.4,
    integral_refine_threshold=0.1,
    vorticity_threshold=0.5,
    momentum_threshold=0.5,
    shear_threshold=0.5,
    condition_type='both',
                                ):
    '''
    inputs: Input image, shape is (h,w,c)
    labels: Input labels, shape is (h,w,c)
    mask: Input mask, shape is (h,w,1) with 0-1 values
    feature_field: Input feature field, shape is (h,w,c)
    '''
    inputs, labels, mask, feature_field = inputs.cuda(), labels.cuda(), mask.cuda(), feature_field.cuda()
    h_, w_, c = inputs.shape

    # Initial region is the entire image
    # N,4
    # N is the number of regions, 4 is for region indexing (top-left and bottom-right corners)
    cur_regions_index = torch.tensor([[0, 0, h_ - 1, w_ - 1]], dtype=torch.long, device='cuda')  # shape: (N,4)

    regions_final_ls = []
    patches_final_ls = []
    labels_final_ls = []

    depth = 0

    # Extract cur_regions from cur_regions_index, shape is (N,h_cur,w_cur,c)
    while cur_regions_index.shape[0] > 0:

        # print(cur_regions_index)

        def get_data(data_ls, x1, y1, x2, y2):
            '''
            data one: shape: (h, w, c), torch.tensor
            x1, y1, x2, y2 shape: (N,), torch.tensor
            return shape: (N, h_, w_, c)
            '''
            h_ = (x2[0] - x1[0] + 1).item()
            w_ = (y2[0] - y1[0] + 1).item()

            N = x1.shape[0]

            rows = torch.arange(h_, device='cuda').unsqueeze(0).expand(N, -1) + x1.unsqueeze(1)
            cols = torch.arange(w_, device='cuda').unsqueeze(0).expand(N, -1) + y1.unsqueeze(1)

            grid_rows = rows.unsqueeze(2).expand(N, h_, w_)
            grid_cols = cols.unsqueeze(1).expand(N, h_, w_)

            grid_rows = grid_rows.view(N, h_, w_)
            grid_cols = grid_cols.view(N, h_, w_)

            if isinstance(data_ls, list):
                data = []
                for data_one in data_ls:
                    data.append(data_one[grid_rows, grid_cols])
                return data
            else:
                return data_ls[grid_rows, grid_cols]

        # Extract current region
        x1s, y1s, x2s, y2s = cur_regions_index[:, 0], cur_regions_index[:, 1], cur_regions_index[:,
                                                                               2], cur_regions_index[:,
                                                                                   3]  # shape: (N,)

        # Determine width and height of each region (all regions same)
        h_cur = (x2s[0] - x1s[0] + 1).item()
        w_cur = (y2s[0] - y1s[0] + 1).item()

        # This must be divisible
        h_sub = h_cur // k
        w_sub = w_cur // k

        # If width or height of current region is less than or equal to 1, stop subdividing
        if h_sub <= 0 or w_sub <= 0 or h_sub < min_size // k:
            break

        # Use advanced indexing to extract corresponding regions, shape is (N,h_cur,w_cur,c)
        cur_regions, cur_labels, cur_masks, cur_features = get_data([inputs, labels, mask, feature_field], x1s, y1s,
                                                                    x2s, y2s)
        # cur_regions shape: (N, h_cur, w_cur, c)
        # cur_labels shape: (N, h_cur, w_cur, c)
        # cur_masks shape: (N, h_cur, w_cur, 1)
        # cur_features shape: (N, h_cur, w_cur, c)

        # Next, calculate condition, get condition value con_res for each region, shape is (N,)
        con_res = unified_condition_patches(cur_regions, cur_masks,
                                            nonzero_ratio_threshold,
                                            min_size, max_size,
                                            common_refine_threshold=common_refine_threshold,
                                            integral_refine_threshold=integral_refine_threshold,
                                            vorticity_threshold=vorticity_threshold,
                                            momentum_threshold=momentum_threshold,
                                            shear_threshold=shear_threshold,
                                            condition_type=condition_type)  # (N, )

        # print(con_res.shape)

        # Based on con_res, discard regions with value 0 (not stored not subdivided), store but not subdivide regions with value 2, store and subdivide regions with value 1, subdivide but not store regions with value -1
        # Find index of regions to store (value 1 or 2), shape is (save_num,)
        # Find index of regions to subdivide (value 1 or -1), shape is (subdivide_num,)

        # Storage logic: subdivide first, then store subdivided regions
        # Subdivision logic: subdivide first, let subdivided regions be further subdivided in next iteration, then store subdivided regions

        subdivide_index = torch.where((con_res == 1) | (con_res == -1) | (con_res == 2))[0]
        # Set value to 0 for regions where con_res is not equal to 1, -1, or 2
        con_res[(con_res != 1) & (con_res != -1) & (con_res != 2)] = 0
        # Remove regions where con_res is 0
        con_res = con_res[con_res != 0]
        # At this point con_res shape is (subdivide_num,)

        # Perform k-way subdivision for each region
        to_subdivide_regions_index = cur_regions_index[subdivide_index]  # shape: (subdivide_num, 4)
        # Index of new regions after k-way subdivision for each region in to_subdivide_regions_index, shape is (subdivide_num, k*k, 4), divided into k*k regions

        subdivide_num = to_subdivide_regions_index.shape[0]

        assert subdivide_num == con_res.shape[0]

        if subdivide_num == 0:
            break

        # print('height', h_cur, 'width', w_cur, 'subdivide_num', subdivide_num)

        # Use arange to generate row and column offsets for subregions (k*k,)
        row_offsets = torch.arange(k, device='cuda').view(-1, 1) * h_sub  # (k, 1)
        col_offsets = torch.arange(k, device='cuda').view(1, -1) * w_sub  # (1, k)

        # Calculate top-left (x1, y1) and bottom-right (x2, y2) for each subregion
        x1s = to_subdivide_regions_index[:, 0]  # (subdivide_num,)
        y1s = to_subdivide_regions_index[:, 1]  # (subdivide_num,)
        x1_sub = x1s.view(-1, 1, 1) + row_offsets  # (subdivide_num, k, 1)
        y1_sub = y1s.view(-1, 1, 1) + col_offsets  # (subdivide_num, 1, k)

        x2_sub = x1_sub + h_sub - 1  # (subdivide_num, k, 1)
        y2_sub = y1_sub + w_sub - 1  # (subdivide_num, 1, k)

        # Use broadcasting to expand to (subdivide_num, k, k) shape
        x1_sub = x1_sub.expand(-1, k, k)  # (subdivide_num, k, k)
        y1_sub = y1_sub.expand(-1, k, k)  # (subdivide_num, k, k)
        x2_sub = x2_sub.expand(-1, k, k)  # (subdivide_num, k, k)
        y2_sub = y2_sub.expand(-1, k, k)  # (subdivide_num, k, k)

        # Flatten (k, k) grid to (k*k,)
        x1_sub = x1_sub.reshape(subdivide_num, k * k)
        y1_sub = y1_sub.reshape(subdivide_num, k * k)
        x2_sub = x2_sub.reshape(subdivide_num, k * k)
        y2_sub = y2_sub.reshape(subdivide_num, k * k)

        # Concatenate into final index, shape is (subdivide_num, k*k, 4)
        subdivided_regions_index = torch.stack([x1_sub, y1_sub, x2_sub, y2_sub], dim=-1)  # (subdivide_num, k*k, 4)

        # Get index of subdivided regions to enter next loop iteration, for regions where con_res==1 and con_res==-1
        next_loop_regions_index = subdivided_regions_index[(con_res == 1) | (con_res == -1)]

        # next_loop_regions_num = next_loop_regions_index.shape[0]
        # print('next_loop_regions_num', next_loop_regions_num)

        # Update cur_regions_index here, shape is (next_loop_regions_num, 4)
        cur_regions_index = next_loop_regions_index.reshape(-1, 4)

        ##########Subdivision complete at this point
        ##########Next, complete storage

        # Next, add index of regions where con_res==1 and con_res==2, shape is (subdivide_save_num,)
        subdivide_save_regions_index = subdivided_regions_index[
            (con_res == 1) | (con_res == 2)]  # shape: (subdivide_save_num, k*k, 4)
        subdivide_save_num = subdivide_save_regions_index.shape[0]

        if subdivide_save_num == 0:
            # depth+=1
            continue

        subdivide_save_regions_index = subdivide_save_regions_index.reshape(-1, 4)  # shape: (subdivide_save_num*k*k, 4)

        # Add regions where con_res==2

        # print('cell num',subdivide_save_regions_index.shape[0])
        regions_final_ls.append(subdivide_save_regions_index)

        # Extract data for each region
        subdivide_save_regions, subdivide_save_labels = get_data([inputs, labels], subdivide_save_regions_index[:, 0],
                                                                 subdivide_save_regions_index[:, 1],
                                                                 subdivide_save_regions_index[:, 2],
                                                                 subdivide_save_regions_index[:, 3])
        # subdivide_save_regions shape: (subdivide_save_num*k*k, h_sub, w_sub, c)
        # subdivide_save_labels shape: (subdivide_save_num*k*k, h_sub, w_sub, c)

        # subdivide_save_num*k*k -> subdivide_save_num, k*k
        subdivide_save_regions = subdivide_save_regions.reshape(subdivide_save_num, k * k, h_sub, w_sub, c)
        subdivide_save_labels = subdivide_save_labels.reshape(subdivide_save_num, k * k, h_sub, w_sub, c)

        # Average over (h_sub, w_sub) region to get (subdivide_save_num, k*k, c) patch
        subdivide_save_patches = subdivide_save_regions.mean(dim=(2, 3))  # shape: (subdivide_save_num, k*k, c)
        subdivide_save_labels = subdivide_save_labels.mean(dim=(2, 3))

        # # Replace mean with L2 norm
        # subdivide_save_patches = torch.sqrt(
        #     (subdivide_save_regions ** 2).mean(dim=(2, 3)))  # shape: (subdivide_save_num, k*k, c)
        # subdivide_save_labels = torch.sqrt((subdivide_save_labels ** 2).mean(dim=(2, 3)))

        # Add position information to subdivide_save_patches, shape: (subdivide_save_num, k*k, c+3)
        # Position information is top-left x1 y1 depth
        subdivide_save_regions_index = subdivide_save_regions_index.reshape(subdivide_save_num, k * k, 4)
        x1 = subdivide_save_regions_index[:, :, 0].unsqueeze(-1)  # (subdivide_save_num, k*k, 1)
        y1 = subdivide_save_regions_index[:, :, 1].unsqueeze(-1)  # (subdivide_save_num, k*k, 1)
        depth_tensor = torch.full((subdivide_save_num, k * k, 1), depth, device='cuda')  # (subdivide_save_num, k*k, 1)
        subdivide_save_patches = torch.cat([subdivide_save_patches, depth_tensor, x1, y1],
                                           dim=-1)  # (subdivide_save_num, k*k, c+3)
        
        # Add position encoding to labels
        # Labels need same position encoding for proper alignment with inputs during loss computation
        subdivide_save_labels = torch.cat([subdivide_save_labels, depth_tensor, x1, y1],
                                         dim=-1)  # (subdivide_save_num, k*k, c+3)

        patches_final_ls.append(subdivide_save_patches)
        labels_final_ls.append(subdivide_save_labels)

        depth += 1

    # patches_final_ls->tensor
    regions_final = torch.cat(regions_final_ls, dim=0)
    patches_final = torch.cat(patches_final_ls, dim=0)
    labels_final = torch.cat(labels_final_ls, dim=0)
    # print(regions_final.shape, patches_final.shape, labels_final.shape)
    return regions_final, patches_final, labels_final

def visualize_quadtree(inputs: torch.tensor, regions: list, save_path: str = "quadtree_visualization"):
    fig, ax = plt.subplots()
    combined_image = torch.sum(inputs[0], dim=-1)
    cax = ax.imshow(combined_image, cmap='gray')

    # Draw rectangle for each region
    for region in regions:
        x, y, height, width = region
        rect = patches.Rectangle((x - 0.5, y - 0.5), width, height, linewidth=0.4, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    fig.colorbar(cax, ax=ax, orientation='vertical')

    # Generate random number and append to filename
    random_num = random.randint(1000, 9999)
    final_save_path = f"{save_path}_{random_num}.png"

    # Save image
    plt.savefig(final_save_path)
    print(f"Image saved to {final_save_path}")
    plt.close(fig)  # Close image to free memory

def visualize_patches(inputs: torch.tensor, patches_ls: list, k: int, max_size: int,
                      save_path: str = "output_image.png"):
    depth_dict = {}

    # Process each patch_set and calculate patch size
    for patch_set in patches_ls:
        # print(patch_set.shape)
        x, y, depth = patch_set[0, 0, -3:].int().tolist()

        if depth not in depth_dict:
            depth_dict[depth] = []

        # Calculate current patch size based on depth
        patch_size = max_size // (k ** depth)

        # Store patch and its size information in depth_dict
        depth_dict[depth].append((x, y, patch_set[:, :, :-3], patch_size))

    # Draw separate image for each depth
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    combined_image = torch.sum(inputs, dim=-1)
    cax = ax.imshow(combined_image, cmap='gray')

    # Draw in order of patch depth, ensuring larger patches are drawn first
    for depth in sorted(depth_dict.keys()):
        patches_at_depth = depth_dict[depth]

        for (x, y, features, patch_size) in patches_at_depth:
            # Iterate through each k*k subregion within the patch, display separately
            for i in range(k):
                for j in range(k):
                    sub_x = x + i * patch_size
                    sub_y = y + j * patch_size
                    feature_value = features[j, i].mean().item()

                    # Draw color block for each subregion
                    rect = patches.Rectangle((sub_x - 0.5, sub_y - 0.5), patch_size, patch_size,
                                             linewidth=0.4, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                    # Display feature value at center of color block
                    ax.text(sub_x + patch_size / 2 - 0.5, sub_y + patch_size / 2 - 0.5, f'{feature_value:.2f}',
                            color='white', fontsize=12 / math.log2(depth + 2), ha='center', va='center')

    fig.colorbar(cax, ax=ax, orientation='vertical')
    plt.title('Visualization of Patches with Features')

    # Generate random number and append to filename
    random_num = random.randint(1000, 9999)
    final_save_path = f"{save_path}_{random_num}.png"

    # Save image
    plt.savefig(final_save_path)
    print(f"Image saved to {final_save_path}")
    plt.close(fig)  # Close image to free memory

def reconstruct_image(inputs_shape: tuple,
                      patches_final: torch.tensor,
                      max_size: int,
                      k: int,
                      is_residual: bool = False
                      ):
    '''
    inputs_shape shape: (h, w, c)
    patches_final shape: (N, k, k, c+3)
    max_size: int
    k: int
    return img_reconstructed shape: (h, w, c)
    '''

    # The last three dimensions of patches are depth, x, y
    depth = patches_final[:, 0, 0, -3]  # (N,)
    x = patches_final[..., -2]  # (N, k, k)
    y = patches_final[..., -1]  # (N, k, k)

    # The metadata of patches is from the beginning to the fourth from last
    data = patches_final[..., :-3]  # (N, k, k, c)

    # Calculate current patch size based on depth
    cell_size = int(max_size) // (k ** (depth + 1))  # (N,)
    min_depth = int(depth.min().item())
    max_depth = int(depth.max().item())

    # Initialize an image, same size as inputs, initial value is NaN
    img_reconstructed = torch.full(inputs_shape, float('nan')).to(patches_final.device)  # (h, w, c)

    for cur_depth in range(min_depth, max_depth + 1):
        cur_depth_mask = (depth == cur_depth)

        cur_data = data[cur_depth_mask]  # (N, k, k, c)
        cur_x1 = x[cur_depth_mask].flatten().long()  # (N*k*k,)
        cur_y1 = y[cur_depth_mask].flatten().long()  # (N*k*k,)
        cur_data = rearrange(cur_data, 'n h w c -> (n h w) c')  # (N*k*k, c)

        cur_cell_size = cell_size[cur_depth_mask]
        cur_cell_size = int(cur_cell_size[0].item())  # Patch size is the same for the same layer

        # Generate 2D index grid
        grid_x, grid_y = torch.meshgrid(
            torch.arange(cur_cell_size),
            torch.arange(cur_cell_size)
        )
        grid_x = grid_x.flatten().cuda()
        grid_y = grid_y.flatten().cuda()

        # Add grid offset to top-left coordinate of each cell
        full_x = cur_x1[:, None] + grid_x  # (N*k*k, cur_cell_size^2)
        full_y = cur_y1[:, None] + grid_y  # (N*k*k, cur_cell_size^2)

        # Flatten offset indices to fit img_reconstructed size
        full_x = full_x.flatten()
        full_y = full_y.flatten()

        # Update image using advanced indexing with full_x and full_y
        if is_residual:
            img_reconstructed[full_x, full_y, :] += cur_data.repeat_interleave(cur_cell_size ** 2, dim=0)
        else:
            img_reconstructed[full_x, full_y, :] = cur_data.float().repeat_interleave(cur_cell_size ** 2, dim=0)

    return img_reconstructed

def visualize_patches_by_color(inputs: torch.tensor,
                               patches_tensor: torch.tensor,
                               k: int,
                               max_size: torch.tensor,
                               title: str = 'Visualization of Patches with Black and White Gradient',
                               sum_channels: bool = True,
                               cmap_name: str = 'viridis'
                               ):
    img = reconstruct_image(inputs.shape, patches_tensor, max_size, k)  # (h, w, c)

    # to cpu
    img = img.cpu().detach()
    data = patches_tensor[..., :-3]  # (N, k, k, c)
    cur_min = data.min().item()
    cur_max = data.max().item()

    # Plot setup
    fig, ax = plt.subplots(figsize=(30, 30), dpi=50)

    if sum_channels:
        # Sum across channels and display as grayscale
        img = img.sum(dim=-1)
        img = (img - cur_min) / (cur_max - cur_min)
        cax = ax.imshow(img, cmap=cmap_name)
        fig.colorbar(cax, ax=ax, orientation='vertical')
    else:
        # Display each channel independently in grayscale
        for i in range(img.shape[-1]):
            plt.figure(figsize=(5, 5), dpi=100)
            plt.imshow(img[:, :, i], cmap=cmap_name, vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f"{title} - Channel {i+1}")
            save_path = f'../temp/{title}_channel_{i+1}.png'
            plt.savefig(save_path)
            print(f"Image for channel {i+1} saved to {save_path}")
            plt.show()

    # Save the summed image if sum_channels is True
    if sum_channels:
        plt.title(title)
        save_path = f'../temp/{title}.png'
        plt.savefig(save_path)
        print(f"Image saved to {save_path}")

def visualize_patches_by_color_beta(inputs_shape,
                               patches_tensor: torch.tensor,
                               title: str = 'Visualization of Patches with Black and White Gradient',
                               sum_channels: bool = True,
                               cmap_name: str = 'viridis'
                               ):
    
    img = reconstruct_amr_tokens(inputs_shape, patches_tensor, False)  # (h, w, c)

    # to cpu
    img = img.cpu().detach()
    data = patches_tensor[..., :-4]  # (N, k, k, c)
    cur_min = data.min().item()
    cur_max = data.max().item()

    # Plot setup
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

    if sum_channels:
        # Sum across channels and display as grayscale
        img = img.sum(dim=-1)
        img = (img - cur_min) / (cur_max - cur_min)
        cax = ax.imshow(img, cmap=cmap_name)
        fig.colorbar(cax, ax=ax, orientation='vertical')
    else:
        # Display each channel independently in grayscale
        for i in range(img.shape[-1]):
            plt.figure(figsize=(5, 5), dpi=100)
            plt.imshow(img[:, :, i], cmap=cmap_name, vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f"{title} - Channel {i+1}")
            save_path = f'../temp/{title}_channel_{i+1}.png'
            plt.savefig(save_path)
            print(f"Image for channel {i+1} saved to {save_path}")
            plt.show()

    # Save the summed image if sum_channels is True
    if sum_channels:
        plt.title(title)
        save_path = f'../temp/{title}.png'
        plt.savefig(save_path)
        print(f"Image saved to {save_path}")

def visualize_patches_by_color_with_borders(inputs: torch.tensor,
                                            patches_tensor: torch.tensor,
                                            k: int,
                                            max_size: torch.tensor,
                                            title: str = 'Visualization of Patches with Borders',
                                            sum_channels: bool = True,
                                            cmap_name: str = 'viridis',
                                            border_color: str = 'black',  # Set border color
                                            border_thickness: int = 0.15        # Set border thickness
                                            ):
    img = reconstruct_image(inputs.shape, patches_tensor, max_size, k)  # (h, w, c)

    # to cpu
    img = img.cpu().detach()
    data = patches_tensor[..., :-3]  # (N, k, k, c)
    cur_min = data.min().item()
    cur_max = data.max().item()

    # Plot setup
    fig, ax = plt.subplots(figsize=(30, 30), dpi=300)

    if sum_channels:
        # Sum across channels and display as grayscale
        img = img.sum(dim=-1)
        img = (img - cur_min) / (cur_max - cur_min)
        cax = ax.imshow(img, cmap=cmap_name, vmin=0, vmax=1.4)
        fig.colorbar(cax, ax=ax, orientation='vertical')
    else:
        # Display each channel independently in grayscale
        for i in range(img.shape[-1]):
            plt.figure(figsize=(5, 5), dpi=100)
            plt.imshow(img[:, :, i], cmap=cmap_name, vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f"{title} - Channel {i+1}")
            save_path = f'../temp/{title}_channel_{i+1}.png'
            plt.savefig(save_path)
            print(f"Image for channel {i+1} saved to {save_path}")
            plt.show()

    # Draw borders for each patch cell
    depth = patches_tensor[:, 0, 0, -3]  # (N,)
    x = patches_tensor[..., -2]  # (N, k, k)
    y = patches_tensor[..., -1]  # (N, k, k)

    # Iterate over patches and draw borders
    for cur_depth in range(int(depth.min().item()), int(depth.max().item()) + 1):
        cur_depth_mask = (depth == cur_depth)

        cur_x1 = x[cur_depth_mask].flatten().long()  # (N*k*k,)
        cur_y1 = y[cur_depth_mask].flatten().long()  # (N*k*k/)
        
        # Compute the cell size directly for the current depth
        cur_cell_size = int(max_size // (k ** (cur_depth + 1)))  # Direct integer division

        # Plot the borders of each patch cell
        for cx, cy in zip(cur_x1, cur_y1):
            rect = Rectangle((cy.item()-0.5, cx.item()-0.5), cur_cell_size, cur_cell_size, 
                             linewidth=border_thickness, edgecolor=border_color, facecolor='none')
            ax.add_patch(rect)

    # Save the final image
    plt.title(title)
    save_path = f'../temp/{title}_with_borders.png'
    plt.savefig(save_path)
    print(f"Image with borders saved to {save_path}")

from matplotlib.patches import Rectangle
def visualize_individual_patches(patches_tensor: torch.tensor,
                                 inputs_shape: tuple,
                                 k: int,
                                 max_size: int,
                                 title_prefix: str = 'Patch Visualization',
                                 cmap_name: str = 'viridis',
                                 border_color: str = 'black',
                                 border_thickness: int = 0.15):
    '''
    patches_tensor shape: (N, k, k, c+3)
    max_size: int
    k: int
    Visualizes each depth independently with patches deployed according to position.
    '''
    depth = patches_tensor[:, 0, 0, -3]  # (N,)
    data = patches_tensor[..., :-3]  # (N, k, k, c)
    cur_min = data.min().item()
    cur_max = data.max().item()
    x = patches_tensor[..., -2]  # (N, k, k)
    y = patches_tensor[..., -1]  # (N, k, k)

    unique_depths = sorted(depth.unique().tolist())  # Get all unique depth values and sort

    for cur_depth in unique_depths:
        cur_depth_mask = (depth == cur_depth)
        cur_patches = patches_tensor[cur_depth_mask]  # Filter patches for current depth

        # Use reconstruct_image to stitch patches of current depth into complete image
        img_reconstructed = reconstruct_image(inputs_shape, cur_patches, max_size, k)

        # Normalize img_reconstructed data and convert to float32 type
        img_reconstructed = img_reconstructed.cpu().detach().sum(dim=-1).float()  # Sum channel data and convert to float32
        img_reconstructed = (img_reconstructed - cur_min) / (cur_max - cur_min)  # Min-Max normalization

        # Visualize and save stitched image for current depth
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        cax = ax.imshow(img_reconstructed, cmap=cmap_name, vmin=0, vmax=1.6)
        # fig.colorbar(cax, ax=ax, orientation='vertical')

        # Draw patch boundaries for shallower and current layers
        for depth_level in unique_depths:
            if depth_level > cur_depth:  # Only draw patch boundaries less than or equal to current depth
                break

            depth_mask = (depth == depth_level)
            depth_x = x[depth_mask].flatten().long()
            depth_y = y[depth_mask].flatten().long()
            cell_size = int(max_size // (k ** (int(depth_level) + 1)))  # Calculate cell size for current depth

            for cx, cy in zip(depth_x, depth_y):
                rect = Rectangle((cy.item() - 0.5, cx.item() - 0.5), cell_size, cell_size, 
                                 linewidth=border_thickness, edgecolor=border_color, facecolor='none')
                ax.add_patch(rect)

        # Add title and save image
        # plt.title(f"{title_prefix} - Depth {int(cur_depth)}")
        # Don't display axes
        plt.axis('off')
        save_path = f'../temp/{title_prefix}_depth_{int(cur_depth)}_with_borders.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        print(f"Image with borders for depth {int(cur_depth)} saved to {save_path}")

def visualize_individual_patches_(patches_tensor: torch.tensor,
                                 inputs_shape: tuple,
                                 k: int,
                                 max_size: int,
                                 title_prefix: str = 'Patch Visualization',
                                 cmap_name: str = 'viridis'):
    '''
    patches_tensor shape: (N, k, k, c+3)
    max_size: int
    k: int
    Visualizes each depth independently with patches deployed according to position.
    '''
    depth = patches_tensor[:, 0, 0, -3]  # (N,)
    data = patches_tensor[..., :-3]  # (N, k, k, c)
    cur_min = data.min().item()
    cur_max = data.max().item()

    for cur_depth in depth.unique():
        cur_depth_mask = (depth == cur_depth)
        cur_patches = patches_tensor[cur_depth_mask]  # Filter patches for current depth

        # Use reconstruct_image to stitch patches of current depth into complete image
        img_reconstructed = reconstruct_image(inputs_shape, cur_patches, max_size, k)

        # Normalize img_reconstructed data and convert to float32 type
        img_reconstructed = img_reconstructed.cpu().detach().sum(dim=-1).float()  # Sum channel data and convert to float32
        img_reconstructed = (img_reconstructed - cur_min) / (cur_max - cur_min)  # Min-Max normalization

        # Visualize and save stitched image for current depth
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        cax = ax.imshow(img_reconstructed, cmap=cmap_name, vmin=0, vmax=1.6)
        # cax.set_clim(0, 1)  # Use 0 and 1 as colormap range
        fig.colorbar(cax, ax=ax, orientation='vertical')
        plt.title(f"{title_prefix} - Depth {int(cur_depth.item())}")
        save_path = f'../temp/{title_prefix}_depth_{int(cur_depth.item())}.png'
        plt.savefig(save_path)
        plt.show()
        print(f"Image for depth {int(cur_depth.item())} saved to {save_path}")

def visualize_individual_patches__(patches_tensor: torch.tensor,
                                   k: int,
                                   max_size: int,
                                   title_prefix: str = 'Patch Visualization',
                                   cmap_name: str = 'viridis'):
    '''
    patches_tensor shape: (N, k, k, c+3)
    max_size: int
    k: int
    Visualizes each patch independently with normalization.
    '''
    # The last three dimensions of patches are depth, x, y
    depth = patches_tensor[:, 0, 0, -3]  # (N,)
    x = patches_tensor[..., -2]  # (N, k, k)
    y = patches_tensor[..., -1]  # (N, k, k)

    # The metadata of patches is from the beginning to the fourth from last
    data = patches_tensor[..., :-3]  # (N, k, k, c)
    cur_min = data.min().item()
    cur_max = data.max().item()
    # Calculate current patch size based on depth
    cell_size = max_size // (k ** (depth + 1))  # (N,)

    for idx, cur_depth in enumerate(depth.unique()):
        cur_depth_mask = (depth == cur_depth)
        cur_data = data[cur_depth_mask]  # (N, k, k, c)

        # Loop through each patch, generate separate image
        for i, patch_data in enumerate(cur_data):
            # Normalize data
            patch_data = patch_data.cpu().detach().numpy()
            patch_img = patch_data.sum(axis=-1)  # Sum channel data
            patch_img = (patch_img - cur_min) / (cur_max - cur_min)  # Min-Max normalization
            # print(patch_img.min(), patch_img.max())

            fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
            cax = ax.imshow(patch_img, cmap=cmap_name, vmin=0, vmax=1.6)
            fig.colorbar(cax, ax=ax, orientation='vertical')
            plt.title(f"{title_prefix} - Depth {int(cur_depth.item())} - Patch {i}")
            save_path = f'../temp_vispatch/patch{int(cur_depth.item())}_{i}.png'
            plt.savefig(save_path)
            # plt.show()
            print(f"Patch image saved to {save_path}")
            

def visualize_patches_by_color___(inputs: torch.tensor,
                                  patches_tensor: torch.tensor,
                                  k: int,
                                  max_size: torch.tensor,
                                  title: str = 'Visualization of Patches with Black and White Gradient'
                                  ):
    inputs = inputs.cpu()
    patches_tensor = patches_tensor.cpu()
    max_size = max_size.item()
    # k = k.item()
    depth_dict = {}

    # Process each patch_set and calculate patch size
    for patch_set in patches_tensor:
        depth, x, y = patch_set[0, 0, -3:].int().tolist()

        if depth not in depth_dict:
            depth_dict[depth] = []

        # Calculate current patch size based on depth
        patch_size = max_size // (k ** (depth + 1))
        # print('depth', depth, 'x', x, 'y', y, 'patch_size', patch_size)

        # Store patch and its size information in depth_dict
        depth_dict[depth].append((x, y, patch_set[:, :, :-3], patch_size))

    # Draw separate image for each depth
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    # combined_image = torch.sum(inputs, dim=-1) # (h, w)
    combined_image = torch.zeros_like(inputs).sum(dim=-1).cpu()
    cax = ax.imshow(combined_image, cmap='gray')

    # Get grayscale colormap
    cmap = cm.get_cmap('gray')

    # Draw in order of patch depth, ensuring larger patches are drawn first
    for depth in sorted(depth_dict.keys()):
        patches_at_depth = depth_dict[depth]

        for (x, y, features, patch_size) in patches_at_depth:
            # Iterate through each k*k subregion within the patch, display separately
            for i in range(k):
                for j in range(k):
                    sub_x = x + i * patch_size
                    sub_y = y + j * patch_size
                    feature_value = features[i, j].mean().item()

                    # Map feature value to grayscale color, ensure RGBA value is returned
                    # normalized_value = feature_value / features.max().item()
                    color = cmap(feature_value)

                    # Draw color block for each subregion and fill with color
                    rect = patches.Rectangle((sub_y - 0.5, sub_x - 0.5), patch_size, patch_size,
                                             linewidth=0.8, edgecolor='r', facecolor=color)
                    ax.add_patch(rect)

    fig.colorbar(cax, ax=ax, orientation='vertical')
    plt.title(title)
    plt.show()

    save_path = f'../temp/save_{title}.png'
    plt.savefig(save_path)


def show_regions_by_size(inputs: torch.tensor, regions: list):
    region_dict = {}
    for region in regions:
        x, y, height, width = region
        size = (height, width)
        if size not in region_dict:
            region_dict[size] = []
        region_dict[size].append(region)

    for size, region_list in region_dict.items():
        fig, ax = plt.subplots()
        combined_image = inputs[0, :, :, 0] + inputs[0, :, :, 1]
        cax = ax.imshow(combined_image, cmap='gray')

        for region in region_list:
            x, y, height, width = region
            rect = patches.Rectangle((x - 0.5, y - 0.5), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        fig.colorbar(cax, ax=ax, orientation='vertical')
        plt.title(f"Regions of size {size}")
        plt.show()


if __name__ == '__main__':
    # Example call
    inputs = torch.randn(1, 18, 23, 2)  # 18, 23
    inputs[(inputs < 1)] = 0
    fill_value = 0.0
    # k-ary tree partitioning
    k = 2

    max_size = random.randint(1, int(math.sqrt(inputs.shape[1] // k)))
    min_size = random.randint(0, max_size)
    # max_size = k ** max_size
    # min_size = k ** min_size
    max_size = 16
    min_size = 1
    print("max_size: ", max_size)
    print("min_size: ", min_size)

    inputs = pad_to_power_of_k(inputs, fill_value, k)
    mask = torch.ones(inputs.shape)

    start_time = time.time()
    regions, patches_ls, labels_ls = quadtree_partition_parallel(inputs, inputs, mask, inputs, k=k, min_size=min_size,
                                                                 max_size=max_size
                                                                 , threshold_rate=0.1, nonzero_ratio_threshold=0.99,
                                                                 condition_type='either', max_depth=int(
            math.log(max_size / min_size, k * k)) + 1)
    print(f"partition time cost: {time.time() - start_time}")
    print(regions)
    print(len(patches_ls))
    regions = regions[0]
    patches_ls = patches_ls[0]

    # Output partitioning results
    # s = 0
    # for region in regions:
    #     print(f"Region: x={region[0]}, y={region[1]}, height={region[2]}, width={region[3]}")
    #     s += region[2] * region[3]
    # print(s, len(regions))

    # Visualize quadtree partitioning
    start_time = time.time()
    visualize_quadtree(inputs, regions)
    visualize_patches(inputs, patches_ls, k, max_size)
    visualize_patches_by_color(inputs, patches_ls, k, max_size)
    print(f"visualize time cost: {time.time() - start_time}")

    # new_region_index = k_maxary_tree_partition(inputs, regions, k, max_size, min_size)
    # new_region_index = new_region_index[0]

    # s = 0
    # for region in new_regions:
    #     print(f"new_regions: x={region[0]}, y={region[1]}, height={region[2]}, width={region[3]}")
    #     s += region[2] * region[3]
    # print(s, len(new_regions))

    # visualize_quadtree(inputs, new_region_index)
    # show_regions_by_size(inputs, new_region_index)
    # # new_regions.pop(0)
    # print(len(new_region_index))# print(new_regions)
    # print(len(set(new_region_index)))
    #
    #

    # def k_maxary_tree_partition(inputs: torch.tensor, regions: list, k: int = 2, max_size: int = 2, min_size: int = 2):
    #     b, h_, w_, c = inputs.shape
    #
    #     # Check if interval contains smaller cells
    #     def is_within_region(rx, ry, rwidth, rheight, x, y, width, height):
    #         # Parameters: small region, large region
    #         if x == rx and y == ry and width == rwidth and height == rheight:
    #             # If two regions are same size, return false
    #             return False
    #         else:
    #             # If large region contains small region, return true
    #             return (x <= rx < x + width and x <= rx + rwidth <= x + width and
    #                     y <= ry < y + height and y <= ry + rheight <= y + height)
    #
    #     def partition(region: torch.tensor, x: int, y: int, height: int, width: int) -> list:
    #         results_ = []
    #         if height < min_size or width < min_size:
    #             # If current region size is smaller than preset size, no longer subdivide current region
    #             return []
    #
    #         sub_regions = [(x, y, height, width)]
    #
    #         # If current region can be subdivided, set sub_regions to empty
    #         for reg in regions:
    #             rx, ry, rheight, rwidth = reg
    #             if is_within_region(rx, ry, rwidth, rheight, x, y, width, height):
    #                 sub_regions = []
    #                 break
    #
    #         # If current region cannot be subdivided and current region size is reasonable, directly return current region
    #         if len(sub_regions) > 0:
    #             if min_size <= height <= max_size and min_size <= width <= max_size:
    #                 return sub_regions
    #
    #         # If current region can be subdivided, further subdivide the region
    #         if min_size <= height <= max_size and min_size <= width <= max_size:
    #             results_.append((x, y, height, width))
    #
    #         k_height = height // k
    #         k_width = width // k
    #
    #         for i in range(k):
    #             for j in range(k):
    #                 # Determine new bottom-left corner (origin is at bottom-left)
    #                 sx = x + i * k_width
    #                 sy = y + j * k_height
    #                 if sx + k_width <= w_ and sy + k_height <= h_ and k_height > 0 and k_width > 0:
    #                     res = partition(region, sx, sy, k_height, k_width)
    #                     if res is not None:
    #                         results_.extend(res)
    #         return results_
    #
    #     results = []
    #     for batch in range(b):
    #         res = partition(inputs[batch], 0, 0, h_, w_)
    #         if res is not None:
    #             results.append(res)
    #     return results
    #
    #
    # def k_ary_tree_partition(inputs: torch.tensor, regions: list, k: int = 2):
    #     b, h_, w_, c = inputs.shape
    #
    #     def is_within_region(rx, ry, rwidth, rheight, x, y, width, height):
    #         if x == rx and y == ry and width == rwidth and height == rheight:
    #             return False
    #         else:
    #             return (x <= rx < x + width and x <= rx + rwidth <= x + width and
    #                     y <= ry < y + height and y <= ry + rheight <= y + height)
    #
    #     def partition(region: torch.tensor, x: int, y: int, height: int, width: int) -> list:
    #         results_ = []
    #         if height <= 1 or width <= 1:
    #             return [(x, y, height, width)]
    #
    #         sub_regions = [(x, y, height, width)]
    #         for reg in regions:
    #             rx, ry, rheight, rwidth = reg
    #             if is_within_region(rx, ry, rwidth, rheight, x, y, width, height):
    #                 sub_regions = []
    #                 break
    #
    #         if len(sub_regions) > 0:
    #             return sub_regions
    #
    #         results_.append((x, y, height, width))
    #
    #         k_height = height // k
    #         k_width = width // k
    #
    #         for i in range(k):
    #             for j in range(k):
    #                 sx = x + i * k_width
    #                 sy = y + j * k_height
    #                 if sx + k_width <= w_ and sy + k_height <= h_ and k_height > 0 and k_width > 0:
    #                     res = partition(region, sx, sy, k_height, k_width)
    #                     if res is not None:
    #                         results_.extend(res)
    #         return results_
    #
    #     results = []
    #     for batch in range(b):
    #         res = partition(inputs[batch], 0, 0, h_, w_)
    #         if res is not None:
    #             results.append(res)
    #     return results

    # def transform_to_patches(region_index: list, region_data: torch.tensor, k: int):
    #     # region_index shape: B, N, 4
    #     #
    #
    #     # Calculate maximum width and height for different sizes
    #     unique_sizes = set((d[2], d[3]) for d in region_index)
    #
    #     patches_dict = defaultdict(list)
    #
    #     # Fill according to size
    #     for size in unique_sizes:
    #         size_data = [d for d in region_index if (d[2], d[3]) == size]
    #
    #         num_patches = len(size_data) // (k * k)
    #         if num_patches * k * k != len(size_data):
    #             raise ValueError("Data length must be a multiple of k*k")
    #
    #         max_width, max_height = size
    #         for patch_idx in range(num_patches):
    #             patch = torch.zeros((k, k, max_width, max_height, 2))
    #             for i in range(k):
    #                 for j in range(k):
    #                     cell_idx = patch_idx * (k * k) + i * k + j
    #                     x, y, width, height = size_data[cell_idx][:4]
    #                     patch[i, j, :width, :height, 0] = x
    #                     patch[i, j, :width, :height, 1] = y
    #             patches_dict[size].append(patch)
    #
    #     # Convert list to tensor
    #     for size in patches_dict:
    #         patches_dict[size] = torch.stack(patches_dict[size])
    #
    #     return patches_dict
    #
    # # Convert to patches
    # patches_dict = transform_to_patches(new_region_index, inputs, k)
    #
    # min_size = min(patches_dict.keys())
    # max_size = max(patches_dict.keys())
    #
    # sorted_patches = list(sorted(patches_dict.items(), key=lambda item: (item[0][0], item[0][1]), reverse=True))
    #
    # for i, (size, patches) in enumerate(sorted_patches):
    #     print(f"Patches of size {size}: shape = {patches.shape}")
    #
    # min_size = min_size[0]
    #
    # device = 'cuda'
    #
    # patches_tensor_ls = [torch.tensor(i[1], device=device) for i in sorted_patches]
    #
    # in_dim = patches_tensor_ls[0].shape[-1]
    # out_dim = in_dim
    #
    # # [k, k, k^0, k^0, dim], [k, k, k^1, k^1, dim], [k, k, k^2, k^2, dim] --> [k, k, 1, 1, dim] token --> transformer --> [h, w, C] --> MSE
    # # Parallelize
    #
    # # [N, k, k, k^i, k^i,dim] i=0,1,2,... -> [N, k, k, k^min(i), k^min(i), dim]
    # # for i = 0,1,2,..
    # # -> [N*k*k, dim, k^i, k^i]
    # # -> [N*k*k, dim, k^(i-1), k^(i-1)]
    # # ..
    # # -> [N*k*k, dim, k^min(i), k^min(i)]
    # # -> [N, k, k, k^min(i), k^min(i), dim]
    #
    # # [N, k, k, k^i, k^i,dim] i=0,1,2,... -> [N, k, k, k^max(i), k^max(i), dim]
    # # Find maximum patch
    # max_size_patch = None
    # max_size = max(max([patch.shape[2] for patch in patches_dict.values()]), max([patch.shape[3] for patch in patches_dict.values()]))
    #
    # # Assume patches_dict stores patches of different sizes
    # for size, patches in patches_dict.items():
    #     if size[0] == max_size and size[1] == max_size:
    #         max_size_patch = patches
    #         break
    #
    #
    # def apply_mflod(patch, frequencies, phases):
    #     """
    #     Apply multi-scale Fourier modulation to patch, converting it to a more detailed representation.
    #     patch: Input of shape [k, k, k^i, k^i, dim]
    #     frequencies: Fourier transform frequency parameters
    #     phases: Fourier transform phase parameters
    #     """
    #     # Use MFLOD for Fourier transform and generate LOD features
    #     transformed_patch = torch.sin(frequencies * patch + phases)
    #     return transformed_patch
    #
    # def integrate_details(base_patch, small_patch):
    #     """
    #     Attach details of small patch to base patch using Fourier fusion.
    #     base_patch: Largest base patch
    #     small_patch: Smaller patch
    #     """
    #     frequencies = torch.randn(base_patch.shape[-1])  # Frequency parameters, dimension matches feature dimension
    #     phases = torch.randn(base_patch.shape[-1])       # Phase parameters
    #
    #     # Apply MFLOD to both patches
    #     base_patch_lod = apply_mflod(base_patch, frequencies, phases)
    #     small_patch_lod = apply_mflod(small_patch, frequencies, phases)
    #
    #     # Fuse details using element-wise multiplication or weighted sum
    #     integrated_patch = base_patch_lod + small_patch_lod
    #     return integrated_patch
    #
    # def enhance_patch_with_lod(patches_dict):
    #     """
    #     Select the largest patch and use LOD information from small patches for detail enhancement.
    #     patches_dict: A dictionary containing patches of different scales.
    #     """
    #     # Find largest patch
    #     max_size = max(patches_dict.keys())
    #     base_patch = patches_dict[max_size]
    #
    #     # Fuse all overlapping small patches
    #     for size, small_patches in patches_dict.items():
    #         if size != max_size:
    #             for small_patch in small_patches:
    #                 base_patch = integrate_details(base_patch, small_patch)
    #
    #     return base_patch
    #
    # # Example call
    # # Assume input is a patch dictionary where keys are patch dimensions and values are corresponding patch tensors
    # patches_dict = {
    #     (2, 2): torch.randn(256, 2, 2, 2, 2, 2),  # [num_patches, k, k, k^i, k^i, dim]
    #     (1, 1): torch.randn(249, 2, 2, 1, 1, 2)
    # }
    #
    # # Enhance details of largest patch
    # enhanced_patch = enhance_patch_with_lod(patches_dict)
    # print(f"Enhanced Patch Shape: {enhanced_patch.shape}")
    #
    #
    #
    # class PatchConvolver(nn.Module):
    #     def __init__(self, kernel_size, in_c, out_c):
    #         super(PatchConvolver, self).__init__()
    #         self.kernel_size = kernel_size
    #         self.conv = nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size, stride=self.kernel_size)
    #         self.out_c = out_c
    #
    #     def forward(self, patches_tensor_ls, min_size):
    #         # patches_tensor_ls: [N, k, k, k^i, k^i,dim]
    #
    #         min_size = min_size
    #
    #         out_ls = []
    #         for patches_tensor in patches_tensor_ls:
    #             N, k, _, w, h, dim = patches_tensor.shape
    #             patches_tensor_reshape = patches_tensor.permute(0,1,2,5,3,4).reshape(N*k*k, dim, w,h,) # [N*k*k, dim, k^i, k^i]
    #             # print(patches_tensor_reshape.shape)
    #
    #             out = patches_tensor_reshape
    #             while 1:
    #                 if out.shape[-1] == min_size:
    #                     out_ls.append(out)
    #                     break
    #                 out = self.conv(out)
    #
    #         out_tensor = torch.cat(out_ls, dim=0) # list of [N*k*k, dim, k^min(i), k^min(i)]
    #
    #         N = out_tensor.shape[0] // (self.kernel_size * self.kernel_size)
    #
    #         out_tensor = out_tensor.reshape(N, self.kernel_size * self.kernel_size, self.out_c, min_size, min_size) # list of [N, k*k, dim, k^min(i), k^min(i)]
    #
    #         print(out_tensor.shape)
    #
    #         return out_tensor
    #
    # net = PatchConvolver(k, in_dim, out_dim)
    # net = net.to(device)
    # out = net(patches_tensor_ls, min_size)
    #
    # torch.Size([5, 4, 2, 16, 16])
    # torch.Size([489, 4, 2, 1, 1])
    # torch.Size([135, 4, 2, 2, 2])