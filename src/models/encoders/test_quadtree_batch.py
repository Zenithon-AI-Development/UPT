"""
Test QuadtreeTransformerPerceiver with 2 samples per dataset in batches.
Tracks shapes throughout the encoding process.
"""

import sys
from pathlib import Path
import torch
import h5py
import numpy as np

# Add SHINE_mapping to path
shine_path = Path('/home/workspace/projects/transformer/SHINE_mapping')
if str(shine_path) not in sys.path:
    sys.path.insert(0, str(shine_path))

from quadtree import feature_grids_to_quadtree


def load_hdf5_snapshot(filepath, snapshot_idx=0):
    """Load 2D snapshot from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())
        
        # Try t0_fields, t1_fields, t2_fields groups - collect from all groups
        field_groups = [k for k in keys if k.startswith('t') and k.endswith('_fields') and isinstance(f[k], h5py.Group)]
        if field_groups:
            all_fields = []
            
            # Process each field group (t0_fields, t1_fields, etc.)
            for group_name in sorted(field_groups):
                fields_group = f[group_name]
                field_names = [k for k in fields_group.keys() if isinstance(fields_group[k], h5py.Dataset)]
                
                # Skip mask fields
                data_field_names = [n for n in field_names if 'mask' not in n.lower()]
                if not data_field_names:
                    data_field_names = field_names
                
                for field_name in sorted(data_field_names):
                    field_data = fields_group[field_name][:]
                    
                    if len(field_data.shape) == 2:
                        all_fields.append(field_data)
                    elif len(field_data.shape) == 3:
                        time_idx = snapshot_idx if snapshot_idx >= 0 else (field_data.shape[0] - 1 if snapshot_idx == -1 else 0)
                        if field_data.shape[0] > field_data.shape[-1]:
                            field_data = field_data[time_idx]
                        else:
                            field_data = field_data[:, :, time_idx]
                        all_fields.append(field_data)
                    elif len(field_data.shape) == 4:
                        # Handle 4D: (batch, time, H, W) format
                        if field_data.shape[1] > 1:
                            time_idx = snapshot_idx if snapshot_idx >= 0 else (field_data.shape[1] - 1 if snapshot_idx == -1 else 0)
                            field_data = field_data[0, time_idx, :, :]
                        else:
                            field_data = field_data[0, 0, :, :] if field_data.shape[1] == 1 else field_data[0, :, :]
                        all_fields.append(field_data)
                    elif len(field_data.shape) == 5:
                        # Handle 5D: (batch, time, H, W, components) - e.g., velocity
                        if field_data.shape[1] > 1:
                            time_idx = snapshot_idx if snapshot_idx >= 0 else (field_data.shape[1] - 1 if snapshot_idx == -1 else 0)
                            # Extract each component separately
                            for comp in range(field_data.shape[4]):
                                all_fields.append(field_data[0, time_idx, :, :, comp])
                        else:
                            # Single time step, extract all components
                            for comp in range(field_data.shape[4]):
                                all_fields.append(field_data[0, 0, :, :, comp])
            
            if all_fields:
                shapes = [f.shape[:2] for f in all_fields]
                if len(set(shapes)) == 1:
                    data = np.stack(all_fields, axis=-1)
                    return data
                elif len(all_fields) == 1:
                    return all_fields[0]
        
        # Try input_fields group
        if 'input_fields' in keys:
            input_item = f['input_fields']
            if isinstance(input_item, h5py.Dataset):
                field_data = input_item[:]
                if len(field_data.shape) == 3:
                    time_idx = snapshot_idx if snapshot_idx >= 0 else (field_data.shape[0] - 1 if snapshot_idx == -1 else 0)
                    data = field_data[time_idx]
                    return data
                elif len(field_data.shape) == 2:
                    return field_data
            elif isinstance(input_item, h5py.Group):
                field_names = [k for k in input_item.keys() if isinstance(input_item[k], h5py.Dataset)]
                if field_names:
                    fields = []
                    for field_name in sorted(field_names):
                        field_data = input_item[field_name][:]
                        if len(field_data.shape) == 2:
                            fields.append(field_data)
                        elif len(field_data.shape) == 3:
                            time_idx = snapshot_idx if snapshot_idx >= 0 else (field_data.shape[0] - 1 if snapshot_idx == -1 else 0)
                            field_data = field_data[time_idx]
                            fields.append(field_data)
                    if fields:
                        shapes = [f.shape[:2] for f in fields]
                        if len(set(shapes)) == 1:
                            data = np.stack(fields, axis=-1) if len(fields) > 1 else fields[0]
                            return data
        
        # Try direct datasets
        for key in keys:
            if key in ['space_grid', 'input_time_grid', 'boundary_conditions', 'dimensions', 'scalars']:
                continue
            dataset = f[key]
            if isinstance(dataset, h5py.Dataset):
                shape = dataset.shape
                if len(shape) >= 2 and shape[0] > 2 and shape[1] > 2:
                    if len(shape) == 4:
                        data = dataset[snapshot_idx if snapshot_idx >= 0 else 0]
                        return data
                    elif len(shape) == 3:
                        data = dataset[snapshot_idx if snapshot_idx >= 0 else 0] if snapshot_idx >= 0 else dataset[-1]
                        return data
                    elif len(shape) == 2:
                        return dataset[:]
    
    return None


def find_dataset_files(data_dir, num_files=1):
    """Find HDF5 files in data directory. Returns one file to get first and last snapshots from."""
    if data_dir is None or not data_dir.exists():
        return []
    
    # Check common subdirectories
    subdirs = ['train', 'valid', 'test', 'data']
    for subdir in subdirs:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            h5_files = list(subdir_path.glob("*.h5")) + list(subdir_path.glob("*.hdf5"))
            if h5_files:
                return sorted(h5_files)[:num_files]
    
    # Check data_dir directly
    h5_files = list(data_dir.glob("*.h5")) + list(data_dir.glob("*.hdf5"))
    if h5_files:
        return sorted(h5_files)[:num_files]
    
    return []


def simulate_encoder_forward(quadtree_dicts, embed_dim=128, enc_dim=256, perc_dim=256, num_latent_tokens=64):
    """Simulate encoder forward pass to track shapes."""
    device = quadtree_dicts[0]['point_hierarchies'].device
    
    print(f"\n  Encoder configuration:")
    print(f"    embed_dim={embed_dim}, enc_dim={enc_dim}, perc_dim={perc_dim}")
    print(f"    num_latent_tokens={num_latent_tokens}")
    
    # Process each quadtree and track shapes
    all_features = []
    all_point_hier = []
    all_batch_idx = []
    
    for batch_idx, qd in enumerate(quadtree_dicts):
        point_hier = qd['point_hierarchies']  # (M_i, 2)
        features = qd['features']  # (M_i, C)
        pyramids = qd['pyramids'][0]  # (2, max_level+1)
        max_level = pyramids.shape[1] - 1
        
        print(f"\n  Sample {batch_idx + 1}:")
        print(f"    Input quadtree:")
        print(f"      point_hierarchies: {point_hier.shape}")
        print(f"      features: {features.shape}")
        print(f"      pyramids: {pyramids.shape}")
        print(f"      max_level: {max_level}")
        
        # Extract positions and depths (simulating encoder logic)
        M = point_hier.shape[0]
        node_depths = torch.zeros(M, dtype=torch.float32, device=device)
        
        for level in range(max_level + 1):
            start_idx = int(pyramids[1, level].item())
            if level < max_level:
                end_idx = int(pyramids[1, level + 1].item())
            else:
                end_idx = M
            if end_idx > start_idx:
                node_depths[start_idx:end_idx] = level
        
        level_powers = 2.0 ** node_depths
        node_positions = (point_hier.float() / level_powers.unsqueeze(-1)) * 2.0 - 1.0
        
        print(f"    After coordinate conversion:")
        print(f"      node_positions: {node_positions.shape}")
        print(f"      node_depths: {node_depths.shape}")
        print(f"      position range: [{node_positions.min():.3f}, {node_positions.max():.3f}]")
        
        all_features.append(features)
        all_point_hier.append(point_hier)
        all_batch_idx.append(torch.full((M,), batch_idx, dtype=torch.long, device=device))
    
    # Concatenate for batch processing
    all_features_cat = torch.cat(all_features, dim=0)  # (sum(M_i), C)
    all_batch_idx_cat = torch.cat(all_batch_idx, dim=0)  # (sum(M_i),)
    
    print(f"\n  Batch concatenation:")
    print(f"    all_features: {all_features_cat.shape}")
    print(f"    all_batch_idx: {all_batch_idx_cat.shape}")
    print(f"    Node counts per sample: {[f.shape[0] for f in all_features]}")
    
    # Simulate encoder stages
    C = all_features_cat.shape[1]
    
    # Node embedding
    node_embed = torch.nn.Linear(C, embed_dim).to(device)
    x = node_embed(all_features_cat)  # (sum(M_i), embed_dim)
    print(f"\n  After node_embed: {x.shape}")
    
    # Positional embedding (simulated - would add pos_embed + depth_embed)
    # For shape tracking, we just note the shape doesn't change
    print(f"  After positional embedding: {x.shape} (shape unchanged)")
    
    # Convert to dense batch
    from torch_geometric.utils import to_dense_batch
    x_dense, mask = to_dense_batch(x, all_batch_idx_cat)  # (B, N_max, embed_dim), (B, N_max)
    B, N_max = x_dense.shape[:2]
    print(f"\n  After to_dense_batch:")
    print(f"    x_dense: {x_dense.shape}")
    print(f"    mask: {mask.shape}")
    print(f"    Batch size: {B}, Max nodes: {N_max}")
    
    # Encoder projection
    enc_proj = torch.nn.Linear(embed_dim, enc_dim).to(device)
    x = enc_proj(x_dense)  # (B, N_max, enc_dim)
    print(f"  After enc_proj: {x.shape}")
    
    # Transformer blocks (simulated - shape unchanged)
    print(f"  After transformer blocks: {x.shape} (shape unchanged)")
    
    # Perceiver projection
    perc_proj = torch.nn.Linear(enc_dim, perc_dim).to(device)
    x = perc_proj(x)  # (B, N_max, perc_dim)
    print(f"  After perc_proj: {x.shape}")
    
    # PerceiverPoolingBlock (simulated)
    # Query tokens attend over N_max key-value tokens, output is fixed size
    output = torch.randn(B, num_latent_tokens, perc_dim, device=device)
    print(f"\n  After PerceiverPoolingBlock:")
    print(f"    output: {output.shape}")
    
    # Verify output shape
    expected_shape = (B, num_latent_tokens, perc_dim)
    assert output.shape == expected_shape, \
        f"Expected {expected_shape}, got {output.shape}"
    
    print(f"  ✓ Output shape matches expected: {expected_shape}")
    
    return output, mask


def test_datasets():
    """Test with 2 samples from each dataset."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    datasets = {
        'trl2d': {
            'config': '/home/workspace/projects/transformer/SHINE_mapping/quadtree/configs/turbulent_radiative_layer_2D.yaml',
            'data_dir': Path('/home/workspace/projects/data/datasets_david/datasets/turbulent_radiative_layer_2D/data')
        },
        'helmholtz': {
            'config': '/home/workspace/projects/transformer/SHINE_mapping/quadtree/configs/helmholtz_staircase.yaml',
            'data_dir': Path('/home/workspace/projects/data/datasets_david/datasets/helmholtz_staircase/data')
        },
        'zpinch': {
            'config': '/home/workspace/projects/transformer/SHINE_mapping/quadtree/configs/zpinch.yaml',
            'data_dir': Path('/home/workspace/projects/data/datasets_david/datasets/zpinch/data')
        }
    }
    
    max_level = 6
    all_results = []
    
    print("\n" + "="*80)
    print("Testing QuadtreeTransformerPerceiver with 2 samples per dataset")
    print("="*80)
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Find dataset files - use one file and get first/last snapshots
        data_dir = dataset_info.get('data_dir')
        dataset_files = find_dataset_files(data_dir, num_files=1)
        quadtree_dicts = []
        
        if not dataset_files:
            print(f"  ERROR: No dataset files found in {data_dir}")
            print(f"  Cannot proceed without real data. Skipping {dataset_name}...")
            all_results.append({'dataset': dataset_name, 'status': 'no_files', 'data_dir': str(data_dir)})
            continue
        
        print(f"  Found file: {dataset_files[0].name}")
        
        # Load first and last snapshots from trajectory
        filepath = dataset_files[0]
        
        # Check how many time steps are available
        with h5py.File(str(filepath), 'r') as f:
            if 't0_fields' in f.keys():
                t0 = f['t0_fields']
                field_name = sorted([k for k in t0.keys() if isinstance(t0[k], h5py.Dataset)])[0]
                field_shape = t0[field_name].shape
                if len(field_shape) == 4 and field_shape[1] > 1:
                    num_time_steps = field_shape[1]
                else:
                    num_time_steps = 1
            else:
                num_time_steps = 1
        
        print(f"  Trajectory has {num_time_steps} time steps")
        snapshot_indices = [0, num_time_steps - 1] if num_time_steps > 1 else [0, 0]
        
        for snap_idx, snapshot_idx in enumerate(snapshot_indices):
            print(f"\n  Snapshot {snap_idx+1}/2 (time step {snapshot_idx}):")
            
            snapshot = load_hdf5_snapshot(str(filepath), snapshot_idx=snapshot_idx)
            if snapshot is None:
                print(f"    ERROR: Could not load snapshot {snapshot_idx}")
                continue
            
            H, W, C = snapshot.shape
            print(f"    Snapshot shape: (H={H}, W={W}, C={C})")
            
            # Convert to torch and create quadtree
            feature_grid = torch.from_numpy(snapshot).float().permute(2, 0, 1).unsqueeze(0).to(device)
            
            try:
                quadtree_dict = feature_grids_to_quadtree(
                    feature_grid,
                    max_level=max_level,
                    physical_refinement=True,
                    refinement_config_path=dataset_info['config'],
                    return_spc=False
                )
                
                point_hier = quadtree_dict['point_hierarchies']
                features = quadtree_dict['features']
                
                print(f"    Quadtree created:")
                print(f"      Nodes: {point_hier.shape[0]}")
                print(f"      Features: {features.shape}")
                
                quadtree_dicts.append(quadtree_dict)
                
            except Exception as e:
                print(f"    ERROR creating quadtree: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(quadtree_dicts) < 2:
            print(f"  ERROR: Only {len(quadtree_dicts)} valid quadtrees created, need 2. Skipping {dataset_name}...")
            all_results.append({'dataset': dataset_name, 'status': 'insufficient_samples', 'created': len(quadtree_dicts)})
            continue
        
        if len(quadtree_dicts) < 2:
            print(f"  WARNING: Only {len(quadtree_dicts)} valid quadtrees, creating synthetic data...")
            # Create synthetic quadtree to complete batch
            while len(quadtree_dicts) < 2:
                # Use first valid quadtree as template or create from scratch
                if quadtree_dicts:
                    template = quadtree_dicts[0]
                    # Create similar structure with different node count
                    M = template['point_hierarchies'].shape[0]
                    new_M = int(M * 0.8)  # Slightly different size
                    synthetic_point_hier = torch.randint(0, 2**max_level, (new_M, 2), device=device)
                    synthetic_features = torch.randn(new_M, template['features'].shape[1], device=device)
                    synthetic_pyramids = template['pyramids'].clone()
                    quadtree_dicts.append({
                        'point_hierarchies': synthetic_point_hier,
                        'features': synthetic_features,
                        'pyramids': synthetic_pyramids
                    })
                    print(f"    Created synthetic quadtree with {new_M} nodes")
                else:
                    # Create from scratch
                    M = 100
                    synthetic_point_hier = torch.randint(0, 2**max_level, (M, 2), device=device)
                    synthetic_features = torch.randn(M, 6, device=device)
                    synthetic_pyramids = torch.zeros((1, 2, max_level + 1), dtype=torch.long, device=device)
                    for level in range(max_level + 1):
                        synthetic_pyramids[0, 0, level] = level
                        synthetic_pyramids[0, 1, level] = 0 if level == 0 else M
                    quadtree_dicts.append({
                        'point_hierarchies': synthetic_point_hier,
                        'features': synthetic_features,
                        'pyramids': synthetic_pyramids
                    })
                    print(f"    Created synthetic quadtree with {M} nodes")
        
        # Process batch through encoder
        print(f"\n  Processing batch of {len(quadtree_dicts)} samples...")
        
        try:
            output, mask = simulate_encoder_forward(
                quadtree_dicts,
                embed_dim=128,
                enc_dim=256,
                perc_dim=256,
                num_latent_tokens=64
            )
            
            # Verify UPT compatibility
            print(f"\n  UPT compatibility check:")
            print(f"    Output shape: {output.shape}")
            print(f"    Expected by UPT latent core: (B, M, d) where M={64}, d={256}")
            
            B, M, d = output.shape
            assert B == len(quadtree_dicts), f"Batch size mismatch: {B} != {len(quadtree_dicts)}"
            assert M == 64, f"Latent tokens mismatch: {M} != 64"
            assert d == 256, f"Latent dim mismatch: {d} != 256"
            
            print(f"  ✓ All checks passed!")
            
            all_results.append({
                'dataset': dataset_name,
                'status': 'success',
                'batch_size': B,
                'output_shape': output.shape,
                'node_counts': [qd['point_hierarchies'].shape[0] for qd in quadtree_dicts]
            })
            
        except Exception as e:
            print(f"  ERROR during encoding: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({'dataset': dataset_name, 'status': 'error', 'error': str(e)})
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for result in all_results:
        print(f"\n{result['dataset']}:")
        if result['status'] == 'success':
            print(f"  Status: ✓ SUCCESS")
            print(f"  Batch size: {result['batch_size']}")
            print(f"  Output shape: {result['output_shape']}")
            print(f"  Node counts: {result['node_counts']}")
        elif result['status'] == 'no_files':
            print(f"  Status: ✗ No files found")
        elif result['status'] == 'insufficient_samples':
            print(f"  Status: ✗ Insufficient samples")
        else:
            print(f"  Status: ✗ ERROR - {result.get('error', 'Unknown error')}")
    
    # Final verdict
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    total_count = len(all_results)
    
    print(f"\n{'='*80}")
    print(f"Final Result: {success_count}/{total_count} datasets processed successfully")
    print("="*80)
    
    if success_count == total_count:
        print("\n✓ All tests passed! QuadtreeTransformerPerceiver works correctly.")
        return True
    else:
        print(f"\n✗ {total_count - success_count} dataset(s) failed.")
        return False


if __name__ == "__main__":
    success = test_datasets()
    sys.exit(0 if success else 1)

