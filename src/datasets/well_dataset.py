import h5py
import torch
import numpy as np
from pathlib import Path
from functools import cached_property

from datasets.base.dataset_base import DatasetBase


class WellDataset(DatasetBase):
    """
    Dataset for The Well uniform grid simulations.
    
    The Well data structure (https://polymathic-ai.org/the_well/):
    - boundary_conditions/: Group with boundary condition info
    - dimensions/: Contains r_coords (X), z_coords (Y), time arrays
    - forcing_fields/: Contains forcing fields like current_drive
    - t0_fields/: Scalar fields (density, pressure) with shape (T, H, W)
    - t1_fields/: Vector fields (b_field, velocity) with shape (T, H, W, 2)
    
    Attributes:
        resolution: Tuple of (height, width) from directory name like "64x64"
        n_input_timesteps: Number of input timesteps
        n_pushforward_timesteps: Number of timesteps to predict (0 for single-step)
        split: train/valid/test split
    """
    
    def __init__(
        self,
        global_dataset_paths=None,
        n_input_timesteps=3,
        n_pushforward_timesteps=0,
        split="train",
        resolution=None,  # e.g., "64x64" or "all" for multi-resolution
        seed=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_input_timesteps = n_input_timesteps
        self.n_pushforward_timesteps = n_pushforward_timesteps
        self.split = split
        self.resolution = resolution
        
        # Get dataset root
        global_root, local_root = self._get_roots(global_dataset_paths, None, "well_dataset")
        self.root = Path(global_root)
        
        # Get all resolution folders or specific one
        if resolution is None or resolution == "all":
            # Load all resolutions
            resolution_dirs = sorted([d for d in self.root.iterdir() if d.is_dir() and 'x' in d.name])
            self.logger.info(f"Loading all resolutions: {[d.name for d in resolution_dirs]}")
        else:
            # Load specific resolution
            resolution_dirs = [self.root / resolution]
        
        # Collect all HDF5 files and track their timesteps (vary by resolution)
        self.file_list = []
        self.file_to_resolution = {}
        self.file_to_n_timesteps = {}  # Track timesteps per file
        self.file_to_n_windows = {}  # Track samples per file
        
        for res_dir in resolution_dirs:
            data_dir = res_dir / "data" / split
            if not data_dir.exists():
                self.logger.warning(f"Data directory not found: {data_dir}")
                continue
            
            files = sorted(list(data_dir.glob("*.hdf5")))
            resolution_name = res_dir.name
            
            for f in files:
                file_idx = len(self.file_list)
                self.file_list.append(f)
                self.file_to_resolution[file_idx] = resolution_name
                
                # Get timesteps for this file
                with h5py.File(f, 'r') as hf:
                    n_timesteps = hf['t0_fields']['density'].shape[0]
                    self.file_to_n_timesteps[file_idx] = n_timesteps
                    n_windows = n_timesteps - self.n_input_timesteps - self.n_pushforward_timesteps
                    self.file_to_n_windows[file_idx] = max(0, n_windows)
            
            self.logger.info(f"  {resolution_name}: {len(files)} files")
        
        if len(self.file_list) == 0:
            raise ValueError(f"No HDF5 files found in any resolution folder")
        
        self.logger.info(f"Total files across all resolutions: {len(self.file_list)}")
        
        # Build cumulative index for sample lookup
        self.file_cumulative_windows = [0]
        for file_idx in range(len(self.file_list)):
            self.file_cumulative_windows.append(
                self.file_cumulative_windows[-1] + self.file_to_n_windows[file_idx]
            )
        
        self.total_samples = self.file_cumulative_windows[-1]
        self.logger.info(f"Total samples: {self.total_samples}")
        
        # Get max timesteps for metadata
        self.n_timesteps_per_sim = max(self.file_to_n_timesteps.values())
        
        # Cache mesh positions per resolution (created on-demand in __getitem__)
        self._mesh_pos_cache = {}
        
        # Compute normalization statistics
        self._compute_normalization_stats()
        
        # Metadata for model compatibility
        self.metadata = {
            'dim': 2,
            'sequence_length_train': self.n_input_timesteps,
            'sequence_length_test': self.n_input_timesteps,
        }
        self.box = None  # No periodic boundary conditions like Lagrangian
    
    def _compute_normalization_stats(self):
        """Compute mean/std for normalization across the dataset (lazy loading)."""
        self.logger.info("Computing normalization statistics (sampling subset)...")
        
        # Fields with extreme ranges: always use log-normalization
        # Based on typical physics simulation ranges
        log_fields = {
            'density': True,  # Can span many orders of magnitude
            'pressure': True,  # Can span many orders of magnitude  
            'b_field': True,  # Magnetic fields can be extreme
            'velocity': True,  # Velocities can be extreme
            'current_drive': True,  # Forcing field can also have large values
        }
        
        # Accumulate statistics using Welford's online algorithm (numerically stable)
        stats = {}
        for field in log_fields.keys():
            stats[field] = {'count': 0, 'mean': 0.0, 'M2': 0.0, 'min': float('inf'), 'max': float('-inf')}
        
        # Sample small subset of files (lazy loading - don't load all files at once)
        n_files_to_sample = min(10, len(self.file_list))
        sampled_files = self.file_list[::max(1, len(self.file_list) // n_files_to_sample)]
        
        for filepath in sampled_files:
            # Lazy loading: only open one file at a time
            with h5py.File(filepath, 'r') as f:
                # Sample only first few timesteps to save memory
                n_timesteps_to_sample = min(10, f['t0_fields']['density'].shape[0])
                
                # Density and pressure
                for field in ['density', 'pressure']:
                    for t in range(0, n_timesteps_to_sample, 2):
                        data = f['t0_fields'][field][t]
                        if log_fields[field]:
                            data = np.log(np.clip(data, 1e-10, None))
                        stats[field]['count'] += data.size
                        delta = data - stats[field]['mean']
                        stats[field]['mean'] += delta.sum() / stats[field]['count']
                        stats[field]['M2'] += (delta * (data - stats[field]['mean'])).sum()
                        stats[field]['min'] = min(stats[field]['min'], data.min())
                        stats[field]['max'] = max(stats[field]['max'], data.max())
                
                # B-field and velocity
                for field in ['b_field', 'velocity']:
                    for t in range(0, n_timesteps_to_sample, 2):
                        data = f['t1_fields'][field][t]
                        if log_fields[field]:
                            data = np.sign(data) * np.log(np.clip(np.abs(data), 1e-10, None))
                        stats[field]['count'] += data.size
                        delta = data - stats[field]['mean']
                        stats[field]['mean'] += delta.sum() / stats[field]['count']
                        stats[field]['M2'] += (delta * (data - stats[field]['mean'])).sum()
                        stats[field]['min'] = min(stats[field]['min'], data.min())
                        stats[field]['max'] = max(stats[field]['max'], data.max())
                
                # Current drive
                for t in range(0, n_timesteps_to_sample, 2):
                    data = f['forcing_fields']['current_drive'][t]
                    if log_fields['current_drive']:
                        data = np.log(np.clip(data, 1e-10, None))
                    stats['current_drive']['count'] += data.size
                    delta = data - stats['current_drive']['mean']
                    stats['current_drive']['mean'] += delta.sum() / stats['current_drive']['count']
                    stats['current_drive']['M2'] += (delta * (data - stats['current_drive']['mean'])).sum()
                    stats['current_drive']['min'] = min(stats['current_drive']['min'], data.min())
                    stats['current_drive']['max'] = max(stats['current_drive']['max'], data.max())
        
        # Store normalization stats
        self.norm_stats = {}
        for field in stats:
            if stats[field]['count'] > 1:
                std = np.sqrt(stats[field]['M2'] / stats[field]['count'])
            else:
                std = 1.0
            
            self.norm_stats[field] = {
                'mean': stats[field]['mean'],
                'std': max(std, 1e-8),
                'min': stats[field]['min'],
                'max': stats[field]['max'],
                'use_log': log_fields[field],
            }
            log_str = " (log-norm)" if log_fields[field] else ""
            self.logger.info(f"  {field}: mean={stats[field]['mean']:.4e}, std={std:.4e}, range=[{stats[field]['min']:.4e}, {stats[field]['max']:.4e}]{log_str}")
    
    def __len__(self):
        return self.total_samples
    
    def _get_mesh_pos(self, filepath):
        """Get or create cached mesh positions for this file's resolution."""
        # Get grid coordinates from file
        with h5py.File(filepath, 'r') as f:
            r_coords = torch.from_numpy(f['dimensions']['r_coords'][...]).float()
            z_coords = torch.from_numpy(f['dimensions']['z_coords'][...]).float()
        
        # Create unique key based on grid shape
        grid_key = f"{len(r_coords)}x{len(z_coords)}"
        
        if grid_key not in self._mesh_pos_cache:
            # Create mesh grid
            z_grid, r_grid = torch.meshgrid(z_coords, r_coords, indexing='ij')
            mesh_pos = torch.stack([r_grid, z_grid], dim=-1)
            self._mesh_pos_cache[grid_key] = mesh_pos
        
        return self._mesh_pos_cache[grid_key]
    
    def __getitem__(self, idx):
        """
        Returns a sample containing input and target fields on a uniform grid.
        
        Returns:
            sample: dict with 'mesh_pos', 'x', 'target', 'timestep'
            ctx: dict with metadata like 'file_idx', 'window_start', etc.
        """
        # Find which file this sample belongs to (binary search in cumulative index)
        file_idx = 0
        for i in range(len(self.file_cumulative_windows) - 1):
            if idx >= self.file_cumulative_windows[i] and idx < self.file_cumulative_windows[i + 1]:
                file_idx = i
                break
        
        # Calculate window index within this file
        window_idx = idx - self.file_cumulative_windows[file_idx]
        window_start = window_idx
        window_end = window_start + self.n_input_timesteps
        target_idx = window_end + self.n_pushforward_timesteps
        
        filepath = self.file_list[file_idx]
        mesh_pos = self._get_mesh_pos(filepath)
        
        with h5py.File(filepath, 'r') as f:
            # Load input timesteps
            input_fields = []
            for t in range(window_start, window_end):
                # Scalar fields: (H, W)
                density = torch.from_numpy(f['t0_fields']['density'][t]).float()
                pressure = torch.from_numpy(f['t0_fields']['pressure'][t]).float()
                
                # Vector fields: (H, W, 2)
                b_field = torch.from_numpy(f['t1_fields']['b_field'][t]).float()
                velocity = torch.from_numpy(f['t1_fields']['velocity'][t]).float()
                
                # Forcing field: (H, W)
                current = torch.from_numpy(f['forcing_fields']['current_drive'][t]).float()
                
                # Normalize with log-transform for extreme ranges
                if self.norm_stats['density']['use_log']:
                    density = torch.log(density.clamp(min=1e-10))
                density = (density - self.norm_stats['density']['mean']) / self.norm_stats['density']['std']
                
                if self.norm_stats['pressure']['use_log']:
                    pressure = torch.log(pressure.clamp(min=1e-10))
                pressure = (pressure - self.norm_stats['pressure']['mean']) / self.norm_stats['pressure']['std']
                
                if self.norm_stats['b_field']['use_log']:
                    b_field = torch.sign(b_field) * torch.log(b_field.abs().clamp(min=1e-10))
                b_field = (b_field - self.norm_stats['b_field']['mean']) / self.norm_stats['b_field']['std']
                
                if self.norm_stats['velocity']['use_log']:
                    velocity = torch.sign(velocity) * torch.log(velocity.abs().clamp(min=1e-10))
                velocity = (velocity - self.norm_stats['velocity']['mean']) / self.norm_stats['velocity']['std']
                
                if self.norm_stats['current_drive']['use_log']:
                    current = torch.log(current.clamp(min=1e-10))
                current = (current - self.norm_stats['current_drive']['mean']) / self.norm_stats['current_drive']['std']
                
                # Stack all fields: (H, W, 7) = (H, W, [density, pressure, current, b_x, b_z, v_x, v_z])
                timestep_fields = torch.cat([
                    density.unsqueeze(-1),
                    pressure.unsqueeze(-1),
                    current.unsqueeze(-1),
                    b_field,
                    velocity,
                ], dim=-1)
                
                input_fields.append(timestep_fields)
            
            # Stack input timesteps: (n_input_timesteps, H, W, 7)
            x = torch.stack(input_fields, dim=0)
            
            # Load target
            target_density = torch.from_numpy(f['t0_fields']['density'][target_idx]).float()
            target_pressure = torch.from_numpy(f['t0_fields']['pressure'][target_idx]).float()
            target_b_field = torch.from_numpy(f['t1_fields']['b_field'][target_idx]).float()
            target_velocity = torch.from_numpy(f['t1_fields']['velocity'][target_idx]).float()
            target_current = torch.from_numpy(f['forcing_fields']['current_drive'][target_idx]).float()
            
            # Normalize target with log-transform for extreme ranges
            if self.norm_stats['density']['use_log']:
                target_density = torch.log(target_density.clamp(min=1e-10))
            target_density = (target_density - self.norm_stats['density']['mean']) / self.norm_stats['density']['std']
            
            if self.norm_stats['pressure']['use_log']:
                target_pressure = torch.log(target_pressure.clamp(min=1e-10))
            target_pressure = (target_pressure - self.norm_stats['pressure']['mean']) / self.norm_stats['pressure']['std']
            
            if self.norm_stats['b_field']['use_log']:
                target_b_field = torch.sign(target_b_field) * torch.log(target_b_field.abs().clamp(min=1e-10))
            target_b_field = (target_b_field - self.norm_stats['b_field']['mean']) / self.norm_stats['b_field']['std']
            
            if self.norm_stats['velocity']['use_log']:
                target_velocity = torch.sign(target_velocity) * torch.log(target_velocity.abs().clamp(min=1e-10))
            target_velocity = (target_velocity - self.norm_stats['velocity']['mean']) / self.norm_stats['velocity']['std']
            
            if self.norm_stats['current_drive']['use_log']:
                target_current = torch.log(target_current.clamp(min=1e-10))
            target_current = (target_current - self.norm_stats['current_drive']['mean']) / self.norm_stats['current_drive']['std']
            
            # Stack target: (H, W, 7)
            target = torch.cat([
                target_density.unsqueeze(-1),
                target_pressure.unsqueeze(-1),
                target_current.unsqueeze(-1),
                target_b_field,
                target_velocity,
            ], dim=-1)
        
        sample = {
            'mesh_pos': mesh_pos,  # (H, W, 2) - varies by resolution
            'x': x,  # (n_input_timesteps, H, W, 7)
            'target': target,  # (H, W, 7)
            'timestep': torch.tensor(window_start, dtype=torch.long),
        }
        
        ctx = {
            'file_idx': file_idx,
            'window_start': window_start,
            'target_idx': target_idx,
        }
        
        return sample, ctx
    
    def getshape_x(self):
        """Return shape of input features for model."""
        n_fields = 7  # density, pressure, current, b_x, b_z, v_x, v_z
        # Return (None, n_timesteps * n_fields) as CFD encoder expects flattened temporal features
        return (None, self.n_input_timesteps * n_fields)
    
    def getshape_target(self):
        """Return shape of target for model."""
        n_fields = 7
        # Return (None, n_fields) as CFD decoder expects per-point features
        return (None, n_fields)
    
    def getshape_timestep(self):
        """Return shape of timestep."""
        return (self.n_timesteps_per_sim,)
    
    def getitem_x(self, idx, ctx=None):
        if ctx is None or 'x' not in ctx:
            sample, _ = self[idx]
            return sample['x']
        return ctx['x']
    
    def getitem_target(self, idx, ctx=None):
        if ctx is None or 'target' not in ctx:
            sample, _ = self[idx]
            return sample['target']
        return ctx['target']
    
    def getitem_mesh_pos(self, idx, ctx=None):
        if ctx is None or 'mesh_pos' not in ctx:
            sample, _ = self[idx]
            return sample['mesh_pos']
        return ctx['mesh_pos']
    
    def getitem_timestep(self, idx, ctx=None):
        if ctx is None or 'timestep' not in ctx:
            sample, _ = self[idx]
            return sample['timestep']
        return ctx['timestep']
    
    def getitem_query_pos(self, idx, ctx=None):
        """Return query positions (same as mesh_pos for uniform grids)."""
        if ctx is None or 'mesh_pos' not in ctx:
            sample, _ = self[idx]
            return sample['mesh_pos']
        return ctx['mesh_pos']
    
    def getitem_mesh_edges(self, idx, ctx=None):
        """Mesh edges are not precomputed for uniform grids - handled by collator."""
        return None
    
    def getitem_geometry2d(self, idx, ctx=None):
        """No geometry information for uniform grids."""
        return None
    
    def getitem_velocity(self, idx, ctx=None):
        """Velocity field - return dummy scalar for batch size inference in conditioner."""
        return torch.tensor(0.0)

