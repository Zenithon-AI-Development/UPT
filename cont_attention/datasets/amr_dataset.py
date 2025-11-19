import os
import json
import h5py
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph, radius
from .base.dataset_base import DatasetBase

from distributed.config import barrier, is_data_rank0
import einops


class AMRDataset(DatasetBase):
    """
    Dataset for Adaptive Mesh Refinement (AMR) simulations.
    
    Each simulation is stored in an HDF5 file with structure:
        - point_clouds/timestep_XXXX: (N_cells, 8) arrays
          Fields: [r, z, dens, pres, velx, vely, magp, magz]
        - time_grid: (n_timesteps,) array of timestamps
        - forcing_fields/: current drive data
        - boundary_conditions/: boundary condition types
    
    Similar to LagrangianDataset, but for Eulerian AMR grids.
    """
    
    def __init__(
            self,
            name,
            n_input_timesteps=3,
            n_pushforward_timesteps=0,
            graph_mode='radius_graph',
            knn_graph_k=1,
            radius_graph_r=0.05,
            radius_graph_max_num_neighbors=int(1e10),
            split="train",
            test_mode='parts_traj',
            n_supernodes=None,
            num_points_range=None,
            global_root=None,
            local_root=None,
            seed=None,
            pos_scale=1.0,
            predict_fields=True,  # True: predict fields, False: predict derivatives
            **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.name = name
        self.n_input_timesteps = n_input_timesteps
        self.n_pushforward_timesteps = n_pushforward_timesteps
        self.graph_mode = graph_mode
        self.knn_graph_k = knn_graph_k
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors
        self.split = split
        self.test_mode = test_mode
        self.n_supernodes = n_supernodes
        self.num_points_range = num_points_range
        self.seed = seed
        self.pos_scale = pos_scale
        self.predict_fields = predict_fields
        
        # Create metadata dict for model compatibility
        self.metadata = {
            'dim': 2,  # Spatial dimension
            'sequence_length_train': 701,
            'sequence_length_test': 701,
        }
        
        # No periodic boundary conditions for AMR
        self.box = None
        
        # Get roots using dataset_config_provider
        global_root, local_root = self._get_roots(global_root, local_root, "amr_dataset")
        
        # Use global_root for now (TODO: add local copying like CFD dataset)
        data_source = os.path.join(global_root, split)
        
        self.data_source = data_source
        self.logger.info(f"data_source (global): '{data_source}'")
        
        # Load simulation files
        self.sim_files = sorted([
            f for f in os.listdir(data_source) 
            if f.endswith('.hdf5')
        ])
        
        if len(self.sim_files) == 0:
            raise ValueError(f"No HDF5 files found in {data_source}")
        
        self.logger.info(f"Found {len(self.sim_files)} simulation files")
        
        # Load metadata from first file
        with h5py.File(os.path.join(data_source, self.sim_files[0]), 'r') as f:
            # Get number of timesteps
            timesteps = sorted(list(f['point_clouds'].keys()))
            self.n_timesteps_per_sim = len(timesteps)
            
            # Get spatial dimension
            sample_data = f['point_clouds'][timesteps[0]][:]
            self.dim = 2  # r, z coordinates
            self.n_fields = 6  # dens, pres, velx, vely, magp, magz
            
            self.logger.info(f"Timesteps per simulation: {self.n_timesteps_per_sim}")
            self.logger.info(f"Spatial dimension: {self.dim}")
            self.logger.info(f"Number of fields: {self.n_fields}")
        
        # Compute number of valid samples per trajectory
        # Need n_input_timesteps + 1 (for target)
        self.n_per_traj = self.n_timesteps_per_sim - self.n_input_timesteps
        self.n_traj = len(self.sim_files)
        
        # Compute normalization statistics
        self._compute_normalization_stats()
        
        # Handle supernodes
        if n_supernodes is not None:
            self.n_supernodes = n_supernodes
        else:
            # Use average number of cells as default
            self.n_supernodes = 4096  # Typical for this dataset
        
        self.logger.info(f"Total trajectories: {self.n_traj}")
        self.logger.info(f"Samples per trajectory: {self.n_per_traj}")
        self.logger.info(f"Total samples: {len(self)}")
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics across all simulations."""
        self.logger.info("Computing normalization statistics...")
        
        # Sample subset of data for statistics
        max_samples = 10000
        samples_per_file = max(1, max_samples // len(self.sim_files))
        
        all_positions = []
        all_fields = []
        
        for sim_file in self.sim_files[:min(10, len(self.sim_files))]:  # Use first 10 files
            filepath = os.path.join(self.data_source, sim_file)
            with h5py.File(filepath, 'r') as f:
                timesteps = sorted(list(f['point_clouds'].keys()))
                
                # Sample timesteps
                step = max(1, len(timesteps) // samples_per_file)
                for ts in timesteps[::step]:
                    data = f['point_clouds'][ts][:]
                    positions = data[:, :2]  # r, z
                    fields = data[:, 2:]  # dens, pres, velx, vely, magp, magz
                    
                    all_positions.append(positions)
                    all_fields.append(fields)
        
        # Concatenate
        all_positions = np.concatenate(all_positions, axis=0)
        all_fields = np.concatenate(all_fields, axis=0)
        
        # Position normalization (scale and offset)
        self.pos_offset = torch.tensor(all_positions.min(axis=0), dtype=torch.float32)
        pos_range = all_positions.max(axis=0) - all_positions.min(axis=0)
        self.pos_scale = torch.tensor(1.0 / (pos_range + 1e-8), dtype=torch.float32) * self.pos_scale
        
        # Field normalization
        # Use log-transform for density, pressure, magnetic pressure (huge dynamic range)
        # Use z-normalization for velocities and magnetic field
        
        dens = all_fields[:, 0]
        pres = all_fields[:, 1]
        velx = all_fields[:, 2]
        vely = all_fields[:, 3]
        magp = all_fields[:, 4]
        magz = all_fields[:, 5]
        
        # Log-scale fields (density, pressure, magnetic pressure)
        self.dens_log_mean = torch.tensor(np.log10(dens + 1e-10).mean(), dtype=torch.float32)
        self.dens_log_std = torch.tensor(np.log10(dens + 1e-10).std() + 1e-8, dtype=torch.float32)
        
        self.pres_log_mean = torch.tensor(np.log10(pres + 1e-10).mean(), dtype=torch.float32)
        self.pres_log_std = torch.tensor(np.log10(pres + 1e-10).std() + 1e-8, dtype=torch.float32)
        
        self.magp_log_mean = torch.tensor(np.log10(magp + 1e-10).mean(), dtype=torch.float32)
        self.magp_log_std = torch.tensor(np.log10(magp + 1e-10).std() + 1e-8, dtype=torch.float32)
        
        # Z-normalize fields (velocities, magnetic field)
        self.velx_mean = torch.tensor(velx.mean(), dtype=torch.float32)
        self.velx_std = torch.tensor(velx.std() + 1e-8, dtype=torch.float32)
        
        self.vely_mean = torch.tensor(vely.mean(), dtype=torch.float32)
        self.vely_std = torch.tensor(vely.std() + 1e-8, dtype=torch.float32)
        
        self.magz_mean = torch.tensor(magz.mean(), dtype=torch.float32)
        self.magz_std = torch.tensor(magz.std() + 1e-8, dtype=torch.float32)
        
        self.logger.info("Normalization statistics computed")
        self.logger.info(f"  Position range: r=[{all_positions[:, 0].min():.4f}, {all_positions[:, 0].max():.4f}], "
                        f"z=[{all_positions[:, 1].min():.4f}, {all_positions[:, 1].max():.4f}]")
        self.logger.info(f"  Density log: mean={self.dens_log_mean:.4f}, std={self.dens_log_std:.4f}")
        self.logger.info(f"  Pressure log: mean={self.pres_log_mean:.4f}, std={self.pres_log_std:.4f}")
        self.logger.info(f"  Velocity X: mean={self.velx_mean:.4f}, std={self.velx_std:.4f}")
    
    def normalize_fields(self, fields):
        """
        Normalize fields: [dens, pres, velx, vely, magp, magz]
        
        Args:
            fields: (..., 6) tensor
        Returns:
            normalized_fields: (..., 6) tensor
        """
        dens, pres, velx, vely, magp, magz = torch.split(fields, 1, dim=-1)
        
        # Log-normalize
        dens_norm = (torch.log10(dens + 1e-10) - self.dens_log_mean.to(dens.device)) / self.dens_log_std.to(dens.device)
        pres_norm = (torch.log10(pres + 1e-10) - self.pres_log_mean.to(pres.device)) / self.pres_log_std.to(pres.device)
        magp_norm = (torch.log10(magp + 1e-10) - self.magp_log_mean.to(magp.device)) / self.magp_log_std.to(magp.device)
        
        # Z-normalize
        velx_norm = (velx - self.velx_mean.to(velx.device)) / self.velx_std.to(velx.device)
        vely_norm = (vely - self.vely_mean.to(vely.device)) / self.vely_std.to(vely.device)
        magz_norm = (magz - self.magz_mean.to(magz.device)) / self.magz_std.to(magz.device)
        
        return torch.cat([dens_norm, pres_norm, velx_norm, vely_norm, magp_norm, magz_norm], dim=-1)
    
    def denormalize_fields(self, fields_norm):
        """
        Denormalize fields back to physical units.
        
        Args:
            fields_norm: (..., 6) tensor
        Returns:
            fields: (..., 6) tensor
        """
        dens_norm, pres_norm, velx_norm, vely_norm, magp_norm, magz_norm = torch.split(fields_norm, 1, dim=-1)
        
        # Denormalize log-scale (epsilon was only for numerical stability in log, don't subtract after exp)
        dens = 10 ** (dens_norm * self.dens_log_std.to(dens_norm.device) + self.dens_log_mean.to(dens_norm.device))
        pres = 10 ** (pres_norm * self.pres_log_std.to(pres_norm.device) + self.pres_log_mean.to(pres_norm.device))
        magp = 10 ** (magp_norm * self.magp_log_std.to(magp_norm.device) + self.magp_log_mean.to(magp_norm.device))
        
        # Denormalize z-scale
        velx = velx_norm * self.velx_std.to(velx_norm.device) + self.velx_mean.to(velx_norm.device)
        vely = vely_norm * self.vely_std.to(vely_norm.device) + self.vely_mean.to(vely_norm.device)
        magz = magz_norm * self.magz_std.to(magz_norm.device) + self.magz_mean.to(magz_norm.device)
        
        return torch.cat([dens, pres, velx, vely, magp, magz], dim=-1)
    
    def scale_pos(self, pos):
        """Scale position coordinates."""
        pos = pos - self.pos_offset.to(pos.device)
        pos = pos * self.pos_scale.to(pos.device)
        return pos
    
    def unscale_pos(self, pos):
        """Unscale position coordinates."""
        pos = pos / self.pos_scale.to(pos.device)
        pos = pos + self.pos_offset.to(pos.device)
        return pos
    
    def get_window(self, idx):
        """
        Load a window of timesteps for training.
        
        For AMR data, grid size can change between timesteps.
        We use the LAST timestep as the reference grid.
        
        Args:
            idx: global sample index
        Returns:
            positions: (N, 2) tensor - positions at last timestep
            input_fields_list: list of (N_t, 6) tensors - fields at each input timestep
            target_fields: (N, 6) tensor - fields at target timestep
        """
        # Compute trajectory and local index
        traj_idx = idx // self.n_per_traj
        local_idx = idx % self.n_per_traj
        
        # Load simulation file
        sim_file = self.sim_files[traj_idx]
        filepath = os.path.join(self.data_source, sim_file)
        
        with h5py.File(filepath, 'r') as f:
            timesteps = sorted(list(f['point_clouds'].keys()))
            
            # Load window of timesteps
            window_timesteps = timesteps[local_idx:local_idx + self.n_input_timesteps + 1]
            
            data_list = []
            for ts in window_timesteps:
                data = f['point_clouds'][ts][:]
                data_list.append(torch.from_numpy(data).float())
            
            # Use last timestep as reference
            target_data = data_list[-1]
            target_positions = target_data[:, :2]  # (N, 2)
            target_fields = target_data[:, 2:]  # (N, 6)
            
            # Input fields from previous timesteps
            input_fields_list = [d[:, 2:] for d in data_list[:-1]]
            
        return target_positions, input_fields_list, target_fields
    
    def __len__(self):
        return self.n_traj * self.n_per_traj
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Since AMR grids can have different sizes at each timestep,
        we use the target timestep grid as reference and compute
        aggregated features from previous timesteps.
        """
        target_positions, input_fields_list, target_fields = self.get_window(idx)
        
        # target_positions: (N, 2) - grid at target timestep
        # input_fields_list: list of (N_t, 6) tensors - fields at input timesteps (N_t can vary)
        # target_fields: (N, 6) - fields at target timestep
        
        curr_pos = target_positions  # Use target grid as current position
        
        # For input features, we need to aggregate fields from previous timesteps
        # Since grid topology changes, we use a simple approach:
        # Compute mean and std of each field across all input cells
        
        # Concatenate all input fields
        all_input_fields = torch.cat(input_fields_list, dim=0)  # (sum(N_t), 6)
        
        # Compute statistics per field
        input_mean = all_input_fields.mean(dim=0, keepdim=True)  # (1, 6)
        input_std = all_input_fields.std(dim=0, keepdim=True) + 1e-8  # (1, 6)
        input_min = all_input_fields.min(dim=0, keepdim=True)[0]  # (1, 6)
        input_max = all_input_fields.max(dim=0, keepdim=True)[0]  # (1, 6)
        
        # Broadcast to target grid
        n_points = curr_pos.shape[0]
        input_features = torch.cat([
            input_mean.expand(n_points, -1),
            input_std.expand(n_points, -1),
            input_min.expand(n_points, -1),
            input_max.expand(n_points, -1),
        ], dim=-1)  # (N, 24) = (N, 4*6)
        
        # Normalize
        curr_pos_scaled = self.scale_pos(curr_pos)
        
        # Normalize input statistics (already aggregated, so just normalize once)
        input_mean_norm = self.normalize_fields(input_mean).expand(n_points, -1)
        input_std_norm = input_std.expand(n_points, -1)  # Don't normalize std
        input_min_norm = self.normalize_fields(input_min).expand(n_points, -1)
        input_max_norm = self.normalize_fields(input_max).expand(n_points, -1)
        
        input_features_norm = torch.cat([
            input_mean_norm,
            input_std_norm,
            input_min_norm,
            input_max_norm,
        ], dim=-1)  # (N, 24)
        
        target_fields_norm = self.normalize_fields(target_fields)  # (N, 6)
        
        # Subsample points if num_points_range is specified
        if self.num_points_range is not None:
            n_points = curr_pos.shape[0]
            if n_points > self.num_points_range[1]:
                # Use deterministic subsampling for overfitting (seed-based)
                generator = self._get_generator(idx)
                if generator is not None:
                    indices = torch.randperm(n_points, generator=generator)[:self.num_points_range[1]]
                else:
                    indices = torch.arange(self.num_points_range[1])
                curr_pos_scaled = curr_pos_scaled[indices]
                input_features_norm = input_features_norm[indices]
                target_fields_norm = target_fields_norm[indices]
        
        return curr_pos_scaled, input_features_norm, target_fields_norm
    
    def getshape_x(self):
        """Return shape of input features."""
        # Returns (mean, std, min, max) for each field
        return None, 4 * self.n_fields
    
    def getshape_target(self):
        """Return shape of target."""
        return None, self.n_fields
    
    def getter(self, idx, ctx):
        """Get or cache data tuple."""
        if ctx is None or len(ctx) == 0:
            return self[idx]
        return ctx
    
    def getitem_x(self, idx, ctx=None):
        """Return input features."""
        curr_pos, x, target = self.getter(idx, ctx)
        # Reshape to (num_input_timesteps, num_channels, num_points) format expected by collator
        # Treat as single timestep with 24 channels
        x = x.T.unsqueeze(0)  # (1, 24, N)
        return x
    
    def getitem_curr_pos(self, idx, ctx=None):
        """Return current positions."""
        curr_pos, x, target = self.getter(idx, ctx)
        return curr_pos
    
    def getitem_curr_pos_full(self, idx, ctx=None):
        """Return full positions (including supernodes if used)."""
        return self.getitem_curr_pos(idx, ctx)
    
    def getitem_target_acc(self, idx, ctx=None):
        """Return target fields (named acc for compatibility with trainer)."""
        curr_pos, x, target = self.getter(idx, ctx)
        return target
    
    def getitem_timestep(self, idx, ctx=None):
        """Return timestep index."""
        if self.split == 'test' or self.split == 'valid':
            return idx % self.n_per_traj
        return idx % self.n_per_traj
    
    def _get_generator(self, idx):
        """Get random generator for consistent sampling."""
        if self.seed is None:
            return None
        return torch.Generator().manual_seed(self.seed + int(idx))
    
    def getitem_edge_index(self, idx, ctx=None):
        """Build edge index for radius graph with supernodes."""
        curr_pos, x, target = self.getter(idx, ctx)
        
        if self.graph_mode == 'radius_graph':
            edge_index = radius_graph(
                x=curr_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
                loop=True,
            )
            edge_index = edge_index.T
        elif self.graph_mode == 'radius_graph_with_supernodes':
            # Select supernodes
            generator = self._get_generator(idx)
            perm_supernodes = torch.randperm(curr_pos.shape[0], generator=generator)[:self.n_supernodes]
            supernodes_pos = curr_pos[perm_supernodes]
            
            # Create edges from cells to supernodes
            edge_index = radius(
                x=curr_pos,
                y=supernodes_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
            )
            # Correct supernode index
            edge_index[0] = perm_supernodes[edge_index[0]]
            edge_index = edge_index.T
        else:
            raise NotImplementedError(f"Graph mode {self.graph_mode} not implemented")
        
        return edge_index
    
    def getitem_edge_index_target(self, idx, ctx=None):
        """Build edge index for target (decoder)."""
        curr_pos, x, target = self.getter(idx, ctx)
        
        if self.graph_mode == 'radius_graph_with_supernodes':
            # Build radius graph on full grid for decoder
            edge_index = radius_graph(
                x=curr_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
                loop=True,
            )
            edge_index = edge_index.T
        else:
            edge_index = self.getitem_edge_index(idx, ctx)
        
        return edge_index
    
    def getitem_target_pos(self, idx, ctx=None):
        """Return target positions (same as current for AMR)."""
        return self.getitem_curr_pos(idx, ctx)
    
    def getitem_prev_pos(self, idx, ctx=None):
        """Return previous positions (same as current for AMR)."""
        return self.getitem_curr_pos(idx, ctx)
    
    def getitem_prev_acc(self, idx, ctx=None):
        """Return previous acceleration (not used for AMR)."""
        return None
    
    def getitem_target_pos_encode(self, idx, ctx=None):
        """Return target positions for encoding."""
        return self.getitem_curr_pos(idx, ctx)
    
    def getitem_perm(self, idx, ctx=None):
        """Return permutation for supernodes."""
        curr_pos, x, target = self.getter(idx, ctx)
        
        if self.graph_mode == 'radius_graph_with_supernodes':
            generator = self._get_generator(idx)
            perm = torch.randperm(curr_pos.shape[0], generator=generator)[:self.n_supernodes]
            n_particles = curr_pos.shape[0]
            return perm, n_particles
        return None
    
    def getitem_target_vel(self, idx, ctx=None):
        """Return target velocity (not used for AMR, returns target fields)."""
        return self.getitem_target_acc(idx, ctx)
    
    def getshape_timestep(self):
        """Return max number of timesteps."""
        return self.n_timesteps_per_sim,

