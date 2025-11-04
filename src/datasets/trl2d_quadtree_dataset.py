"""
Dataset for Turbulent Radiative Layer 2D (TRL2D) data for quadtree-UPT.

Uses The Well dataset infrastructure but outputs grid format for quadtree partitioning.
"""

from pathlib import Path
import torch
import einops
from torch.utils.data import Dataset

from the_well.data import WellDataset


class TRL2DQuadtreeDataset(Dataset):
    """
    TRL2D dataset from The Well for quadtree-based UPT training.
    
    Wraps WellDataset and converts to grid format for quadtree partitioning.
    Based on TRL2D specs from https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_2D/
    - Spatial resolution: 128 x 384
    - Fields: density, pressure, velocity (vx, vy)
    - Timesteps: 101 per trajectory
    """
    
    def __init__(
        self,
        data_dir="/home/workspace/projects/data/datasets_david/datasets/",
        split="train",
        n_input_timesteps=1,
        max_num_sequences=None,
    ):
        super().__init__()
        
        self.split = split
        self.n_input_timesteps = n_input_timesteps
        
        # Use The Well dataset loader
        from the_well.data.normalization import ZScoreNormalization
        
        self.well = WellDataset(
            well_base_path=Path(data_dir),
            well_dataset_name="turbulent_radiative_layer_2D",
            well_split_name=split,
            n_steps_input=n_input_timesteps,
            n_steps_output=1,  # Next timestep only
            use_normalization=True,
            normalization_type=ZScoreNormalization,
        )
        
        # Limit sequences if requested
        self._indices = list(range(len(self.well)))
        if max_num_sequences is not None:
            self._indices = self._indices[:max_num_sequences]
        
        # Get metadata from first sample
        sample0 = self.well[0]
        T_in, H, W, F = sample0["input_fields"].shape
        self.H, self.W, self.F = int(H), int(W), int(F)
        
        print(f"TRL2D Quadtree Dataset ({split}):")
        print(f"  Samples: {len(self._indices)}")
        print(f"  Spatial size: {self.H} x {self.W}")
        print(f"  Channels: {self.F}")
        print(f"  Input timesteps: {n_input_timesteps}")
    
    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, idx):
        """
        Get a training sample in grid format for quadtree partitioning.
        
        Returns:
            dict with:
                - input_grid: (C, H, W) normalized input state
                - target_grid: (C, H, W) normalized target state  
                - timestep: scalar timestep index
                - index: sample global index
        """
        actual_idx = self._indices[idx]
        sample = self.well[actual_idx]
        
        # sample["input_fields"]: (T_in, H, W, F)
        # sample["output_fields"]: (T_out, H, W, F)
        
        input_fields = sample["input_fields"]  # (T_in, H, W, F)
        output_fields = sample["output_fields"]  # (1, H, W, F)
        
        # Average input timesteps
        input_state = input_fields.mean(dim=0)  # (H, W, F)
        
        # Get target (first output timestep)
        target_state = output_fields[0]  # (H, W, F)
        
        # Convert to (C, H, W) format
        input_grid = input_state.permute(2, 0, 1)  # (F, H, W)
        target_grid = target_state.permute(2, 0, 1)  # (F, H, W)
        
        return {
            "input_grid": input_grid,
            "target_grid": target_grid,
            "timestep": 0,  # Will be updated during training
            "index": actual_idx,
        }

