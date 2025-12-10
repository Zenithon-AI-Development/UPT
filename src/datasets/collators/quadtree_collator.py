"""
Collator for quadtree-based training with lazy data loading and tree construction.

Handles:
- Lazy loading of snapshots (only when needed)
- Quadtree construction from grid data
- Tree reuse when same snapshot appears in multiple samples
- Normalization per channel (0 mean, 1 std) - uses dataset's normalization
- Batching of variable-length quadtree nodes
- 4 input timesteps + 1 output timestep prediction
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper
from torch_geometric.utils import to_dense_batch

# Add SHINE_mapping quadtree to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "SHINE_mapping" / "quadtree"))
try:
    from kaolin_quadtree_2D import feature_grids_to_quadtree
except ImportError:
    raise ImportError(
        "Could not import feature_grids_to_quadtree from SHINE_mapping/quadtree. "
        "Ensure the SHINE_mapping directory is accessible."
    )


class QuadtreeCollator(KDSingleCollator):
    """
    Collator that constructs quadtrees from grid data lazily.
    
    For each sample in batch:
    1. Extracts grid data (4 input timesteps + 1 output timestep)
    2. Constructs quadtree for each timestep (reusing when possible)
    3. Collates quadtree dicts with proper batch_idx for variable-length nodes
    
    Tree reuse: If the same (file_idx, timestep) appears in multiple samples,
    the quadtree is constructed once and reused.
    """
    
    def __init__(
        self,
        max_level: int = 6,
        physical_refinement: bool = True,
        refinement_config_path: Optional[str] = None,
        cache_quadtrees: bool = True,
        dataset: Any = None,
        **kwargs
    ):
        """
        Args:
            max_level: Maximum quadtree level
            physical_refinement: Whether to use physics-based refinement
            refinement_config_path: Path to YAML config for physical refinement
            cache_quadtrees: Whether to cache constructed quadtrees for reuse
            dataset: Optional dataset reference for accessing raw data via _item method
        """
        super().__init__(**kwargs)
        self.max_level = max_level
        self.physical_refinement = physical_refinement
        self.refinement_config_path = refinement_config_path
        self.cache_quadtrees = cache_quadtrees
        self.dataset = dataset
        
        # Cache for quadtree reuse: (file_idx, timestep) -> quadtree_dict
        self._quadtree_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        
    def _clear_cache(self):
        """Clear quadtree cache (call between epochs if needed)."""
        self._quadtree_cache.clear()
    
    def _get_grid_shape_from_sample(self, sample: Tuple, dataset_mode: str) -> Tuple[int, int, int]:
        """
        Infer grid shape (H, W, C) from sample.
        
        Dataset provides flattened data, so we need to reconstruct original shape.
        For WellTrl2dDataset: input_fields is (T_in, H, W, C), output_fields is (T_out, H, W, C)
        """
        # Try to get input_fields from sample
        try:
            input_fields = ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="input_fields")
            if input_fields is not None and hasattr(input_fields, 'shape'):
                # If we have access to raw input_fields, use its shape
                if input_fields.ndim == 4:  # (T, H, W, C)
                    _, H, W, C = input_fields.shape
                    return int(H), int(W), int(C)
        except (KeyError, AttributeError):
            pass
        
        # Fallback: try to infer from dataset metadata
        # This is a heuristic - in practice, dataset should provide this
        # For now, return None and handle in collate
        return None, None, None
    
    def _extract_grid_from_sample(
        self,
        sample: Tuple,
        dataset_mode: str,
        timestep_idx: int,
        is_input: bool = True,
        dataset: Any = None,
        sample_idx: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Extract grid data for a specific timestep from sample.
        
        Args:
            sample: Sample tuple from dataset
            dataset_mode: Dataset mode string
            timestep_idx: Index of timestep (0-3 for input, 0 for output)
            is_input: Whether this is an input timestep (True) or output (False)
            dataset: Optional dataset object to access raw data
        
        Returns:
            Grid tensor (H, W, C) or None if not available
        """
        # Try to get raw sample data from dataset if available
        if dataset is not None and sample_idx is not None:
            try:
                # Access raw sample via dataset's _item method
                if hasattr(dataset, '_item'):
                    raw_sample = dataset._item(sample_idx)
                    if isinstance(raw_sample, dict):
                        if is_input:
                            input_fields = raw_sample.get('input_fields')
                            if input_fields is not None:
                                input_fields = torch.as_tensor(input_fields, dtype=torch.float32)
                                # Apply normalization (dataset does this, but we need it here too)
                                if hasattr(dataset, '_apply_norm'):
                                    input_fields = dataset._apply_norm(input_fields)
                                if input_fields.ndim == 4:  # (T, H, W, C)
                                    return input_fields[timestep_idx]  # (H, W, C)
                                elif input_fields.ndim == 3:  # (H, W, C)
                                    return input_fields if timestep_idx == 0 else None
                        else:
                            output_fields = raw_sample.get('output_fields')
                            if output_fields is not None:
                                output_fields = torch.as_tensor(output_fields, dtype=torch.float32)
                                # Apply normalization
                                if hasattr(dataset, '_apply_norm'):
                                    output_fields = dataset._apply_norm(output_fields)
                                if output_fields.ndim == 4:  # (T, H, W, C)
                                    return output_fields[timestep_idx]  # (H, W, C)
                                elif output_fields.ndim == 3:  # (H, W, C)
                                    return output_fields if timestep_idx == 0 else None
            except (AttributeError, KeyError, IndexError, TypeError):
                pass
        
        # Fallback: try ModeWrapper (may not work for raw grid data)
        try:
            if is_input:
                input_fields = ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="input_fields")
                if input_fields is not None:
                    if input_fields.ndim == 4:  # (T, H, W, C)
                        return input_fields[timestep_idx]  # (H, W, C)
                    elif input_fields.ndim == 3:  # (H, W, C) - single timestep
                        return input_fields if timestep_idx == 0 else None
            else:
                output_fields = ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="output_fields")
                if output_fields is not None:
                    if output_fields.ndim == 4:  # (T, H, W, C)
                        return output_fields[timestep_idx]  # (H, W, C)
                    elif output_fields.ndim == 3:  # (H, W, C) - single timestep
                        return output_fields if timestep_idx == 0 else None
        except (KeyError, AttributeError, IndexError):
            pass
        
        return None
    
    def _construct_quadtree_from_grid(
        self,
        grid: torch.Tensor,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Construct quadtree from a single grid snapshot.
        
        Args:
            grid: (H, W, C) tensor of normalized grid data
            device: Device to place quadtree on
        
        Returns:
            Quadtree dict with 'point_hierarchies', 'pyramids', 'features'
        """
        # Convert (H, W, C) to (1, C, H, W) for feature_grids_to_quadtree
        if grid.ndim == 3:
            grid_batch = grid.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        else:
            grid_batch = grid.unsqueeze(0)  # Assume (C, H, W) -> (1, C, H, W)
        
        grid_batch = grid_batch.to(device)
        
        # Construct quadtree
        quadtree_dict = feature_grids_to_quadtree(
            feature_grids=grid_batch,
            max_level=self.max_level,
            physical_refinement=self.physical_refinement,
            refinement_config_path=self.refinement_config_path,
            return_spc=False
        )
        
        return quadtree_dict
    
    def _get_sample_identifier(self, sample: Tuple, dataset_mode: str) -> Optional[Tuple[int, int]]:
        """
        Get identifier for sample to enable quadtree reuse.
        
        Returns:
            (file_idx, timestep) tuple or None if not available
        """
        # Try to extract file/trajectory identifier and timestep from sample
        # This depends on dataset implementation
        try:
            # WellTrl2dDataset stores this in metadata or we can infer from index
            # For now, return None to disable caching (can be enhanced)
            return None
        except (KeyError, AttributeError):
            return None
    
    def collate(self, batch: List[Tuple], dataset_mode: str, ctx: Optional[Dict] = None) -> Tuple[Tuple, Dict]:
        """
        Collate batch of samples into quadtree format.
        
        For each sample:
        - Extracts 4 input timesteps + 1 output timestep
        - Constructs quadtrees (reusing when possible)
        - Collates into batch format with batch_idx
        
        Args:
            batch: List of (sample, context) tuples
            dataset_mode: Dataset mode string
            ctx: Optional context dict
        
        Returns:
            (collated_batch, ctx) tuple where:
            - collated_batch: Tuple of collated data
            - ctx: Context dict with 'quadtree_dicts' and 'batch_idx'
        """
        if ctx is None:
            ctx = {}
        
        assert isinstance(batch, (tuple, list)) and len(batch) > 0
        assert isinstance(batch[0], tuple) and len(batch[0]) == 2
        
        # Unpack batch and context
        samples, sample_ctxs = zip(*batch)
        
        # Use dataset from __init__ or try to get from context
        dataset = self.dataset
        if dataset is None and len(sample_ctxs) > 0:
            # Try to get dataset from first context
            first_ctx = sample_ctxs[0] if isinstance(sample_ctxs[0], dict) else {}
            dataset = first_ctx.get('dataset')
        
        # Determine device from first sample
        device = None
        for sample in samples:
            try:
                # Try to get a tensor from sample to determine device
                timestep = ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="timestep")
                if timestep is not None and isinstance(timestep, torch.Tensor):
                    device = timestep.device
                    break
            except (KeyError, AttributeError):
                continue
        
        if device is None:
            device = torch.device("cpu")
        
        # Collect quadtree dicts for all timesteps across all samples
        quadtree_dicts: List[Dict[str, torch.Tensor]] = []
        batch_indices: List[int] = []
        sample_to_quadtree_map: List[List[int]] = []  # Maps sample_idx -> list of quadtree indices
        
        num_input_timesteps = 4
        num_output_timesteps = 1
        
        for sample_idx, (sample, sample_ctx) in enumerate(zip(samples, sample_ctxs)):
            sample_quadtree_indices = []
            
            # Try to get sample index from context for dataset access
            sample_dataset_idx = None
            if isinstance(sample_ctx, dict):
                sample_dataset_idx = sample_ctx.get('idx') or sample_ctx.get('dataset_idx')
            
            # Get sample identifier for caching
            sample_id = self._get_sample_identifier(sample, dataset_mode)
            
            # Process input timesteps (0-3)
            for t_idx in range(num_input_timesteps):
                cache_key = (sample_id[0], t_idx) if sample_id is not None else None
                
                # Check cache
                if self.cache_quadtrees and cache_key is not None and cache_key in self._quadtree_cache:
                    quadtree_dict = self._quadtree_cache[cache_key]
                else:
                    # Extract grid for this timestep
                    grid = self._extract_grid_from_sample(
                        sample, dataset_mode, t_idx, is_input=True, 
                        dataset=dataset, sample_idx=sample_dataset_idx
                    )
                    if grid is None:
                        raise ValueError(
                            f"Could not extract input grid for timestep {t_idx} from sample {sample_idx}. "
                            f"Ensure dataset provides 'input_fields' in (T, H, W, C) format or pass dataset to collator."
                        )
                    
                    # Construct quadtree
                    quadtree_dict = self._construct_quadtree_from_grid(grid, device)
                    
                    # Cache if enabled
                    if self.cache_quadtrees and cache_key is not None:
                        self._quadtree_cache[cache_key] = quadtree_dict
                
                # Add to batch
                quadtree_idx = len(quadtree_dicts)
                quadtree_dicts.append(quadtree_dict)
                sample_quadtree_indices.append(quadtree_idx)
            
            # Process output timestep (0)
            t_idx = 0
            cache_key = (sample_id[0], num_input_timesteps + t_idx) if sample_id is not None else None
            
            # Check cache
            if self.cache_quadtrees and cache_key is not None and cache_key in self._quadtree_cache:
                quadtree_dict = self._quadtree_cache[cache_key]
            else:
                # Extract grid for output timestep
                grid = self._extract_grid_from_sample(
                    sample, dataset_mode, t_idx, is_input=False,
                    dataset=dataset, sample_idx=sample_dataset_idx
                )
                if grid is None:
                    raise ValueError(
                        f"Could not extract output grid from sample {sample_idx}. "
                        f"Ensure dataset provides 'output_fields' in (T, H, W, C) format or pass dataset to collator."
                    )
                
                # Construct quadtree
                quadtree_dict = self._construct_quadtree_from_grid(grid, device)
                
                # Cache if enabled
                if self.cache_quadtrees and cache_key is not None:
                    self._quadtree_cache[cache_key] = quadtree_dict
            
            # Add to batch
            quadtree_idx = len(quadtree_dicts)
            quadtree_dicts.append(quadtree_dict)
            sample_quadtree_indices.append(quadtree_idx)
            
            sample_to_quadtree_map.append(sample_quadtree_indices)
        
        # Collate quadtree dicts into batch format
        # For input: concatenate all 4 timesteps per sample
        # For output: use single timestep per sample
        
        # Collect features and create batch_idx
        all_features: List[torch.Tensor] = []
        all_batch_idx: List[int] = []
        
        for sample_idx, quadtree_indices in enumerate(sample_to_quadtree_map):
            # Input quadtrees (first 4)
            input_indices = quadtree_indices[:num_input_timesteps]
            for q_idx in input_indices:
                qd = quadtree_dicts[q_idx]
                features = qd['features']  # (M, C)
                M = features.shape[0]
                all_features.append(features)
                all_batch_idx.extend([sample_idx] * M)
            
            # Output quadtree (last 1)
            output_idx = quadtree_indices[num_input_timesteps]
            qd = quadtree_dicts[output_idx]
            features = qd['features']  # (M, C)
            M = features.shape[0]
            all_features.append(features)
            all_batch_idx.extend([sample_idx] * M)
        
        # Concatenate all features
        if len(all_features) > 0:
            all_features_tensor = torch.cat(all_features, dim=0)  # (N_total, C)
            all_batch_idx_tensor = torch.tensor(all_batch_idx, dtype=torch.long, device=device)
        else:
            raise ValueError("No quadtree features collected from batch")
        
        # Store in context for trainer to extract
        ctx['quadtree_features'] = all_features_tensor
        ctx['quadtree_batch_idx'] = all_batch_idx_tensor
        ctx['quadtree_dicts'] = quadtree_dicts
        ctx['sample_to_quadtree_map'] = sample_to_quadtree_map
        ctx['num_input_timesteps'] = num_input_timesteps
        ctx['num_output_timesteps'] = num_output_timesteps
        
        # Also store per-sample quadtree dicts for easier access
        # Format: list of dicts, each dict has 'input_quadtrees' (list of 4) and 'output_quadtree' (1)
        per_sample_quadtrees = []
        for sample_idx, quadtree_indices in enumerate(sample_to_quadtree_map):
            input_quadtrees = [quadtree_dicts[i] for i in quadtree_indices[:num_input_timesteps]]
            output_quadtree = quadtree_dicts[quadtree_indices[num_input_timesteps]]
            per_sample_quadtrees.append({
                'input_quadtrees': input_quadtrees,
                'output_quadtree': output_quadtree
            })
        ctx['per_sample_quadtrees'] = per_sample_quadtrees
        
        # Return standard collated batch format
        # For compatibility, return empty tuple for collated_batch
        # The actual data is in ctx for trainer to extract
        collated_batch = tuple()
        
        return collated_batch, ctx
    
    @property
    def default_collate_mode(self):
        raise RuntimeError("QuadtreeCollator does not support default_collate_mode")
    
    def __call__(self, batch):
        raise NotImplementedError("wrap KDSingleCollator with KDSingleCollatorWrapper")

