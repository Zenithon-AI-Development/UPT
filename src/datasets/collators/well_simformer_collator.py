import einops
import torch
from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper


class WellSimformerCollator(KDSingleCollator):
    """
    Collator for The Well uniform grid datasets.
    Similar to CFDSimformerCollator but simpler since grids are uniform within each batch.
    """
    
    def __init__(self, num_supernodes=None, **kwargs):
        super().__init__(**kwargs)
        self.num_supernodes = num_supernodes
    
    def collate(self, batch, dataset_mode, ctx=None):
        # Unpack batch - same pattern as CFD collator
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)
        batch, old_ctx = zip(*batch)
        ctx = {}
        
        # Extract mesh_pos for all samples
        mesh_pos = []
        mesh_lens = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="mesh_pos")
            # item is (H, W, 2), flatten to (H*W, 2)
            H, W, _ = item.shape
            item_flat = item.reshape(-1, 2)
            mesh_lens.append(len(item_flat))
            mesh_pos.append(item_flat)
        collated_batch = {"mesh_pos": torch.cat(mesh_pos)}
        
        # Select supernodes
        if self.num_supernodes is not None:
            supernodes_offset = 0
            supernode_idxs = []
            for i in range(len(mesh_lens)):
                perm = torch.randperm(mesh_lens[i])[:self.num_supernodes] + supernodes_offset
                supernode_idxs.append(perm)
                supernodes_offset += mesh_lens[i]
            ctx["supernode_idxs"] = torch.cat(supernode_idxs)
        
        # Create batch_idx tensor
        batch_idx = torch.empty(sum(mesh_lens), dtype=torch.long)
        start = 0
        for i in range(len(mesh_lens)):
            end = start + mesh_lens[i]
            batch_idx[start:end] = i
            start = end
        ctx["batch_idx"] = batch_idx
        
        # Extract x
        x = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="x")
            # item is (T, H, W, C), flatten to (H*W, T*C)
            T, H, W, C = item.shape
            item_flat = einops.rearrange(item, "t h w c -> (h w) (t c)")
            assert len(item_flat) == mesh_lens[i]
            x.append(item_flat)
        collated_batch["x"] = torch.cat(x)
        
        # grid_pos = mesh_pos for uniform grids
        collated_batch["grid_pos"] = collated_batch["mesh_pos"]
        
        # Extract query_pos and target
        query_pos = []
        target = []
        for i in range(len(batch)):
            query_item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="query_pos")
            target_item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="target")
            # Flatten both
            query_item_flat = query_item.reshape(-1, 2)
            target_item_flat = target_item.reshape(-1, target_item.shape[-1])
            query_pos.append(query_item_flat)
            target.append(target_item_flat)
        # Pad query_pos for batch format
        from torch.nn.utils.rnn import pad_sequence
        collated_batch["query_pos"] = pad_sequence(query_pos, batch_first=True)
        collated_batch["target"] = torch.cat(target)
        
        # Create unbatch indices accounting for padding (same as CFD collator logic)
        batch_size = len(mesh_lens)
        query_lens = mesh_lens  # Same for uniform grids
        maxlen = max(query_lens)
        unbatch_idx = torch.empty(maxlen * batch_size, dtype=torch.long)
        unbatch_select = []
        unbatch_start = 0
        cur_unbatch_idx = 0
        
        for i in range(len(query_lens)):
            # Add indices for actual data
            unbatch_end = unbatch_start + query_lens[i]
            unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
            unbatch_select.append(cur_unbatch_idx)
            cur_unbatch_idx += 1
            unbatch_start = unbatch_end
            
            # Add indices for padding
            padding = maxlen - query_lens[i]
            if padding > 0:
                unbatch_end = unbatch_start + padding
                unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
                cur_unbatch_idx += 1
                unbatch_start = unbatch_end
        
        unbatch_select = torch.tensor(unbatch_select)
        ctx["unbatch_idx"] = unbatch_idx
        ctx["unbatch_select"] = unbatch_select
        
        # Extract timestep and velocity
        timestep = []
        velocity = []
        for i in range(len(batch)):
            timestep.append(ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="timestep"))
            velocity.append(ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="velocity"))
        collated_batch["timestep"] = torch.stack(timestep)
        collated_batch["velocity"] = torch.stack(velocity)
        
        # mesh_edges, geometry2d are None for uniform grids
        collated_batch["mesh_edges"] = None
        collated_batch["geometry2d"] = None
        
        # Convert to tuple matching dataset_mode order: "x mesh_pos query_pos mesh_edges geometry2d timestep velocity target"
        # This is required by ModeWrapper.get_item
        result_tuple = (
            collated_batch["x"],
            collated_batch["mesh_pos"],
            collated_batch["query_pos"],
            collated_batch["mesh_edges"],
            collated_batch["geometry2d"],
            collated_batch["timestep"],
            collated_batch["velocity"],
            collated_batch["target"],
        )
        
        return result_tuple, ctx

