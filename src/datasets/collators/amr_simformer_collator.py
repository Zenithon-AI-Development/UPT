import einops
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate

from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper
from torch.utils.data import default_collate
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
from torch_geometric.nn.pool import radius_graph, radius


class AMRSimformerCollator(KDSingleCollator):
    """Collator for AMR dataset. Based on LagrangianSimformerCollator."""
    
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError("wrap KDSingleCollator with KDSingleCollatorWrapper")

    def collate(self, batch, dataset_mode, ctx=None):
        """Collate AMR samples similar to Lagrangian collator."""
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)
        batch, old_ctx = zip(*batch)
        
        # For AMR, ctx is the tuple (curr_pos, x, target) from __getitem__
        # Don't extract time_idx/traj_idx since they're not in ctx
        ctx = {}
        
        collated_batch = {}
        lens = None
        
        if ModeWrapper.has_item(mode=dataset_mode, item="x"):
            # x: batch_size * (num_input_timesteps, num_channels, num_points)
            x = [ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="x") for sample in batch]
            if lens is None:
                lens = [xx.size(2) for xx in x]
            x_flat = einops.rearrange(torch.concat(x, dim=2), "timesteps channels flat -> flat timesteps channels")
            collated_batch["x"] = x_flat
        else:
            raise NotImplementedError
        
        # Collate positions
        pos_items = ("curr_pos", "target_pos_encode")
        for pos_item in pos_items:
            if ModeWrapper.has_item(mode=dataset_mode, item=pos_item):
                pos = [ModeWrapper.get_item(mode=dataset_mode, batch=sample, item=pos_item) for sample in batch]
                if lens is None:
                    lens = [c.size(0) for c in pos]
                flat_pos = torch.concat(pos)
                collated_batch[pos_item] = flat_pos

        # Edge indices handled by collator (return None from getitem methods)
        assert ModeWrapper.has_item(mode=dataset_mode, item="edge_index")
        edge_index = []
        edge_index_offset = 0
        for i in range(len(batch)):
            idx = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="edge_index")
            if idx is not None:
                edge_index.append(idx + edge_index_offset)
            edge_index_offset += lens[i]
        if len(edge_index) > 0:
            collated_batch["edge_index"] = torch.concat(edge_index)
        else:
            collated_batch["edge_index"] = None

        if ModeWrapper.has_item(mode=dataset_mode, item="edge_index_target"):
            edge_index_target = []
            edge_index_offset = 0
            for i in range(len(batch)):
                idx = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="edge_index_target")
                if idx is not None:
                    edge_index_target.append(idx + edge_index_offset)
                edge_index_offset += lens[i]
            if len(edge_index_target) > 0:
                collated_batch["edge_index_target"] = torch.concat(edge_index_target)
            else:
                collated_batch["edge_index_target"] = None

        if ModeWrapper.has_item(mode=dataset_mode, item="perm"):
            perm_batch = []
            for i in range(len(batch)):
                perm_item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="perm")
                if perm_item is not None:
                    perm, n_particles = perm_item
                    perm_batch.append(perm + i*n_particles)
            if len(perm_batch) > 0:
                collated_batch["perm"] = torch.concat(perm_batch)

        if lens is not None and ctx is not None:
            batch_size = len(lens)
            maxlen = max(lens)
            
            # Create batch_idx: which batch each point belongs to
            batch_idx = torch.cat([torch.full((lens[i],), i, dtype=torch.long) for i in range(batch_size)])
            ctx["batch_idx"] = batch_idx
            
            # Create unbatch_idx for variable-size batches
            unbatch_idx = torch.zeros(batch_size * maxlen, dtype=torch.long)
            unbatch_start = 0
            cur_unbatch_idx = 0
            unbatch_select = []
            for i in range(batch_size):
                unbatch_end = unbatch_start + lens[i]
                unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
                unbatch_select.append(cur_unbatch_idx)
                cur_unbatch_idx += 1
                unbatch_start = unbatch_end
                padding = maxlen - lens[i]
                if padding > 0:
                    unbatch_end = unbatch_start + padding
                    unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
                    cur_unbatch_idx += 1
                    unbatch_start = unbatch_end
            unbatch_select = torch.tensor(unbatch_select)
            ctx["unbatch_idx"] = unbatch_idx
            ctx["unbatch_select"] = unbatch_select

        # Default collation for other properties
        result = []
        for item in dataset_mode.split(" "):
            if item in collated_batch:
                result.append(collated_batch[item])
            else:
                items_to_collate = [
                    ModeWrapper.get_item(mode=dataset_mode, batch=sample, item=item)
                    for sample in batch
                ]
                # Skip None values (e.g., prev_acc not used in AMR)
                if all(it is None for it in items_to_collate):
                    result.append(None)
                else:
                    result.append(default_collate(items_to_collate))

        return tuple(result), ctx

    @property
    def default_collate_mode(self):
        return "x curr_pos curr_pos_full edge_index edge_index_target timestep target_acc target_pos prev_pos prev_acc target_pos_encode perm target_vel"
