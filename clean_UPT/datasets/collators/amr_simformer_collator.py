import torch
from torch.utils.data import default_collate

import einops
from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper


class AmrSimformerCollator(KDSingleCollator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, batch):
        raise NotImplementedError("wrap KDSingleCollator with KDSingleCollatorWrapper")

    def collate(self, batch, dataset_mode, ctx=None):
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)
        batch, ctx = zip(*batch)

        if ctx is None:
            ctx = [{} for _ in range(len(batch))]

        collated_batch = {}
        lens = None

        # gather auxiliary context (time/traj indices) if available
        time_idx_list = []
        traj_idx_list = []
        for i in range(len(batch)):
            cur_ctx = ctx[i] or {}
            time_idx = cur_ctx.get("time_idx")
            if time_idx is None:
                time_idx = torch.tensor(0)
            elif not torch.is_tensor(time_idx):
                time_idx = torch.tensor(time_idx)
            time_idx_list.append(time_idx)

            traj_idx = cur_ctx.get("traj_idx")
            if traj_idx is None:
                traj_idx = torch.tensor(0)
            elif not torch.is_tensor(traj_idx):
                traj_idx = torch.tensor(traj_idx)
            traj_idx_list.append(traj_idx)

        ctx_dict = {
            "time_idx": torch.stack(time_idx_list),
            "traj_idx": torch.stack(traj_idx_list),
        }

        if ModeWrapper.has_item(mode=dataset_mode, item="x"):
            x = [ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="x") for sample in batch]
            if lens is None:
                lens = [xx.size(2) for xx in x]
            x_flat = einops.rearrange(torch.concat(x, dim=2), "timesteps channels flat -> flat timesteps channels")
            collated_batch["x"] = x_flat
        else:
            raise NotImplementedError

        pos_items = ("curr_pos", "target_pos_encode")
        for pos_item in pos_items:
            if ModeWrapper.has_item(mode=dataset_mode, item=pos_item):
                pos = [ModeWrapper.get_item(mode=dataset_mode, batch=sample, item=pos_item) for sample in batch]
                if lens is None:
                    lens = [c.size(0) for c in pos]
                flat_pos = torch.concat(pos)
                collated_batch[pos_item] = flat_pos

        assert ModeWrapper.has_item(mode=dataset_mode, item="edge_index")
        edge_index = []
        edge_index_offset = 0
        for i in range(len(batch)):
            idx = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="edge_index") + edge_index_offset
            edge_index.append(idx)
            edge_index_offset += lens[i]
        collated_batch["edge_index"] = torch.concat(edge_index)

        if ModeWrapper.has_item(mode=dataset_mode, item="edge_index_target"):
            edge_index_target = []
            edge_index_offset = 0
            for i in range(len(batch)):
                idx = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="edge_index_target") + edge_index_offset
                edge_index_target.append(idx)
                edge_index_offset += lens[i]
            collated_batch["edge_index_target"] = torch.concat(edge_index_target)

        if ModeWrapper.has_item(mode=dataset_mode, item="edge_features"):
            edge_features = []
            edge_index_offset = 0
            for i in range(len(batch)):
                idx = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="edge_features") + edge_index_offset
                edge_features.append(idx)
                edge_index_offset += lens[i]
            collated_batch["edge_features"] = torch.concat(edge_features)

        if ModeWrapper.has_item(mode=dataset_mode, item="perm"):
            perm_batch = []
            for i in range(len(batch)):
                perm_info = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="perm")
                if perm_info is None:
                    continue
                perm, n_particles = perm_info
                perm_batch.append(perm + i * n_particles)
            if len(perm_batch) > 0:
                perm_batch = torch.concat(perm_batch)
                collated_batch["perm"] = perm_batch

        if lens is not None:
            batch_size = len(lens)
            batch_idx = torch.empty(sum(lens), dtype=torch.long)
            start = 0
            for i, ln in enumerate(lens):
                end = start + ln
                batch_idx[start:end] = i
                start = end
            ctx_dict["batch_idx"] = batch_idx

            target = [ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="target_acc") for sample in batch]
            lens_target = [tt.size(0) for tt in target]
            maxlen = max(lens_target)
            unbatch_idx = torch.empty(maxlen * batch_size, dtype=torch.long)
            unbatch_select = []
            unbatch_start = 0
            cur_unbatch_idx = 0
            for ln in lens_target:
                unbatch_end = unbatch_start + ln
                unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
                unbatch_select.append(cur_unbatch_idx)
                cur_unbatch_idx += 1
                unbatch_start = unbatch_end
                padding = maxlen - ln
                if padding > 0:
                    unbatch_end = unbatch_start + padding
                    unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
                    cur_unbatch_idx += 1
                    unbatch_start = unbatch_end
            ctx_dict["unbatch_idx"] = unbatch_idx
            ctx_dict["unbatch_select"] = torch.tensor(unbatch_select)

        result = []
        for item in dataset_mode.split(" "):
            if item in collated_batch:
                result.append(collated_batch[item])
            else:
                result.append(
                    default_collate([
                        ModeWrapper.get_item(mode=dataset_mode, batch=sample, item=item)
                        for sample in batch
                    ])
                )

        return tuple(result), ctx_dict

    @property
    def default_collate_mode(self):
        raise RuntimeError



