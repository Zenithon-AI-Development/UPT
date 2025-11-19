"""
Slot-based collator: assigns cells to slots and creates slot representation.

Based on CfdSimformerCollator but adds slot assignment step.
"""

from typing import Optional

import einops
import torch
from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate

from slot_upt.subnodes.interfaces import SubnodeBatch, make_backend
from slot_upt.subnodes.amr_backend import AmrBackendConfig


class CfdSlotSimformerCollator(KDSingleCollator):
    """
    Slot-based collator for CFD simulations.
    
    Assigns cells to slots using voxel grid, then creates slot representation.
    """
    
    def __init__(
        self,
        M=None,
        N=None,
        num_supernodes=None,
        ndim=2,
        subnode_backend: str = "voxel",
        backend_kwargs: Optional[dict] = None,
        amr_field_spec=None,
        amr_run_mode: str = "t0",
        amr_use_mask: bool = False,
        amr_mask_channel: int = 0,
        amr_field_channels: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            M: Number of supernodes (voxel grid size)
            N: Number of slots per supernode
            num_supernodes: (deprecated, use M instead)
            ndim: Spatial dimension (2 or 3)
        """
        super().__init__(**kwargs)
        self.M = M or num_supernodes
        self.N = N
        self.ndim = ndim
        self.subnode_backend = subnode_backend
        backend_kwargs = dict(backend_kwargs or {})
        self.amr_field_spec = amr_field_spec
        self.amr_run_mode = amr_run_mode
        self.amr_use_mask = amr_use_mask
        self.amr_mask_channel = amr_mask_channel
        self.amr_field_channels = amr_field_channels

        if self.M is None or self.N is None:
            raise ValueError("M and N must be specified")
        
        if self.subnode_backend == "amr":
            backend_kwargs = {"config": AmrBackendConfig(**backend_kwargs)}

        self.backend = make_backend(
            self.subnode_backend,
            M=self.M,
            N=self.N,
            ndim=self.ndim,
            **backend_kwargs,
        )
    
    def collate(self, batch, dataset_mode, ctx=None):
        """
        Collate batch and assign cells to slots.
        
        Args:
            batch: List of samples
            dataset_mode: Dataset mode string
            ctx: Context dict
        
        Returns:
            collated_batch: Tuple of collated tensors
            ctx: Updated context dict
        """
        # make sure that batch was not collated
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)
        batch, ctx = zip(*batch)
        # properties in context can have variable shapes (e.g. perm) -> delete ctx
        ctx = {}
        # collect collated properties
        collated_batch = {}
        
        # to sparse tensor: batch_size * (num_mesh_points, ndim) -> (batch_size * num_mesh_points, ndim)
        mesh_pos = []
        mesh_lens = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="mesh_pos")
            mesh_lens.append(len(item))
            mesh_pos.append(item)
        collated_batch["mesh_pos"] = torch.concat(mesh_pos)
        
        # create batch_idx tensor
        batch_idx = torch.empty(sum(mesh_lens), dtype=torch.long)
        start = 0
        cur_batch_idx = 0
        for i in range(len(mesh_lens)):
            end = start + mesh_lens[i]
            batch_idx[start:end] = cur_batch_idx
            start = end
            cur_batch_idx += 1
        ctx["batch_idx"] = batch_idx
        
        # batch_size * (num_mesh_points, num_input_timesteps * num_channels) ->
        # (batch_size * num_mesh_points, num_input_timesteps * num_channels)
        x = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="x")
            assert len(item) == mesh_lens[i]
            x.append(item)
        collated_batch["x"] = torch.concat(x)
        
        # Optional per-cell AMR levels
        def _mode_has_item(mode: str, item: str) -> bool:
            return item in mode.split(" ")

        levels_list = []
        have_levels = False
        has_amr = _mode_has_item(dataset_mode, "amr_level")
        has_level = _mode_has_item(dataset_mode, "level")
        if has_amr or has_level:
            have_levels = True
            for i in range(len(batch)):
                lvl = None
                if has_amr:
                    lvl = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="amr_level")
                if lvl is None and has_level:
                    lvl = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="level")
                if lvl is None:
                    have_levels = False
                    break
                levels_list.append(lvl)
        levels_per_cell = None
        if have_levels and len(levels_list) > 0:
            levels_per_cell = torch.concat(levels_list)

        # query_pos to sparse tensor: batch_size * (num_mesh_points, ndim) -> (batch_size * num_mesh_points, ndim)
        # target to sparse tensor: batch_size * (num_mesh_points, dim) -> (batch_size * num_mesh_points, dim)
        query_pos = []
        query_lens = []
        target = []
        geometry2d = []
        constant_fields = []
        constant_available = True
        for i in range(len(batch)):
            query_pos_item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="query_pos")
            target_item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="target")
            assert len(query_pos_item) == len(target_item)
            query_lens.append(len(query_pos_item))
            query_pos.append(query_pos_item)
            target.append(target_item)
            geometry2d.append(ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="geometry2d"))
            try:
                const_item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="constant_fields")
                constant_fields.append(const_item)
            except (AttributeError, KeyError):
                constant_available = False
        collated_batch["query_pos"] = pad_sequence(query_pos, batch_first=True)
        collated_batch["target"] = torch.concat(target)
        collated_batch["geometry2d"] = default_collate(geometry2d)
        if constant_available and len(constant_fields) > 0:
            collated_batch["constant_fields"] = default_collate(constant_fields)
        else:
            collated_batch["constant_fields"] = None
        
        # create unbatch_idx tensors (unbatch via torch_geometrics.utils.unbatch)
        # e.g. batch_size=2, num_points=[2, 3] -> unbatch_idx=[0, 0, 1, 2, 2, 2] unbatch_select=[0, 2]
        # then unbatching can be done via unbatch(dense, unbatch_idx)[unbatch_select]
        batch_size = len(query_lens)
        maxlen = max(query_lens)
        unbatch_idx = torch.empty(maxlen * batch_size, dtype=torch.long)
        unbatch_select = []
        unbatch_start = 0
        cur_unbatch_idx = 0
        for i in range(len(query_lens)):
            unbatch_end = unbatch_start + query_lens[i]
            unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
            unbatch_select.append(cur_unbatch_idx)
            cur_unbatch_idx += 1
            unbatch_start = unbatch_end
            padding = maxlen - query_lens[i]
            if padding > 0:
                unbatch_end = unbatch_start + padding
                unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
                cur_unbatch_idx += 1
                unbatch_start = unbatch_end
        unbatch_select = torch.tensor(unbatch_select)
        ctx["unbatch_idx"] = unbatch_idx
        ctx["unbatch_select"] = unbatch_select
        
        # sparse mesh_edges:  batch_size * (num_points, ndim) -> (batch_size * num_points, ndim)
        mesh_edges = []
        mesh_edges_offset = 0
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="mesh_edges")
            # if None -> create graph on GPU
            if item is None:
                break
            idx = item + mesh_edges_offset
            mesh_edges.append(idx)
            mesh_edges_offset += mesh_lens[i]
        if len(mesh_edges) > 0:
            # noinspection PyTypedDict
            collated_batch["mesh_edges"] = torch.concat(mesh_edges)
        else:
            collated_batch["mesh_edges"] = None

        num_timesteps = self._infer_num_timesteps(collated_batch)

        backend_ctx = self._build_backend_context(
            collated_batch=collated_batch,
            mesh_lens=mesh_lens,
            batch_idx=batch_idx,
            num_timesteps=num_timesteps,
            levels_per_cell=levels_per_cell,
        )
        subnode_batch = self.backend.generate(backend_ctx)
        subnode_batch = self._broadcast_subnode_batch(subnode_batch, num_timesteps)

        ctx["subnode_feats"] = subnode_batch.subnode_feats
        ctx["subnode_mask"] = subnode_batch.subnode_mask
        ctx["slot2cell"] = subnode_batch.slot2cell
        ctx["slot_positions"] = subnode_batch.slot_positions
        if subnode_batch.subnode_level is not None:
            ctx["subnode_level"] = subnode_batch.subnode_level
        if subnode_batch.subnode_pos is not None:
            ctx["subnode_pos"] = subnode_batch.subnode_pos

        # normal collation for other properties (timestep, velocity, geometry2d)
        result = []
        for item in dataset_mode.split(" "):
            if item in collated_batch:
                if item == "velocity" and collated_batch[item].ndim == 0:
                    result.append(collated_batch[item].unsqueeze(0))
                    continue
                result.append(collated_batch[item])
            else:
                result.append(
                    default_collate([
                        ModeWrapper.get_item(mode=dataset_mode, batch=sample, item=item)
                        for sample in batch
                    ])
                )
        
        return tuple(result), ctx
    
    def _infer_num_timesteps(self, collated_batch: dict) -> int:
        x = collated_batch["x"]
        tgt = collated_batch.get("target")
        if x.ndim != 2:
            return 1
        if tgt is not None and tgt.ndim == 2 and tgt.shape[1] > 0:
            channels = tgt.shape[1]
            if channels > 0 and x.shape[1] % channels == 0:
                return max(1, x.shape[1] // channels)
        return 1

    def _build_backend_context(
        self,
        collated_batch: dict,
        mesh_lens,
        batch_idx: torch.Tensor,
        num_timesteps: int,
        levels_per_cell: Optional[torch.Tensor],
    ):
        ctx = {
            "mesh_pos": collated_batch["mesh_pos"],
            "features": collated_batch["x"],
            "batch_idx": batch_idx,
            "num_timesteps": num_timesteps,
        }
        if levels_per_cell is not None:
            ctx["levels_per_cell"] = levels_per_cell

        if self.subnode_backend == "amr":
            amr_ctx = self._build_amr_context(collated_batch, mesh_lens, num_timesteps)
            ctx.update(amr_ctx)

        return ctx

    def _build_amr_context(
        self,
        collated_batch: dict,
        mesh_lens,
        num_timesteps: int,
    ) -> dict:
        geometry = collated_batch.get("geometry2d")
        if geometry is None:
            raise ValueError("AMR backend requires 'geometry2d' to infer grid resolution.")
        B = geometry.shape[0]
        H = geometry.shape[-2]
        W = geometry.shape[-1]

        features_concat = collated_batch["x"]
        device = features_concat.device
        dtype = features_concat.dtype

        if self.amr_field_channels is not None:
            channels = int(self.amr_field_channels)
        else:
            target = collated_batch.get("target")
            if target is not None and target.ndim == 2 and target.shape[1] > 0:
                channels = int(target.shape[1])
            else:
                channels = int(features_concat.shape[1] // max(num_timesteps, 1))
        if channels <= 0:
            raise ValueError("Unable to infer field channel count for AMR backend.")

        inferred_timesteps = max(1, features_concat.shape[1] // channels)
        if inferred_timesteps != num_timesteps:
            num_timesteps = inferred_timesteps

        time_indices = self._select_time_indices(num_timesteps)

        amr_inputs = []
        amr_masks = [] if self.amr_use_mask else None
        offset = 0
        constant_fields = collated_batch.get("constant_fields")
        spatial_shape = None

        for b in range(B):
            length = mesh_lens[b]
            x_flat = features_concat[offset:offset + length]
            offset += length

            x_spatial = x_flat.view(H, W, num_timesteps, channels).permute(2, 0, 1, 3).contiguous()
            selected = x_spatial[time_indices]
            selected = self._apply_amr_field_spec(selected)
            amr_inputs.append(selected.contiguous())
            if spatial_shape is None:
                spatial_shape = (selected.shape[1], selected.shape[2])

            if self.amr_use_mask:
                if constant_fields is None:
                    raise ValueError("AMR mask requested but constant_fields not available.")
                mask_tensor = constant_fields[b]
                if mask_tensor is None or mask_tensor.numel() == 0:
                    raise ValueError("AMR mask requested but constant_fields is empty.")
                if mask_tensor.dim() == 3:
                    mask_tensor = mask_tensor[..., self.amr_mask_channel]
                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(-1)
                mask_tensor = mask_tensor.to(device=device, dtype=dtype)
                mask_tensor = mask_tensor.unsqueeze(0).expand(selected.shape[0], -1, -1, -1).contiguous()
                amr_masks.append(mask_tensor)

        amr_inputs_tensor = torch.stack(amr_inputs, dim=0)
        context = {
            "amr_inputs": amr_inputs_tensor,
            "amr_feature_field": amr_inputs_tensor,
            "spatial_shape": torch.tensor(spatial_shape or (H, W), device=device, dtype=torch.long),
        }
        if amr_masks is not None:
            context["amr_mask"] = torch.stack(amr_masks, dim=0)
        return context

    def _select_time_indices(self, num_timesteps: int):
        mode = (self.amr_run_mode or "t0").lower()
        if mode == "t0":
            return [0]
        if mode == "all":
            return list(range(num_timesteps))
        if mode.startswith("t"):
            try:
                idx = int(mode[1:])
            except ValueError as exc:
                raise ValueError(f"Invalid amr_run_mode '{self.amr_run_mode}'.") from exc
            if not 0 <= idx < num_timesteps:
                raise ValueError(f"amr_run_mode index {idx} out of bounds for {num_timesteps} timesteps.")
            return [idx]
        raise ValueError(f"Unsupported amr_run_mode '{self.amr_run_mode}'.")

    def _apply_amr_field_spec(self, tensor: torch.Tensor) -> torch.Tensor:
        spec = self.amr_field_spec
        if spec is None:
            return tensor
        if isinstance(spec, str):
            key = spec.lower()
            if key in {"real_imag", "pressure_real_imag"}:
                return tensor[..., :2]
            if key in {"magnitude", "pressure_magnitude"}:
                if tensor.shape[-1] < 2:
                    raise ValueError("Cannot compute magnitude with less than two channels.")
                real = tensor[..., 0]
                imag = tensor[..., 1]
                mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
                return mag.unsqueeze(-1)
            raise ValueError(f"Unknown amr_field_spec '{spec}'.")
        if isinstance(spec, (list, tuple)):
            indices = list(spec)
            return tensor[..., indices]
        if isinstance(spec, dict):
            spec_type = spec.get("type", "").lower()
            if spec_type == "channels":
                indices = spec.get("indices")
                if indices is None:
                    raise ValueError("amr_field_spec dict with type='channels' requires 'indices'.")
                return tensor[..., indices]
            if spec_type == "magnitude":
                indices = spec.get("indices", [0, 1])
                if len(indices) < 2:
                    raise ValueError("Magnitude spec requires at least two indices.")
                real = tensor[..., indices[0]]
                imag = tensor[..., indices[1]]
                mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
                return mag.unsqueeze(-1)
        raise ValueError(f"Unsupported amr_field_spec '{spec}'.")

    def _broadcast_subnode_batch(self, batch: SubnodeBatch, target_t: int) -> SubnodeBatch:
        current_t = batch.subnode_feats.shape[1]
        if current_t == target_t:
            return batch
        if current_t != 1:
            raise ValueError(f"Cannot broadcast SubnodeBatch with T={current_t} to target {target_t}.")

        def _repeat(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            repeat_shape = [1, target_t] + [1] * (tensor.dim() - 2)
            return tensor.repeat(*repeat_shape)

        return SubnodeBatch(
            subnode_feats=_repeat(batch.subnode_feats),
            subnode_mask=_repeat(batch.subnode_mask),
            slot2cell=_repeat(batch.slot2cell),
            slot_positions=batch.slot_positions,
            subnode_level=_repeat(batch.subnode_level),
            subnode_pos=_repeat(batch.subnode_pos),
        )
    
    @property
    def default_collate_mode(self):
        raise RuntimeError
    
    def __call__(self, batch):
        raise NotImplementedError("wrap KDSingleCollator with KDSingleCollatorWrapper")

