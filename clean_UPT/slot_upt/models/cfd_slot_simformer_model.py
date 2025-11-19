"""
Slot-based UPT model: wraps vanilla UPT with fixed [M, N] slot layer.

Based on CfdSimformerModel but replaces encoder/decoder with slot-based versions.
Latent core and conditioner remain unchanged.
"""

import einops
import torch
import torch.nn.functional as F

from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create
from utils.amp_utils import NoopContext
from slot_upt.slot_assignment import scatter_slots_to_cells


class CfdSlotSimformerModel(CompositeModelBase):
    """
    Slot-based UPT model for CFD simulations.
    
    Architecture:
    1. Assign cells to slots (in collator)
    2. Encode: subnode_feats → slot aggregator → encoder → latent
    3. Latent core: (unchanged, same as baseline)
    4. Decode: latent → decoder → slot splitter → subnode_feats
    5. Scatter: subnode_feats → original cells via slot2cell
    """
    
    def __init__(
        self,
        encoder,
        latent,
        decoder,
        force_decoder_fp32=True,
        conditioner=None,
        geometry_encoder=None,
        M=None,
        N=None,
        **kwargs,
    ):
        num_input_timesteps = kwargs.pop("num_input_timesteps", None)
        super().__init__(**kwargs)
        self.force_decoder_fp32 = force_decoder_fp32
        
        # Get M, N from static_ctx or kwargs
        self.M = M or self.static_ctx.get("num_supernodes", None)
        self.N = N or self.static_ctx.get("num_slots_per_supernode", None)
        assert self.M is not None and self.N is not None, "M and N must be specified"

        self.num_input_timesteps = num_input_timesteps or self.static_ctx.get("num_input_timesteps")
        if self.num_input_timesteps is None:
            self.num_input_timesteps = 1
        
        # Store in static_ctx for encoder/decoder
        self.static_ctx["num_supernodes"] = self.M
        self.static_ctx["num_slots_per_supernode"] = self.N
        self.static_ctx["num_input_timesteps"] = self.num_input_timesteps
        
        common_kwargs = dict(
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        
        # timestep embed (reuse existing)
        self.conditioner = create(
            conditioner,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.input_shape,
        )
        
        # geometry encoder (optional, reuse existing)
        self.geometry_encoder = create(
            geometry_encoder,
            model_from_kwargs,
            **common_kwargs,
        )
        
        # set static_ctx["num_static_tokens"]
        if self.geometry_encoder is not None:
            assert self.geometry_encoder.output_shape is not None and len(self.geometry_encoder.output_shape) == 2
            self.static_ctx["num_static_tokens"] = self.geometry_encoder.output_shape[0]
        else:
            self.static_ctx["num_static_tokens"] = 0
        
        # set static_ctx["dim"]
        if self.conditioner is not None:
            self.static_ctx["dim"] = self.conditioner.dim
        elif self.geometry_encoder is not None:
            self.static_ctx["dim"] = self.geometry_encoder.output_shape[1]
        else:
            self.static_ctx["dim"] = latent["kwargs"]["dim"]
        
        # encoder (slot-based)
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            **common_kwargs,
            M=self.M,
            N=self.N,
        )
        assert self.encoder.output_shape is not None
        
        # dynamics (latent core - unchanged)
        self.latent = create(
            latent,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
            **common_kwargs,
        )
        
        # decoder (slot-based)
        self.decoder = create(
            decoder,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.latent.output_shape,
            output_shape=self.output_shape,
            M=self.M,
            N=self.N,
        )
    
    @property
    def submodels(self):
        return dict(
            **(dict(conditioner=self.conditioner) if self.conditioner is not None else {}),
            **(dict(geometry_encoder=self.geometry_encoder) if self.geometry_encoder is not None else {}),
            encoder=self.encoder,
            latent=self.latent,
            decoder=self.decoder,
        )
    
    def forward(
        self,
        subnode_feats,
        subnode_mask,
        slot2cell,
        timestep,
        velocity,
        batch_idx,
        unbatch_idx,
        unbatch_select,
        target=None,
        detach_reconstructions=True,
        reconstruct_prev_x=False,
        reconstruct_dynamics=False,
    ):
        """
        Forward pass: slot-based encoding → latent core → slot-based decoding.
        
        Args:
            subnode_feats: [B, T, M, N, C] slot features
            subnode_mask: [B, T, M, N] mask
            slot2cell: [B, T, M, N] cell indices (-1 for empty slots)
            timestep: [B] or [B*T] timestep indices
            velocity: [B] or [B*T] velocity features
            batch_idx: [B*T*M] batch indices
            unbatch_idx: (for compatibility, not used in slot model)
            unbatch_select: (for compatibility, not used in slot model)
            target: (optional, for reconstruction losses)
            detach_reconstructions: Whether to detach in reconstruction
            reconstruct_prev_x: Whether to reconstruct previous timestep
            reconstruct_dynamics: Whether to reconstruct dynamics
        
        Returns:
            outputs: Dict with 'x_hat' (scattered to original cells) and other outputs
        """
        outputs = {}
        B, T, M, N, C = subnode_feats.shape
        device = subnode_feats.device
        
        # encode timestep t
        if self.conditioner is not None:
            # Handle time-varying timestep
            if timestep.ndim == 1 and timestep.shape[0] == B:
                # Expand to [B*T] if needed
                timestep_expanded = timestep.unsqueeze(1).expand(B, T).reshape(B * T)
            else:
                timestep_expanded = timestep
            condition = self.conditioner(timestep=timestep_expanded, velocity=velocity)
        else:
            condition = None
        outputs["condition"] = condition
        
        # encode geometry (optional, deprecated)
        if self.geometry_encoder is not None:
            raise NotImplementedError("geometry_encoder is deprecated in slot model")
        else:
            static_tokens = None
        
        # encode data: subnode_feats → slot aggregator → encoder → latent
        # [B, T, M, N, C] → [B*T, n_latent, d]
        prev_dynamics = self.encoder(
            subnode_feats=subnode_feats,
            subnode_mask=subnode_mask,
            batch_idx=batch_idx,
            condition=condition,
            static_tokens=static_tokens,
        )
        outputs["prev_dynamics"] = prev_dynamics
        
        # predict current latent (dynamic_{t-1} → dynamic_t)
        # Latent core is unchanged - same as baseline
        dynamics = self.latent(
            prev_dynamics,
            condition=condition,
            static_tokens=static_tokens,
        )
        outputs["dynamics"] = dynamics
        
        # decode next_latent to next_data: latent → decoder → slot splitter → subnode_feats
        if self.force_decoder_fp32:
            with torch.autocast(device_type=str(dynamics.device).split(":")[0], enabled=False):
                subnode_feats_hat = self.decoder(
                    dynamics.float(),
                    subnode_mask=subnode_mask,
                    batch_idx=batch_idx,
                    condition=condition.float(),
                )
        else:
            subnode_feats_hat = self.decoder(
                dynamics,
                subnode_mask=subnode_mask,
                batch_idx=batch_idx,
                condition=condition,
            )
        
        # Scatter subnode_feats back to original cells
        # [B, T, M, N, C_out] → [B*K, T*C_out] or [B*K, T, C_out]
        num_cells = slot2cell.max().item() + 1
        x_hat = scatter_slots_to_cells(subnode_feats_hat, slot2cell, num_cells)
        outputs["x_hat"] = x_hat
        
        # reconstruct dynamics_t from (x_{t-1}, \hat{x}_t)
        if reconstruct_dynamics:
            # calculate t+1
            if self.conditioner is not None:
                next_timestep = torch.clamp_max(timestep_expanded + 1, max=self.conditioner.num_total_timesteps - 1)
                next_condition = self.conditioner(timestep=next_timestep, velocity=velocity)
            else:
                next_condition = None
            
            # reconstruct dynamics_t
            # For slot model, we need to re-encode the predicted subnode_feats
            # This requires re-assigning cells to slots, which is complex
            # For now, we'll skip this or use a simplified version
            # TODO: Implement proper reconstruction for slot model
            outputs["dynamics_hat"] = prev_dynamics  # Placeholder
        
        # reconstruct x_{t-1} from dynamic_{t-1}
        if reconstruct_prev_x:
            # calculate t-1
            if self.conditioner is not None:
                prev_timestep = F.relu(timestep_expanded - 1)
                prev_condition = self.conditioner(timestep=prev_timestep, velocity=velocity)
            else:
                prev_condition = None
            
            # reconstruct prev_x_hat
            if self.force_decoder_fp32:
                with torch.autocast(device_type=str(device).split(":")[0], enabled=False):
                    prev_subnode_feats_hat = self.decoder(
                        prev_dynamics.detach().float() if detach_reconstructions else prev_dynamics.float(),
                        subnode_mask=subnode_mask,
                        batch_idx=batch_idx,
                        condition=prev_condition.float() if prev_condition is not None else None,
                    )
            else:
                prev_subnode_feats_hat = self.decoder(
                    prev_dynamics.detach() if detach_reconstructions else prev_dynamics,
                    subnode_mask=subnode_mask,
                    batch_idx=batch_idx,
                    condition=prev_condition,
                )
            
            # Scatter to cells
            prev_x_hat = scatter_slots_to_cells(prev_subnode_feats_hat, slot2cell, num_cells)
            outputs["prev_x_hat"] = prev_x_hat
        
        return outputs
    
    @torch.no_grad()
    def rollout(
        self,
        subnode_feats,
        subnode_mask,
        slot2cell,
        velocity,
        batch_idx,
        num_rollout_timesteps=None,
        mode="image",
        intermediate_results=True,
        clip=None,
    ):
        """
        Autoregressive rollout for multiple timesteps.
        
        Args:
            subnode_feats: [B, T, M, N, C] initial slot features
            subnode_mask: [B, T, M, N] mask
            slot2cell: [B, T, M, N] cell indices
            velocity: [B] or [B*T] velocity features
            batch_idx: [B*T*M] batch indices
            num_rollout_timesteps: Number of rollout steps
            mode: "image" (autoregressive) or "latent" (latent space)
            intermediate_results: Whether to return intermediate results
            clip: Optional clipping value
        
        Returns:
            predictions: List of predictions for each timestep
        """
        B, T, M, N, C = subnode_feats.shape
        device = subnode_feats.device
        
        # check num_rollout_timesteps
        max_timesteps = self.data_container.get_dataset().getdim_timestep()
        num_rollout_timesteps = num_rollout_timesteps or max_timesteps
        assert 0 < num_rollout_timesteps <= max_timesteps
        
        # setup
        x_hats = []
        timestep = torch.zeros(B, device=device, dtype=torch.long)
        condition = None
        
        if mode == "latent":
            # rollout via latent (depending on dynamics_transformer, encoder is either not used at all or only for t0)
            # initial forward
            if self.conditioner is not None:
                timestep_expanded = timestep.unsqueeze(1).expand(B, T).reshape(B * T)
                condition = self.conditioner(timestep=timestep_expanded, velocity=velocity)
            
            # encode mesh
            dynamics = self.encoder(
                subnode_feats=subnode_feats,
                subnode_mask=subnode_mask,
                batch_idx=batch_idx,
                condition=condition,
            )
            
            # predict initial latent
            dynamics = self.latent(
                dynamics,
                condition=condition,
            )
            
            if intermediate_results:
                if self.force_decoder_fp32:
                    with torch.autocast(device_type=str(device).split(":")[0], enabled=False):
                        subnode_feats_hat = self.decoder(
                            dynamics.float(),
                            subnode_mask=subnode_mask,
                            batch_idx=batch_idx,
                            condition=condition.float(),
                        )
                else:
                    subnode_feats_hat = self.decoder(
                        dynamics,
                        subnode_mask=subnode_mask,
                        batch_idx=batch_idx,
                        condition=condition,
                    )
                
                num_cells = slot2cell.max().item() + 1
                x_hat = scatter_slots_to_cells(subnode_feats_hat, slot2cell, num_cells)
                x_hats.append(x_hat)
            
            # rollout
            for i in range(num_rollout_timesteps - 1):
                # encode timestep
                if self.conditioner is not None:
                    timestep = timestep + 1
                    timestep_expanded = timestep.unsqueeze(1).expand(B, T).reshape(B * T)
                    condition = self.conditioner(timestep=timestep_expanded, velocity=velocity)
                
                # predict next latent
                dynamics = self.latent(
                    dynamics,
                    condition=condition,
                )
                
                if intermediate_results or i == num_rollout_timesteps - 2:
                    # decode dynamic to data
                    if self.force_decoder_fp32:
                        with torch.autocast(device_type=str(device).split(":")[0], enabled=False):
                            subnode_feats_hat = self.decoder(
                                dynamics.float(),
                                subnode_mask=subnode_mask,
                                batch_idx=batch_idx,
                                condition=condition.float(),
                            )
                    else:
                        subnode_feats_hat = self.decoder(
                            dynamics,
                            subnode_mask=subnode_mask,
                            batch_idx=batch_idx,
                            condition=condition,
                        )
                    
                    num_cells = slot2cell.max().item() + 1
                    x_hat = scatter_slots_to_cells(subnode_feats_hat, slot2cell, num_cells)
                    if clip is not None:
                        x_hat = x_hat.clip(-clip, clip)
                    x_hats.append(x_hat)
        
        elif mode == "image":
            assert intermediate_results
            # initial forward pass
            outputs = self(
                subnode_feats=subnode_feats,
                subnode_mask=subnode_mask,
                slot2cell=slot2cell,
                timestep=timestep,
                velocity=velocity,
                batch_idx=batch_idx,
                unbatch_idx=None,
                unbatch_select=None,
            )
            x_hat = outputs["x_hat"]
            x_hats.append(x_hat)
            
            for _ in range(num_rollout_timesteps - 1):
                # For slot model, we need to re-assign cells to slots for next timestep
                # This is complex - for now, we'll use a simplified approach
                # TODO: Implement proper autoregressive rollout with slot reassignment
                # For now, shift last prediction into history (simplified)
                # In practice, you'd need to re-run slot assignment on the new positions
                timestep = timestep + 1
                outputs = self(
                    subnode_feats=subnode_feats,  # Simplified - should re-assign
                    subnode_mask=subnode_mask,
                    slot2cell=slot2cell,
                    timestep=timestep,
                    velocity=velocity,
                    batch_idx=batch_idx,
                    unbatch_idx=None,
                    unbatch_select=None,
                )
                x_hat = outputs["x_hat"]
                if clip is not None:
                    x_hat = x_hat.clip(-clip, clip)
                x_hats.append(x_hat)
        else:
            raise NotImplementedError(f"Unknown rollout mode: {mode}")
        
        if not intermediate_results:
            assert len(x_hats) == 1
        
        # Stack predictions: [num_rollout_timesteps, B*K, C] or [num_rollout_timesteps, B*K, T*C]
        return torch.stack(x_hats, dim=0)

