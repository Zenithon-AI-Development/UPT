"""
Clean UPT model that composes encoder, latent_core, decoder, and conditioner.

This is the main model interface for Stage 0 baseline.
"""

import torch
from torch import nn
from typing import Optional, Dict, List, Any

from .encoder import BaseEncoder
from .latent_core import BaseLatentCore
from .decoder import BaseDecoder
from .conditioner import BaseConditioner


class UPTModel(nn.Module):
    """
    Universal Physics Transformer model.
    
    Clean modular composition:
    1. Encoder: data → [B, M, d]
    2. Latent Core: [B, M, d] → [B, M, d] (temporal propagation)
    3. Decoder: [B, M, d] → outputs
    
    Supports:
    - Multiple encoders/decoders (via config)
    - Extra per-token conditioning (hooks for grid-structure embeddings)
    - Standardized conditioning interface
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        latent_core: BaseLatentCore,
        decoder: BaseDecoder,
        conditioner: Optional[BaseConditioner] = None,
        use_multiple_encoders: bool = False,
        use_multiple_decoders: bool = False,
        encoders: Optional[List[BaseEncoder]] = None,
        decoders: Optional[List[BaseDecoder]] = None,
        **kwargs
    ):
        """
        Args:
            encoder: Main encoder (or first encoder if multiple)
            latent_core: Latent core transformer
            decoder: Main decoder (or first decoder if multiple)
            conditioner: Optional conditioner for timestep/params
            use_multiple_encoders: If True, use encoders list
            use_multiple_decoders: If True, use decoders list
            encoders: List of encoders (if use_multiple_encoders=True)
            decoders: List of decoders (if use_multiple_decoders=True)
        """
        super().__init__()
        
        # Main components
        self.conditioner = conditioner
        self.latent_core = latent_core
        
        # Single or multiple encoders/decoders
        self.use_multiple_encoders = use_multiple_encoders
        self.use_multiple_decoders = use_multiple_decoders
        
        if use_multiple_encoders:
            assert encoders is not None and len(encoders) > 0
            self.encoders = nn.ModuleList(encoders)
            self.encoder = self.encoders[0]  # Main encoder for compatibility
        else:
            self.encoder = encoder
            self.encoders = None
        
        if use_multiple_decoders:
            assert decoders is not None and len(decoders) > 0
            self.decoders = nn.ModuleList(decoders)
            self.decoder = self.decoders[0]  # Main decoder for compatibility
        else:
            self.decoder = decoder
            self.decoders = None
        
        # Hook for extra per-token conditioning (e.g., grid-structure embeddings)
        self._extra_token_conditioning_hook = None
    
    def register_extra_token_conditioning_hook(self, hook_fn):
        """
        Register a hook function for extra per-token conditioning.
        
        The hook function should take (latent, condition, **kwargs) and return
        [B, M, cond_dim] per-token conditioning vectors.
        
        Example:
            def grid_structure_hook(latent, condition, grid_pos, **kwargs):
                # Compute grid-structure embeddings
                return grid_embeddings  # [B, M, cond_dim]
            
            model.register_extra_token_conditioning_hook(grid_structure_hook)
        """
        self._extra_token_conditioning_hook = hook_fn
    
    def forward(
        self,
        x: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, torch.Tensor]] = None,
        extra_token_conditioning: Optional[torch.Tensor] = None,
        encoder_idx: Optional[int] = None,
        decoder_idx: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input data (format depends on encoder)
            query_pos: Query positions for decoder
            timestep: [B] timestep indices
            velocity: [B] or [B, ...] velocity or other features
            params: Dict of additional parameters
            extra_token_conditioning: [B, M, cond_dim] per-token conditioning (optional)
            encoder_idx: Which encoder to use (if multiple)
            decoder_idx: Which decoder to use (if multiple)
            **kwargs: Additional arguments passed to encoder/decoder
        
        Returns:
            outputs: Dict with:
                - 'latent': [B, M, d] latent representation
                - 'output': Decoded output
                - 'condition': [B, cond_dim] global conditioning (if used)
        """
        outputs = {}
        
        # Get conditioning
        condition = None
        if self.conditioner is not None:
            condition = self.conditioner(
                timestep=timestep,
                velocity=velocity,
                params=params,
                **kwargs
            )
            outputs['condition'] = condition
        
        # Select encoder
        encoder = self.encoder
        if self.use_multiple_encoders:
            if encoder_idx is None:
                encoder_idx = 0
            encoder = self.encoders[encoder_idx]
        
        # Encode: data → [B, M, d]
        latent = encoder(
            x=x,
            condition=condition,
            extra_token_conditioning=extra_token_conditioning,
            **kwargs
        )
        outputs['latent'] = latent
        
        # Get extra per-token conditioning from hook if provided
        if self._extra_token_conditioning_hook is not None:
            hook_conditioning = self._extra_token_conditioning_hook(
                latent=latent,
                condition=condition,
                **kwargs
            )
            if extra_token_conditioning is None:
                extra_token_conditioning = hook_conditioning
            else:
                extra_token_conditioning = extra_token_conditioning + hook_conditioning
        
        # Latent core: [B, M, d] → [B, M, d]
        latent = self.latent_core(
            x=latent,
            condition=condition,
            extra_token_conditioning=extra_token_conditioning,
            **kwargs
        )
        outputs['latent_after_core'] = latent
        
        # Select decoder
        decoder = self.decoder
        if self.use_multiple_decoders:
            if decoder_idx is None:
                decoder_idx = 0
            decoder = self.decoders[decoder_idx]
        
        # Decode: [B, M, d] → outputs
        output = decoder(
            latent=latent,
            query_pos=query_pos,
            condition=condition,
            extra_token_conditioning=extra_token_conditioning,
            **kwargs
        )
        outputs['output'] = output
        
        return outputs
    
    @torch.no_grad()
    def rollout(
        self,
        x: torch.Tensor,
        query_pos: torch.Tensor,
        num_steps: int,
        timestep: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "autoregressive",
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Autoregressive rollout for multiple timesteps.
        
        Args:
            x: Initial input data
            query_pos: Query positions for decoder
            num_steps: Number of rollout steps
            timestep: Initial timestep [B] (default: 0)
            velocity: Velocity or other features
            params: Dict of additional parameters
            mode: "autoregressive" (use predictions as next input) or "latent" (rollout in latent space)
            **kwargs: Additional arguments
        
        Returns:
            predictions: List of predictions for each timestep
        """
        self.eval()
        
        if timestep is None:
            timestep = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        
        predictions = []
        current_x = x
        
        for step in range(num_steps):
            # Forward pass
            outputs = self(
                x=current_x,
                query_pos=query_pos,
                timestep=timestep,
                velocity=velocity,
                params=params,
                **kwargs
            )
            
            pred = outputs['output']
            predictions.append(pred)
            
            if mode == "autoregressive":
                # Use prediction as next input (shift history)
                # This depends on input format - may need to be customized
                current_x = pred  # Simplified - may need to concatenate with history
            elif mode == "latent":
                # Rollout in latent space (use latent as next input)
                current_x = outputs['latent_after_core']
            else:
                raise ValueError(f"Unknown rollout mode: {mode}")
            
            # Increment timestep
            timestep = timestep + 1
        
        return predictions

