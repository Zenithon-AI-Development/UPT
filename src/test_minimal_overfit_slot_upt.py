#!/usr/bin/env python3
"""
Minimal overfitting test for the slot-based UPT model on a single CFD sample.
Logs metrics to Weights & Biases.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from configs.static_config import StaticConfig
from providers.dataset_config_provider import DatasetConfigProvider
from providers.path_provider import PathProvider
from datasets import dataset_from_kwargs
from models import model_from_kwargs

from slot_upt.slot_assignment import assign_cells_to_slots_voxel_grid


class MiniDC:
    """Minimal data container wrapper for the composite model."""

    def __init__(self, dataset):
        self._dataset = dataset

    def get_dataset(self, *_, **__):
        return self._dataset


def build_slot_inputs(mesh_pos, features, *, M, N, num_timesteps, device):
    """Create slot representation from raw mesh positions and features for B=1."""
    num_cells = mesh_pos.size(0)
    batch_idx = torch.zeros(num_cells, dtype=torch.long, device=mesh_pos.device)

    subnode_feats, subnode_mask, slot2cell, _ = assign_cells_to_slots_voxel_grid(
        mesh_pos=mesh_pos,
        features=features,
        M=M,
        N=N,
        batch_idx=batch_idx,
        num_timesteps=num_timesteps,
        ndim=mesh_pos.size(-1),
    )

    # Move to target device
    subnode_feats = subnode_feats.to(device)
    subnode_mask = subnode_mask.to(device)
    slot2cell = slot2cell.to(device)

    # batch_idx for encoder (not used internally but kept for API parity)
    B, T = subnode_feats.shape[:2]
    batch_idx_supernodes = torch.arange(B * T, device=device).repeat_interleave(M)

    return subnode_feats, subnode_mask, slot2cell, batch_idx_supernodes


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    static = StaticConfig(uri=str(REPO_ROOT / "static_config.yaml"))
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static.get_global_dataset_paths(),
        local_dataset_path=static.get_local_dataset_path(),
        data_source_modes=static.get_data_source_modes(),
    )
    path_provider = PathProvider(
        output_path=static.output_path,
        model_path=static.model_path,
        stage_name="slot_test",
        stage_id="slot_test",
        temp_path=static.temp_path,
    )

    # Hyperparameters (aligned with baseline minimal test)
    config = dict(
        dataset_kind="well_trl2d_dataset",
        num_input_timesteps=4,
        M=4096,
        N=256,
        lr=0.01,
        steps=1000,
        encoder_dim=32,
        latent_dim=64,
        decoder_dim=64,
        num_latent_tokens=32,
    )

    wandb.init(project="slot-upt-overfit", name="slot_minimal_overfit", config=config)

    dataset = dataset_from_kwargs(
        kind=config["dataset_kind"],
        split="train",
        well_base_path="/home/workspace/projects/data/datasets_david/datasets/",
        num_input_timesteps=config["num_input_timesteps"],
        norm="mean0std1",
        clamp=0,
        clamp_mode="log",
        max_num_timesteps=101,
        max_num_sequences=1,
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
    )

    assert len(dataset) == 1, f"Expected 1 sample, got {len(dataset)}"

    # Fetch sample (B=1)
    x = dataset.getitem_x(0).to(device)
    target = dataset.getitem_target(0).to(device)
    mesh_pos = dataset.getitem_mesh_pos(0).to(device)
    query_pos = dataset.getitem_query_pos(0).to(device)
    timestep = torch.as_tensor(dataset.getitem_timestep(0), dtype=torch.long, device=device).view(1)
    velocity = torch.as_tensor(dataset.getitem_velocity(0), dtype=torch.float32, device=device).view(1)

    channels_per_timestep = x.shape[1] // config["num_input_timesteps"]
    features = x.view(-1, x.shape[1])  # [K, num_input_timesteps * C]

    subnode_feats, subnode_mask, slot2cell, batch_idx_supernodes = build_slot_inputs(
        mesh_pos=mesh_pos,
        features=features,
        M=config["M"],
        N=config["N"],
        num_timesteps=config["num_input_timesteps"],
        device=device,
    )

    model = model_from_kwargs(
        kind="cfd_slot_simformer_model",
        conditioner=dict(
            kind="conditioners.timestep_velocity_conditioner_pdearena",
            kwargs=dict(dim=config["decoder_dim"] // 2),
        ),
        encoder=dict(
            kind="cfd_slot_pool_transformer_perceiver",
            num_latent_tokens=config["num_latent_tokens"],
            enc_depth=1,
            kwargs=dict(
                gnn_dim=config["encoder_dim"],
                enc_dim=config["encoder_dim"],
                perc_dim=config["latent_dim"],
                enc_num_attn_heads=2,
                perc_num_attn_heads=2,
                M=config["M"],
                N=config["N"],
            ),
        ),
        latent=dict(
            kind="latent.transformer_model",
            depth=1,
            kwargs=dict(dim=config["latent_dim"], num_attn_heads=2),
        ),
        decoder=dict(
            kind="cfd_slot_transformer_perceiver",
            depth=1,
            use_last_norm=False,
            clamp=0,
            clamp_mode="log",
            kwargs=dict(
                dim=config["decoder_dim"],
                perc_dim=config["latent_dim"],
                num_attn_heads=2,
                perc_num_attn_heads=2,
                M=config["M"],
                N=config["N"],
                ndim=mesh_pos.shape[-1],
            ),
        ),
        input_shape=dataset.getshape_x(),
        output_shape=dataset.getshape_target(),
        update_counter=None,
        path_provider=path_provider,
        data_container=MiniDC(dataset),
        M=config["M"],
        N=config["N"],
        num_input_timesteps=config["num_input_timesteps"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    def forward_model():
        outputs = model(
            subnode_feats=subnode_feats,
            subnode_mask=subnode_mask,
            slot2cell=slot2cell,
            timestep=timestep,
            velocity=velocity,
            batch_idx=batch_idx_supernodes,
            unbatch_idx=None,
            unbatch_select=None,
        )
        return outputs["x_hat"]

    model.train()
    for step in range(config["steps"]):
        optimizer.zero_grad()
        pred = forward_model()

        loss = F.mse_loss(pred, target)
        rel_l1 = (pred - target).abs().sum() / (target.abs().sum() + 1e-12)
        rel_l2 = torch.linalg.vector_norm(pred - target) / (torch.linalg.vector_norm(target) + 1e-12)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if step in [0, 1, 5, 10, 50, 100, 500, config["steps"] - 1]:
            print(
                f"Step {step:4d}: Loss={loss.item():.8f}, "
                f"Rel L1={rel_l1.item():.8f} ({rel_l1.item()*100:.4f}%), "
                f"Rel L2={rel_l2.item():.8f} ({rel_l2.item()*100:.4f}%)"
            )

        wandb.log(
            {
                "loss": loss.item(),
                "rel_l1": rel_l1.item(),
                "rel_l2": rel_l2.item(),
            },
            step=step,
        )

    print("=" * 70)
    print("FINAL RESULTS (Slot-UPT)")
    print("=" * 70)
    print(f"Final Loss: {loss.item():.8f}")
    print(f"Final Rel L1: {rel_l1.item():.8f} ({rel_l1.item()*100:.4f}%)")
    print(f"Final Rel L2: {rel_l2.item():.8f} ({rel_l2.item()*100:.4f}%)")
    wandb.log(
        {
            "final_loss": loss.item(),
            "final_rel_l1": rel_l1.item(),
            "final_rel_l1_pct": rel_l1.item() * 100.0,
            "final_rel_l2": rel_l2.item(),
            "final_rel_l2_pct": rel_l2.item() * 100.0,
        },
        step=config["steps"],
    )

    wandb.finish()


if __name__ == "__main__":
    main()
