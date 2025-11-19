#!/usr/bin/env python
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets.well_trl2d_dataset import WellTrl2dDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(in_dim: int, out_dim: int, hidden_dim: int, depth: int) -> nn.Module:
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    dataset = WellTrl2dDataset(
        split="train",
        num_input_timesteps=args.num_input,
        num_output_timesteps=args.num_output,
        norm="mean0std1_auto",
        clamp=0,
        clamp_mode="log",
        max_num_sequences=1,
    )
    x = dataset.getitem_x(0)
    y = dataset.getitem_target(0)

    if args.max_points > 0 and args.max_points < x.shape[0]:
        perm = torch.randperm(x.shape[0])[: args.max_points]
        x = x[perm]
        y = y[perm]

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = build_model(x.shape[-1], y.shape[-1], args.hidden_dim, args.depth).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Training tiny MLP on {len(dataset)} points (in_dim={x.shape[-1]}, out_dim={y.shape[-1]})")
    for epoch in range(args.epochs):
        running_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(dataset)
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:04d}] loss={epoch_loss:.6f}")

    with torch.no_grad():
        final_pred = model(x.to(device))
        final_loss = criterion(final_pred, y.to(device)).item()
    print(f"Final MSE: {final_loss:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny MLP baseline for TRL2D overfit sanity check.")
    parser.add_argument("--num-input", type=float, default=1)
    parser.add_argument("--num-output", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())



