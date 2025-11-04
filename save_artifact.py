import os, wandb, yaml, torch

# fill these in from your run
ENTITY      = "Zenithon-AI"
PROJECT     = "transolver-trl2d-sweep"    # or your sweep project
ARTIFACT    = "model-best:latest"        # example, check the run's Artifacts tab

api = wandb.Api()
artifact = api.artifact(f"{ENTITY}/{PROJECT}/{ARTIFACT}", type="model")
root = artifact.download()               # local dir with files

# inspect files
print("Downloaded to:", root)
print("Files:", os.listdir(root))

# find checkpoint + hp
ckpt_path = [f for f in os.listdir(root) if f.endswith((".pt", ".pth", ".ckpt"))][0]
hp_path   = "hp_resolved.yaml" if "hp_resolved.yaml" in os.listdir(root) else None
print("ckpt:", ckpt_path, "hp:", hp_path)

