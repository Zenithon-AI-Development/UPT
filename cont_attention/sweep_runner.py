# sweep_runner.py
import argparse, os, subprocess, tempfile, textwrap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encdec_width", type=int, required=True)
    ap.add_argument("--latent_dim", type=int, required=True)
    ap.add_argument("--num_attn_heads", type=int, required=True)
    ap.add_argument("--num_latent_tokens", type=int, required=True)
    args = ap.parse_args()

    # Sanity: multi-head attention expects dims divisible by heads
    assert args.encdec_width % args.num_attn_heads == 0, "encdec_width must be divisible by num_attn_heads"
    assert args.latent_dim   % args.num_attn_heads == 0, "latent_dim must be divisible by num_attn_heads"

    overlay = textwrap.dedent(f"""
    template: ${{yaml:yamls/trl2d/trl2d.yaml}}

    # override model sizes that the base YAML normally fills via ${{
    # select:...}} presets — we replace the entire kwargs dicts.
    template.model.encoder.num_latent_tokens: {args.num_latent_tokens}

    template.model.encoder.kwargs:
      gnn_dim: {args.encdec_width}
      enc_dim: {args.encdec_width}
      perc_dim: {args.latent_dim}
      enc_num_attn_heads: {args.num_attn_heads}
      perc_num_attn_heads: {args.num_attn_heads}

    template.model.latent.kwargs:
      dim: {args.latent_dim}
      num_attn_heads: {args.num_attn_heads}

    template.model.decoder.kwargs:
      dim: {args.latent_dim}
      perc_dim: {args.encdec_width}
      num_attn_heads: {args.num_attn_heads}
      perc_num_attn_heads: {args.num_attn_heads}
    """)

    # Write overlay to a temp file
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(overlay)
        hp_path = f.name

    # Give W&B a bit more time to initialize; recommended in W&B support threads.
    env = os.environ.copy()
    env.setdefault("WANDB_INIT_TIMEOUT", "180")   # seconds
    env.setdefault("WANDB_HTTP_TIMEOUT", "60")    # seconds

    # Launch training with a 2h walltime per run (as you had)
    cmd = ["python", "main_train.py", "--hp", hp_path, "--accelerator", "gpu", "--devices", "0"]
    subprocess.run(cmd, check=True, timeout=7200, env=env)

if __name__ == "__main__":
    main()
