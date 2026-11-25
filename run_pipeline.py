#!/usr/bin/env python
"""
End-to-end ZTC generative pipeline.

  python run_pipeline.py featurize          # CIFs + Table 1 -> data/processed/graph_dataset.npz (+ scaler)
  python run_pipeline.py train              # conditional WGAN-GP  (add --model cvae for the VAE)
  python run_pipeline.py generate --target CO2_298K=48 HVF=0.93 --n 200
  python run_pipeline.py smoke-test         # 30-second check that PyTorch + the whole chain run
  python run_pipeline.py all                # featurize -> train -> generate with defaults

Paths default to the folder layout this was delivered in:
  ../ZTC structures relaxed/*.cif   and   ../Table 1 ... .docx
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_CIF_DIR = HERE.parent / "ZTC structures relaxed"
DEFAULT_TABLE = HERE.parent / "Table 1. Calculated loading capacity versus structural features for the 68 ZTCs.docx"
DEFAULT_PROPS = HERE / "data" / "ztc_properties.csv"
DEFAULT_DATA = HERE / "data" / "processed"
DEFAULT_RUN = HERE / "runs" / "wgan_gp"
DEFAULT_PROPERTIES = ["HVF", "APV", "SSA", "Mass", "Density", "CO2_298K"]


def cmd_featurize(a):
    from ztc_gen.featurize import build_dataset
    from ztc_gen.properties import sanity_flags, table_docx_to_csv
    import pandas as pd
    if Path(a.table).exists():
        df = table_docx_to_csv(a.table, a.props)
        print(f"Table 1 -> {a.props} ({len(df)} rows)")
    else:
        df = pd.read_csv(a.props)
        print(f"Using existing {a.props}")
    for f in sanity_flags(df):
        print("  check Table 1:", f)
    build_dataset(a.cif_dir, a.props, a.data, nmax=a.nmax, property_columns=a.properties,
                  fragments_per_100_atoms=a.frag_density, seed=a.seed)


def cmd_train(a):
    from ztc_gen.train import train_cvae, train_wgan_gp
    if a.model == "wgan_gp":
        train_wgan_gp(a.data, a.run, epochs=a.epochs, batch_size=a.batch_size, n_critic=a.n_critic,
                      gp_lambda=a.gp_lambda, lr_g=a.lr, lr_d=a.lr, critic=a.critic, edge_head=a.edge_head,
                      device=a.device, seed=a.seed, max_steps=a.max_steps, save_every=a.save_every)
    else:
        train_cvae(a.data, a.run, epochs=a.epochs, batch_size=a.batch_size, edge_head=a.edge_head,
                   device=a.device, seed=a.seed, max_steps=a.max_steps, save_every=a.save_every)


def _parse_targets(items):
    t = {}
    for it in items or []:
        k, v = it.split("=")
        t[k.strip()] = float(v)
    return t


def cmd_generate(a):
    from ztc_gen.generate import generate
    generate(Path(a.run) / "generator.pt", a.data, a.props, a.out or Path(a.run) / "generated", n=a.n,
             targets=_parse_targets(a.target), top_k=a.top_k, rank_by=a.rank_by, threshold=a.threshold,
             z_temperature=a.temperature, max_cifs=a.max_cifs, device=a.device, seed=a.seed)


def cmd_smoke(a):
    a.run = str(HERE / "runs" / "smoke_test")
    if not (Path(a.data) / "graph_dataset.npz").exists():
        cmd_featurize(a)
    a.epochs, a.max_steps, a.save_every = 1, 12, 1
    for model, edge, critic in [("wgan_gp", "pairwise", "gcn"), ("wgan_gp", "dense", "mlp"), ("cvae", "pairwise", "gcn")]:
        print(f"\n=== smoke test: {model} edge_head={edge} critic={critic}")
        a.model, a.edge_head, a.critic = model, edge, critic
        a.run = str(HERE / "runs" / f"smoke_{model}_{edge}")
        cmd_train(a)
        a.n, a.max_cifs, a.out = 8, 3, None
        cmd_generate(a)
    print("\nSmoke test passed - the pipeline runs end to end. Outputs are meaningless after 12 steps.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["featurize", "train", "generate", "smoke-test", "all"])
    ap.add_argument("--cif-dir", default=str(DEFAULT_CIF_DIR))
    ap.add_argument("--table", default=str(DEFAULT_TABLE))
    ap.add_argument("--props", default=str(DEFAULT_PROPS))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--properties", nargs="+", default=DEFAULT_PROPERTIES,
                    help="conditioning columns from Table 1 (also: CO2_173K CO2_673K)")
    ap.add_argument("--nmax", type=int, default=150)
    ap.add_argument("--frag-density", type=float, default=6.0, help="BFS fragments per 100 atoms for N>Nmax")
    # training
    ap.add_argument("--model", choices=["wgan_gp", "cvae"], default="wgan_gp")
    ap.add_argument("--critic", choices=["mlp", "gcn"], default="mlp")
    ap.add_argument("--edge-head", choices=["dense", "pairwise"], default="dense")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--gp-lambda", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=None)
    # generation
    ap.add_argument("--target", nargs="*", help="e.g. CO2_298K=48 HVF=0.93 SSA=3000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rank-by", default="CO2_298K")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-cifs", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.command == "featurize":
        cmd_featurize(a)
    elif a.command == "train":
        cmd_train(a)
    elif a.command == "generate":
        cmd_generate(a)
    elif a.command == "smoke-test":
        cmd_smoke(a)
    else:
        cmd_featurize(a); cmd_train(a); cmd_generate(a)


if __name__ == "__main__":
    main()
