"""
Step 2 - train the conditional WGAN-GP (default) or conditional VAE on the graph tensors.

WGAN-GP objective (Gulrajani et al., 2017), conditional on properties p:
    L_D = E_fake[D(G(z,p),p)] - E_real[D(x,p)] + lambda * E_xhat[(||grad D(xhat,p)||_2 - 1)^2]
    L_G = -E_fake[D(G(z,p),p)]
The critic is updated n_critic times per generator update; Adam(lr=1e-4, betas=(0.0, 0.9)).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .featurize import load_dataset
from .models import CVAE, Generator, build_critic, cvae_loss, gradient_penalty


def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(data_dir, batch_size, prop_noise=0.0):
    d = load_dataset(Path(data_dir) / "graph_dataset.npz")
    ds = TensorDataset(torch.from_numpy(d["X"]), torch.from_numpy(d["A"]), torch.from_numpy(d["P_scaled"]))
    meta = json.loads((Path(data_dir) / "scaler.json").read_text())
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True), d, meta


def train_wgan_gp(data_dir, out_dir, epochs=300, batch_size=32, z_dim=128, n_critic=5, gp_lambda=10.0,
                  lr_g=1e-4, lr_d=1e-4, betas=(0.0, 0.9), critic="mlp", edge_head="dense",
                  prop_noise=0.05, device="auto", seed=0, save_every=25, log_every=1, max_steps=None):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = get_device(device)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    loader, d, meta = _loader(data_dir, batch_size)
    N, F = d["X"].shape[1:]
    R, P = d["A"].shape[-1], d["P_scaled"].shape[-1]

    G = Generator(N, F, R, P, z_dim=z_dim, edge_head=edge_head).to(dev)
    D = build_critic(critic, N, F, R, P).to(dev)
    optG = torch.optim.Adam(G.parameters(), lr=lr_g, betas=betas)
    optD = torch.optim.Adam(D.parameters(), lr=lr_d, betas=betas)
    cfg = dict(model="wgan_gp", nmax=N, n_feat=F, n_bond=R, n_prop=P, z_dim=z_dim, critic=critic,
               edge_head=edge_head, n_critic=n_critic, gp_lambda=gp_lambda, lr_g=lr_g, lr_d=lr_d,
               betas=list(betas), batch_size=batch_size, epochs=epochs, prop_noise=prop_noise,
               property_columns=meta["property_columns"], n_train_graphs=int(len(d["X"])))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    n_g = sum(p.numel() for p in G.parameters()); n_d = sum(p.numel() for p in D.parameters())
    print(f"device={dev}  graphs={len(d['X'])}  G params={n_g/1e6:.1f}M  D params={n_d/1e6:.1f}M")

    log_f = open(out / "training_log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "step", "loss_D", "loss_G", "wasserstein", "gp", "grad_norm", "edge_density_fake",
                  "edge_density_real", "sec"])
    step, t0 = 0, time.time()
    for ep in range(1, epochs + 1):
        acc = []
        for k, (X, A, p) in enumerate(loader):
            X, A, p = X.to(dev), A.to(dev), p.to(dev)
            if prop_noise:
                p = p + prop_noise * torch.randn_like(p)          # smooths the conditional manifold
            # ---- critic
            z = torch.randn(X.shape[0], z_dim, device=dev)
            with torch.no_grad():
                Xf, Af = G(z, p)
            d_real, d_fake = D(X, A, p).mean(), D(Xf, Af, p).mean()
            gp, gnorm = gradient_penalty(D, X, A, Xf, Af, p, gp_lambda)
            loss_D = d_fake - d_real + gp
            optD.zero_grad(set_to_none=True); loss_D.backward(); optD.step()
            # ---- generator every n_critic critic steps
            if k % n_critic == 0:
                z = torch.randn(X.shape[0], z_dim, device=dev)
                Xf, Af = G(z, p)
                loss_G = -D(Xf, Af, p).mean()
                optG.zero_grad(set_to_none=True); loss_G.backward(); optG.step()
                acc.append([float(loss_D), float(loss_G), float(d_real - d_fake), float(gp), float(gnorm),
                            float((Af > 0.5).float().mean()), float(A.mean())])
            step += 1
            if max_steps and step >= max_steps:
                break
        if acc and ep % log_every == 0:
            m = np.mean(acc, 0)
            log.writerow([ep, step, *[f"{v:.5f}" for v in m], f"{time.time()-t0:.1f}"]); log_f.flush()
            print(f"ep {ep:4d} | L_D {m[0]:+.3f} | L_G {m[1]:+.3f} | W {m[2]:.3f} | GP {m[3]:.3f} "
                  f"| |g| {m[4]:.2f} | edges fake/real {m[5]:.4f}/{m[6]:.4f}")
        if ep % save_every == 0 or ep == epochs or (max_steps and step >= max_steps):
            torch.save(dict(G=G.state_dict(), D=D.state_dict(), optG=optG.state_dict(), optD=optD.state_dict(),
                            cfg=cfg, epoch=ep), out / "checkpoint.pt")
            torch.save(dict(G=G.state_dict(), cfg=cfg), out / "generator.pt")
        if max_steps and step >= max_steps:
            break
    log_f.close()
    _plot_log(out / "training_log.csv", out / "training_curves.png")
    return out / "generator.pt"


def train_cvae(data_dir, out_dir, epochs=300, batch_size=32, z_dim=128, lr=1e-3, beta_max=1.0,
               warmup_epochs=50, edge_head="pairwise", device="auto", seed=0, save_every=25, max_steps=None):
    torch.manual_seed(seed)
    dev = get_device(device)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    loader, d, meta = _loader(data_dir, batch_size)
    N, F = d["X"].shape[1:]; R, P = d["A"].shape[-1], d["P_scaled"].shape[-1]
    model = CVAE(N, F, R, P, z_dim=z_dim, edge_head=edge_head).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    cfg = dict(model="cvae", nmax=N, n_feat=F, n_bond=R, n_prop=P, z_dim=z_dim, edge_head=edge_head,
               epochs=epochs, property_columns=meta["property_columns"])
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log_f = open(out / "training_log.csv", "w", newline=""); log = csv.writer(log_f)
    log.writerow(["epoch", "loss", "rec_x", "rec_a", "kl", "beta"])
    step = 0
    for ep in range(1, epochs + 1):
        beta = beta_max * min(1.0, ep / max(warmup_epochs, 1))     # KL annealing against posterior collapse
        acc = []
        for X, A, p in loader:
            X, A, p = X.to(dev), A.to(dev), p.to(dev)
            _, _, xl, el, mu, lv = model(X, A, p)
            loss, parts = cvae_loss(X, A, xl, el, mu, lv, beta=beta)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            acc.append([float(loss), parts["rec_x"], parts["rec_a"], parts["kl"]])
            step += 1
            if max_steps and step >= max_steps:
                break
        m = np.mean(acc, 0)
        log.writerow([ep, *[f"{v:.5f}" for v in m], f"{beta:.3f}"]); log_f.flush()
        print(f"ep {ep:4d} | loss {m[0]:.4f} | rec_x {m[1]:.4f} | rec_a {m[2]:.4f} | KL {m[3]:.3f} | beta {beta:.2f}")
        if ep % save_every == 0 or ep == epochs or (max_steps and step >= max_steps):
            torch.save(dict(G=model.decoder.state_dict(), full=model.state_dict(), cfg=cfg), out / "generator.pt")
        if max_steps and step >= max_steps:
            break
    log_f.close()
    return out / "generator.pt"


def load_generator(path, device="auto"):
    dev = get_device(device)
    try:
        ck = torch.load(path, map_location=dev, weights_only=False)   # torch >= 1.13
    except TypeError:
        ck = torch.load(path, map_location=dev)
    c = ck["cfg"]
    G = Generator(c["nmax"], c["n_feat"], c["n_bond"], c["n_prop"], z_dim=c["z_dim"], edge_head=c["edge_head"])
    G.load_state_dict(ck["G"]); G.to(dev).eval()
    return G, c, dev


def _plot_log(csv_path, png_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.read_csv(csv_path)
        if df.empty:
            return
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].plot(df.epoch, df.loss_D, label="critic"); ax[0].plot(df.epoch, df.loss_G, label="generator")
        ax[0].set_title("losses"); ax[0].legend()
        ax[1].plot(df.epoch, df.wasserstein); ax[1].set_title("Wasserstein estimate  E[D(real)] - E[D(fake)]")
        ax[2].plot(df.epoch, df.edge_density_fake, label="generated"); ax[2].plot(df.epoch, df.edge_density_real, label="real")
        ax[2].set_title("bond density"); ax[2].legend()
        for a in ax:
            a.set_xlabel("epoch")
        fig.tight_layout(); fig.savefig(png_path, dpi=150); plt.close(fig)
    except Exception as e:      # plotting must never break training
        print("plot skipped:", e)
