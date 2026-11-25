"""
Step 4 - property-conditioned generation: target properties -> z-scored p -> G(z, p) -> graphs -> CIFs.

Targets are given in the units of Table 1 (HVF, APV cm3/g, SSA m2/g, Mass g/mol, Density kg/m3,
CO2 wt%). Unspecified properties are taken from a reference: the mean of the top-k ZTCs by
`--rank-by` (default CO2_298K), so you can ask "like the best performers, but with more CO2 uptake".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .decode import tensors_to_graph, write_structure_cif
from .evaluate import reference_from_dataset, summarize, wl_hash
from .featurize import load_dataset
from .train import load_generator


def build_target_vector(meta, props_csv, targets: dict, top_k=5, rank_by="CO2_298K"):
    cols = meta["property_columns"]
    df = pd.read_csv(props_csv)
    ref = df.sort_values(rank_by, ascending=False).head(top_k)[cols].mean() if rank_by in df else df[cols].mean()
    raw = np.array([float(targets.get(c, ref[c])) for c in cols])
    z = (raw - np.array(meta["scaler_mean"])) / np.array(meta["scaler_scale"])
    lo, hi = df[cols].min().to_numpy(), df[cols].max().to_numpy()
    warn = [f"{c}={v:g} outside training range [{l:g}, {h:g}] (z={zz:+.2f}) - extrapolation"
            for c, v, l, h, zz in zip(cols, raw, lo, hi, z) if v < l or v > h]
    return raw, z.astype(np.float32), warn


@torch.no_grad()
def generate(generator_path, data_dir, props_csv, out_dir, n=100, targets=None, top_k=5, rank_by="CO2_298K",
             threshold=0.5, z_temperature=1.0, write_cifs=True, max_cifs=50, device="auto", seed=0):
    torch.manual_seed(seed)
    out = Path(out_dir); (out / "cif").mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(data_dir) / "scaler.json").read_text())
    G, cfg, dev = load_generator(generator_path, device)
    raw, z_p, warns = build_target_vector(meta, props_csv, targets or {}, top_k, rank_by)
    for w in warns:
        print("WARNING:", w)
    p = torch.from_numpy(np.tile(z_p, (n, 1))).to(dev)
    z = torch.randn(n, cfg["z_dim"], device=dev) * z_temperature
    X, A = G(z, p)
    X, A = X.cpu().numpy(), A.cpu().numpy()
    np.savez_compressed(out / "generated_tensors.npz", X=X, A=A.astype(np.float16), p_raw=raw, p_scaled=z_p)

    graphs = [tensors_to_graph(X[i], A[i], threshold=threshold) for i in range(n)]
    d = load_dataset(Path(data_dir) / "graph_dataset.npz")
    real = reference_from_dataset(d)
    train_hashes = set(wl_hash(g) for g in real)
    s_gen, hashes, stats = summarize(graphs, train_hashes, "generated")
    s_real, _, _ = summarize(real, None, "training")
    rows, seen = [], set()
    dens_col = meta["property_columns"].index("Density") if "Density" in meta["property_columns"] else None
    n_written = 0
    for i, (g, h, st) in enumerate(zip(graphs, hashes, stats)):
        row = dict(id=f"gen_{i:04d}", n_atoms=st["n_atoms"], n_bonds=st["n_bonds"], valid=st["valid"], ztc_like=st["ztc_like"],
                   novel=h not in train_hashes, duplicate=h in seen, frac_deg3=round(st["frac_deg3"], 3),
                   wl_hash=h)
        seen.add(h)
        if write_cifs and st["valid"] and not row["duplicate"] and n_written < max_cifs:
            tag = "novel" if row["novel"] else "known"          # 'known' = reproduces a training graph
            fname = f"{row['id']}_{tag}.cif"
            info = write_structure_cif(out / "cif" / fname, row["id"], g,
                                       target_density=raw[dens_col] if dens_col is not None else None,
                                       seed=i, comment="target " + ", ".join(
                                           f"{c}={v:g}" for c, v in zip(meta["property_columns"], raw)))
            row.update(cif=f"cif/{fname}", cell_a=round(info["cell_a"], 3),
                       min_nonbonded_A=round(info["min_nonbonded"], 3))
            n_written += 1
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "generated_summary.csv", index=False)
    report = dict(target_properties=dict(zip(meta["property_columns"], raw.tolist())), warnings=warns,
                  generated=s_gen, training_reference=s_real, cifs_written=n_written)
    (out / "generation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report
