# ZTC-GEN-Conditional-graph-WGAN-GP-
A conditional graph-based Wasserstein generative adversarial network with gradient penalty (WGAN-GP) was developed to investigate the generation of new ZTC architectures as a function of structural descriptors and CO₂ uptake. 
# ztc_wgan_gp — conditional graph generation of zeolite‑templated carbons



```
ZTC generation/
├── ZTC structures relaxed/*.cif          (your 68 structures, read in place)
├── Table 1 ... 68 ZTCs.docx              (read in place)
└── ztc_wgan_gp/
    ├── run_pipeline.py                   one CLI for every step
    ├── requirements.txt
    ├── data/ztc_properties.csv           Table 1 as CSV (already extracted)
    ├── data/processed/                   graph tensors + scaler (already built, 1073 graphs)
    └── ztc_gen/
        ├── cif_io.py      CIF reader/writer (no pymatgen needed)
        ├── featurize.py   CIF → G=(V,E) → X (Nmax,F), A (Nmax,Nmax,4); fragments; StandardScaler
        ├── properties.py  Table 1 .docx → CSV + sanity checks
        ├── models.py      Generator, MLP critic, graph-conv critic, gradient penalty, cVAE
        ├── train.py       WGAN-GP and cVAE training loops, checkpoints, curves
        ├── decode.py      tensors → graph → placeholder 3-D coordinates → cubic P1 CIF
        ├── evaluate.py    validity, ZTC-likeness, uniqueness, novelty, ring statistics
        └── generate.py    target properties → CIFs + report
```

## Quick start (Windows, Anaconda Prompt)

```bat

pip install torch
pip install -r requirements.txt

REM 1) ~1-2 min: 12 training steps of each model variant + generation, checks everything runs
python run_pipeline.py smoke-test
REM 2) conditional WGAN-GP, 300 epochs
python run_pipeline.py train
REM 3) 500 graphs conditioned on high CO2 uptake
python run_pipeline.py generate --target CO2_298K=45 HVF=0.93 --n 500
```

`data/processed/` is already built. Re-run `python run_pipeline.py featurize` only if you change the structures, `--nmax` or `--properties`.

**No GPU?** Use the lighter, permutation-aware variant (≈7 M parameters instead of ≈95 M):
`python run_pipeline.py train --edge-head pairwise --critic gcn --run runs/wgan_gcn`

Outputs in `runs/wgan_gp/`: `generator.pt`, `checkpoint.pt` (resumable), `config.json`, `training_log.csv`, `training_curves.png`, and after generation `generated/cif/*.cif`, `generated/generated_summary.csv`, `generated/generation_report.json`.

## What the pipeline does

### 1. Graph representation (`featurize.py`)
ZTCs are periodic all-carbon frameworks, so they have **no SMILES string**. RDKit cannot parse them, and it would reject bonds that cross the cell boundary. The graph *G = (V, E)* is therefore built directly from each CIF:

* **Edges**: C–C pairs within 1.80 Å under the minimum-image convention (bonds across the periodic boundary are kept). Across the 68 cells, bonds lie between 1.20 and 1.75 Å, and 98 % of atoms are 3-coordinated.
* **Bond order** (adjacency channels single / double / triple / aromatic): sp2–sp2 → aromatic; otherwise by length (< 1.28 Å triple, < 1.38 Å double) and coordination (sp3 → single).
* **Node features** (F = 19, RDKit-style): atom type one-hot (`none` padding, C, H, N, O), degree one-hot (0–4), hybridisation one-hot (none/sp/sp2/sp3), aromatic flag, valence, H count, atomic number, mass. All are computed in the full periodic framework.
* **Nmax = 150**: the cells hold 64–624 atoms, and only 13 have ≤ 150. Small cells are used whole and zero-padded (plus 3 BFS re-orderings as permutation augmentation). Larger cells contribute connected **breadth-first fragments of 150 atoms**, about 6 per 100 atoms. Each fragment inherits its parent's properties. The result is 1073 training graphs.
* **Conditioning vector p**: default `HVF APV SSA Mass Density CO2_298K` (change with `--properties`, e.g. add `CO2_173K CO2_673K`). It is z-scored with `StandardScaler` fitted on the 68 structures, not the fragments, so large cells don't dominate. The mean/scale are saved to `scaler.json` / `scaler.pkl` and reused at inference.

### 2. Conditional WGAN‑GP (`models.py`, `train.py`)
* **Generator** G(z, p): z ∈ ℝ¹²⁸ ‖ p → dense layers 256‑512‑512 (LayerNorm, LeakyReLU) → flattened node tensor (150×19) and edge tensor (150×150×4). The edge tensor is symmetrised with a zeroed diagonal and passed through a **sigmoid** (bond probabilities in [0, 1]). Node blocks get a softmax (one-hot blocks) or a sigmoid (flags/scalars). `--edge-head pairwise` instead predicts per-node embeddings and scores each pair.
* **Critic** D(X, A, p): X and A are flattened, projected, concatenated with p, and passed through fully connected layers (LayerNorm, never BatchNorm, because of the per-sample gradient penalty) to a scalar Wasserstein score. `--critic gcn` is a relational graph-convolution critic with gated readout (the "convolutional" option).
* **Losses**: L_D = E[D(G(z,p),p)] − E[D(x,p)] + λ·E[(‖∇_x̂ D(x̂,p)‖₂ − 1)²]. The penalty is taken over X and A jointly at random interpolates x̂ = εx + (1−ε)G(z,p). L_G = −E[D(G(z,p),p)].
* **Defaults**: n_critic = 5, λ = 10, Adam lr = 1e‑4, β = (0.0, 0.9), batch 32, 300 epochs, small Gaussian noise (σ = 0.05) on p during training to smooth the conditional manifold.
* **cVAE alternative**: `--model cvae`. A graph-conv encoder q(z|X,A,p) and the same generator as decoder; loss = per-block CE/BCE reconstruction + weighted BCE on bonds + β·KL with linear KL warm-up.

### 3. Graph → CIF decoder (`decode.py`)
1. Atom types = argmax of the atom-type block; `none` nodes are dropped.
2. Bonds: max channel probability > 0.5 (`--threshold`); type = argmax channel. Bonds are accepted most-confident first, never exceeding 4 per carbon. The largest connected component is kept.
3. Placeholder coordinates: 3-D Kamada–Kawai embedding of the bond graph, scaled to a 1.42 Å mean bond, then a quick numpy clean-up (harmonic bonds + soft non-bonded repulsion) to remove overlaps.
4. A cubic P1 cell: the edge comes from the target density (Table 1 units, kg m⁻³), but is never smaller than the molecule + 2×3 Å.
5. A CIF with an explicit `_geom_bond` loop, which opens in VESTA, ASE and pymatgen.

**These geometries are not physical.** Relax them (e.g. LAMMPS/AIREBO, ReaxFF, GFN-FF, or DFT) and then recompute HVF/APV/SSA (Zeo++/RASPA) and CO2 uptake (GCMC) before claiming any property.

### 4. Evaluation (`evaluate.py`)
Every metric is reported for the generated set **and** for the 1073 training graphs as a reference:

| metric | meaning | training reference |
|---|---|---|
| validity | connected, ≤ 4 bonds per C | 1.00 |
| ztc_like | valid **and** ≥ 60 % 3-coordinated C **and** ≥ 70 % of bonds in 5-/6-/7-rings | 0.986 |
| uniqueness | distinct Weisfeiler–Lehman graph hashes among valid graphs | 0.958 |
| novelty | valid graphs whose WL hash is not in the training set (permutation-invariant) | – |
| ring_size_distribution | smallest ring through each bond | 6: 61 %, 5: 16 %, 7: 11 % |
| frac_atoms_deg3 | sp2 fraction | 0.82 |

CIFs are written for valid, non-duplicate graphs and named `*_novel.cif` or `*_known.cif`. A `known` structure reproduces a training graph, which is a sign of memorisation.

## Generating higher-CO2 candidates
```bat
python run_pipeline.py generate --target CO2_298K=45 --n 500
python run_pipeline.py generate --target CO2_298K=50 HVF=0.94 APV=1.4 SSA=3000 --rank-by CO2_298K --top-k 5
```

