"""
Step 1 - convert periodic ZTC crystals into machine-learning-ready graph tensors.

Why not SMILES/RDKit?
    ZTCs are infinite, periodic, all-carbon frameworks. They have no finite
    SMILES string, and RDKit's sanitiser rejects the periodic graph (bonds
    cross the cell boundary). We therefore build the molecular graph G=(V,E)
    directly from the CIF: atoms are nodes and C-C bonds are found with
    minimum-image distances (cutoff 1.80 A, see `BOND_CUTOFF`). The node
    features are the same RDKit-style atom descriptors (atomic number,
    valence, degree, aromaticity, hybridisation, H count, mass), computed
    from the periodic graph so that bonds across the cell boundary count.

Nmax = 150:
    The 68 unit cells contain 64-624 carbon atoms (only 13 have <= 150).
    Structures with N <= Nmax are used whole (zero-padded). Larger cells are
    represented by several connected, breadth-first subgraphs ("fragments") of
    exactly Nmax atoms, each carrying the parent's conditioning properties.
    This keeps the tensor shape fixed at (Nmax, F) and (Nmax, Nmax, 4),
    and it also augments a 68-structure dataset to around 1-2 thousand training graphs.
    Node features always describe the atom in the *full* periodic framework,
    so a fragment's boundary atoms keep their true hybridisation.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cif_io import ATOMIC_MASS, ATOMIC_NUMBER, Crystal, read_cif

BOND_CUTOFF = 1.80          # Angstrom; relaxed ZTC C-C bonds span 1.20-1.75 A
BOND_TYPES = ["single", "double", "triple", "aromatic"]   # adjacency channels
ATOM_TYPES = ["none", "C", "H", "N", "O"]                 # 'none' = padding node
HYBRID = ["none", "sp", "sp2", "sp3"]
MAX_DEGREE = 4

# feature schema: (block name, width, kind) - kind controls the generator's output activation
FEATURE_BLOCKS = [
    ("atom_type", len(ATOM_TYPES), "onehot"),
    ("degree", MAX_DEGREE + 1, "onehot"),       # 0..4
    ("hybridization", len(HYBRID), "onehot"),
    ("aromatic", 1, "binary"),
    ("valence", 1, "scalar"),                   # sum of bond orders / 4
    ("num_h", 1, "scalar"),                     # attached H / 4
    ("atomic_number", 1, "scalar"),             # Z / 8
    ("mass", 1, "scalar"),                      # mass / 16
]
N_FEATURES = sum(w for _, w, _ in FEATURE_BLOCKS)
BOND_ORDER = {"single": 1.0, "double": 2.0, "triple": 3.0, "aromatic": 1.5}


def block_slices():
    out, s = {}, 0
    for name, w, kind in FEATURE_BLOCKS:
        out[name] = (slice(s, s + w), kind)
        s += w
    return out


# ---------------------------------------------------------------- periodic bonding
def periodic_bonds(cr: Crystal, cutoff: float = BOND_CUTOFF):
    """Return list of (i, j, distance) for every bonded pair (i<j, nearest image only,
    plus i-j pairs that are bonded through two different images are kept once)."""
    lat = cr.lattice
    heights = 1.0 / np.linalg.norm(np.linalg.inv(lat), axis=0)   # interplanar spacings
    n = np.ceil(cutoff / heights).astype(int)
    shifts = np.array([[i, j, k] for i in range(-n[0], n[0] + 1)
                       for j in range(-n[1], n[1] + 1) for k in range(-n[2], n[2] + 1)], float)
    img = shifts @ lat                                           # (S,3)
    X = cr.cart
    bonds = []
    for i in range(cr.n_atoms):
        d = np.linalg.norm(X[i][None, None, :] - X[:, None, :] - img[None, :, :], axis=-1)  # (N,S)
        d[i][np.all(shifts == 0, axis=1)] = np.inf
        dmin = d.min(axis=1)
        for j in np.where(dmin < cutoff)[0]:
            if j > i:
                bonds.append((i, int(j), float(dmin[j])))
    return bonds


def assign_bond_orders(n_atoms: int, bonds, species):
    """Heuristic bond orders for sp/sp2/sp3 carbon (RDKit-like perception)."""
    deg = np.zeros(n_atoms, int)
    for i, j, _ in bonds:
        deg[i] += 1
        deg[j] += 1
    typed = []
    for i, j, d in bonds:
        if "H" in (species[i], species[j]) or deg[i] >= 4 or deg[j] >= 4:
            t = "single"
        elif deg[i] == 3 and deg[j] == 3:
            t = "aromatic"                       # delocalised graphene-like C=C
        elif d < 1.28:
            t = "triple"
        elif d < 1.38:
            t = "double"
        elif deg[i] == 3 or deg[j] == 3:
            t = "aromatic" if d < 1.47 else "single"
        else:
            t = "single"
        typed.append((i, j, d, t))
    return typed, deg


def node_features(species, deg, typed_bonds) -> np.ndarray:
    n = len(species)
    sl = block_slices()
    F = np.zeros((n, N_FEATURES), np.float32)
    val = np.zeros(n)
    nh = np.zeros(n)
    arom = np.zeros(n, bool)
    for i, j, _, t in typed_bonds:
        val[i] += BOND_ORDER[t]
        val[j] += BOND_ORDER[t]
        if t == "aromatic":
            arom[i] = arom[j] = True
        if species[j] == "H":
            nh[i] += 1
        if species[i] == "H":
            nh[j] += 1
    for k, s in enumerate(species):
        F[k, sl["atom_type"][0].start + ATOM_TYPES.index(s if s in ATOM_TYPES else "C")] = 1
        F[k, sl["degree"][0].start + min(int(deg[k]), MAX_DEGREE)] = 1
        hyb = {0: "none", 1: "sp", 2: "sp", 3: "sp2", 4: "sp3"}[min(int(deg[k]), 4)]
        if s == "H":
            hyb = "none"
        F[k, sl["hybridization"][0].start + HYBRID.index(hyb)] = 1
        F[k, sl["aromatic"][0].start] = float(arom[k])
        F[k, sl["valence"][0].start] = min(val[k], 4.0) / 4.0
        F[k, sl["num_h"][0].start] = nh[k] / 4.0
        F[k, sl["atomic_number"][0].start] = ATOMIC_NUMBER.get(s, 6) / 8.0
        F[k, sl["mass"][0].start] = ATOMIC_MASS.get(s, 12.011) / 16.0
    return F


@dataclass
class PeriodicGraph:
    name: str
    species: list
    features: np.ndarray          # (N, F)
    edges: list                   # (i, j, bond_type_index)
    n_atoms: int
    lattice: np.ndarray
    volume: float
    mass: float


def crystal_to_graph(cr: Crystal) -> PeriodicGraph:
    bonds = periodic_bonds(cr)
    typed, deg = assign_bond_orders(cr.n_atoms, bonds, cr.species)
    feats = node_features(cr.species, deg, typed)
    edges = [(i, j, BOND_TYPES.index(t)) for i, j, _, t in typed]
    return PeriodicGraph(cr.name, cr.species, feats, edges, cr.n_atoms, cr.lattice, cr.volume, cr.mass)


# ---------------------------------------------------------------- fixed-size tensors
def _neighbors(g: PeriodicGraph):
    nb = [[] for _ in range(g.n_atoms)]
    for i, j, t in g.edges:
        nb[i].append((j, t))
        nb[j].append((i, t))
    return nb


def bfs_fragment(g: PeriodicGraph, seed: int, nmax: int, rng: np.random.Generator):
    """Connected subgraph of up to nmax atoms in BFS order (randomised tie-breaking)."""
    nb = _neighbors(g)
    order, seen, q = [], {seed}, deque([seed])
    while q and len(order) < nmax:
        u = q.popleft()
        order.append(u)
        nxt = [v for v, _ in nb[u] if v not in seen]
        rng.shuffle(nxt)
        for v in nxt:
            seen.add(v)
            q.append(v)
    return order


def graph_to_tensors(g: PeriodicGraph, order, nmax: int):
    """Dense padded tensors X (nmax, F) and A (nmax, nmax, 4) for atoms in `order`."""
    X = np.zeros((nmax, N_FEATURES), np.float32)
    A = np.zeros((nmax, nmax, len(BOND_TYPES)), np.uint8)
    idx = {a: k for k, a in enumerate(order)}
    X[: len(order)] = g.features[order]
    for i, j, t in g.edges:
        if i in idx and j in idx:
            A[idx[i], idx[j], t] = A[idx[j], idx[i], t] = 1
    # padding nodes: atom_type 'none', hybridisation 'none', degree 0
    sl = block_slices()
    for k in range(len(order), nmax):
        X[k, sl["atom_type"][0].start] = 1
        X[k, sl["degree"][0].start] = 1
        X[k, sl["hybridization"][0].start] = 1
    return X, A


def build_dataset(cif_dir: str | Path, properties_csv: str | Path, out_dir: str | Path,
                  nmax: int = 150, fragments_per_100_atoms: float = 6.0, min_fragments: int = 4,
                  property_columns=None, seed: int = 0, verbose: bool = True):
    """Featurise every CIF and save `graph_dataset.npz`, `scaler.json`, `scaler.pkl`."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    cif_dir, out_dir = Path(cif_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    props = pd.read_csv(properties_csv)
    if property_columns is None:
        property_columns = [c for c in props.columns if c not in ("name", "sno")]
    props = props.set_index("name")

    Xs, As, Ps, parent, natoms, frag_sizes = [], [], [], [], [], []
    graphs_meta = []
    for f in sorted(cif_dir.glob("*.cif")):
        name = f.stem.replace("_nozeolite_relaxed", "")
        if name not in props.index:
            print(f"  ! {name}: no row in property table, skipped")
            continue
        g = crystal_to_graph(read_cif(f))
        degs = np.bincount([i for e in g.edges for i in e[:2]], minlength=g.n_atoms)
        graphs_meta.append(dict(name=name, n_atoms=g.n_atoms, n_bonds=len(g.edges),
                                frac_sp2=float(np.mean(degs == 3)), volume=g.volume, mass=g.mass))
        if g.n_atoms <= nmax:
            orders = [list(range(g.n_atoms))]
            # extra copies with BFS re-ordering = permutation augmentation
            k_extra = max(min_fragments - 1, 0)
            orders += [bfs_fragment(g, int(rng.integers(g.n_atoms)), nmax, rng) for _ in range(k_extra)]
        else:
            k = max(min_fragments, int(round(fragments_per_100_atoms * g.n_atoms / 100)))
            seeds = rng.choice(g.n_atoms, size=k, replace=k > g.n_atoms)
            orders = [bfs_fragment(g, int(s), nmax, rng) for s in seeds]
        for o in orders:
            X, A = graph_to_tensors(g, o, nmax)
            Xs.append(X); As.append(A)
            Ps.append(props.loc[name, property_columns].to_numpy(float))
            parent.append(name); natoms.append(g.n_atoms); frag_sizes.append(len(o))
        if verbose:
            print(f"  {name:<10} N={g.n_atoms:>4}  bonds={len(g.edges):>4}  sp2={graphs_meta[-1]['frac_sp2']:.2f}"
                  f"  -> {len(orders)} graph(s)")

    X = np.stack(Xs); A = np.stack(As); P = np.stack(Ps)
    scaler = StandardScaler().fit(P[np.unique(parent, return_index=True)[1]])  # fit on structures, not fragments
    P_scaled = scaler.transform(P).astype(np.float32)

    # store adjacency sparsely (it is >99 % zeros)
    s_idx, i_idx, j_idx, t_idx = np.nonzero(A)
    np.savez_compressed(out_dir / "graph_dataset.npz", X=X, P=P.astype(np.float32), P_scaled=P_scaled,
                        A_s=s_idx.astype(np.int32), A_i=i_idx.astype(np.int16), A_j=j_idx.astype(np.int16),
                        A_t=t_idx.astype(np.int8), A_shape=np.array(A.shape),
                        parent=np.array(parent), parent_n_atoms=np.array(natoms),
                        frag_size=np.array(frag_sizes), property_columns=np.array(property_columns))
    meta = dict(nmax=nmax, n_features=N_FEATURES, feature_blocks=FEATURE_BLOCKS, bond_types=BOND_TYPES,
                atom_types=ATOM_TYPES, hybridization=HYBRID, bond_cutoff=BOND_CUTOFF,
                property_columns=list(property_columns),
                scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                n_graphs=int(len(X)), n_structures=len(graphs_meta))
    (out_dir / "scaler.json").write_text(json.dumps(meta, indent=2))
    try:
        import joblib
        joblib.dump(scaler, out_dir / "scaler.pkl")
    except ImportError:
        import pickle
        (out_dir / "scaler.pkl").write_bytes(pickle.dumps(scaler))
    pd.DataFrame(graphs_meta).to_csv(out_dir / "structure_graph_summary.csv", index=False)
    if verbose:
        print(f"\nSaved {len(X)} graphs from {len(graphs_meta)} structures -> {out_dir/'graph_dataset.npz'}")
        print(f"X: {X.shape}  A: {A.shape}  P: {P.shape}  properties: {property_columns}")
    return meta


def load_dataset(path: str | Path):
    """Return X (S,N,F) float32, A (S,N,N,4) float32, P_scaled, P, parent, meta-dict."""
    d = np.load(path, allow_pickle=False)
    A = np.zeros(tuple(d["A_shape"]), np.float32)
    A[d["A_s"], d["A_i"], d["A_j"], d["A_t"]] = 1.0
    return dict(X=d["X"], A=A, P_scaled=d["P_scaled"], P=d["P"], parent=d["parent"],
                frag_size=d["frag_size"], property_columns=[str(c) for c in d["property_columns"]])
