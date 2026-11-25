"""
Step 3 - graph-to-structure decoder: generator tensors -> discrete graph -> placeholder 3-D CIF.

1. infer atom types from the node feature block (argmax over ['none','C','H','N','O']; 'none' = padding)
2. read bonds from the adjacency tensor (exists if max_r A_ijr > threshold; type = argmax_r)
3. light chemistry repair (never more than 4 bonds per C, keeping the most probable ones)
4. placeholder coordinates: 3-D Kamada-Kawai embedding of the bond graph, rescaled so the
   mean bond length equals 1.42 A (graphitic C-C)
5. embed in a cubic periodic cell (edge from the target density if given, never smaller than
   the molecule + 2 x `vacuum`) and write a P1 CIF with an explicit _geom_bond loop

The geometry is a starting point for VESTA/ASE/pymatgen inspection and for force-field or DFT
relaxation, not a relaxed structure: the generator learns topology, not coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .cif_io import ATOMIC_MASS, write_cif
from .featurize import ATOM_TYPES, BOND_TYPES, MAX_DEGREE, block_slices

CC_BOND = 1.42
MAX_VALENCE = {"C": 4, "N": 3, "O": 2, "H": 1}


@dataclass
class DecodedGraph:
    species: list
    edges: list                      # (i, j, bond_type)
    declared_degree: list            # degree written in the node features
    edge_prob: dict = field(default_factory=dict)

    def to_nx(self) -> nx.Graph:
        g = nx.Graph()
        for k, s in enumerate(self.species):
            g.add_node(k, el=s, ddeg=self.declared_degree[k])
        for i, j, t in self.edges:
            g.add_edge(i, j, bond=t)
        return g


def tensors_to_graph(X: np.ndarray, A: np.ndarray, threshold: float = 0.5, repair: bool = True,
                     largest_component: bool = True) -> DecodedGraph:
    sl = block_slices()
    at = X[:, sl["atom_type"][0]].argmax(-1)
    keep = np.where(at != ATOM_TYPES.index("none"))[0]
    species = [ATOM_TYPES[t] for t in at[keep]]
    ddeg = X[keep][:, sl["degree"][0]].argmax(-1).tolist()
    Ak = A[np.ix_(keep, keep)]
    p_exist = Ak.max(-1)
    btype = Ak.argmax(-1)
    iu, ju = np.triu_indices(len(keep), 1)
    m = p_exist[iu, ju] > threshold
    cand = sorted(zip(iu[m], ju[m]), key=lambda e: -p_exist[e])
    edges, deg, prob = [], np.zeros(len(keep), int), {}
    for i, j in cand:                                            # greedy, most confident bonds first
        if repair and (deg[i] >= MAX_VALENCE.get(species[i], 4) or deg[j] >= MAX_VALENCE.get(species[j], 4)):
            continue
        edges.append((int(i), int(j), BOND_TYPES[btype[i, j]]))
        prob[(int(i), int(j))] = float(p_exist[i, j]); deg[i] += 1; deg[j] += 1
    dg = DecodedGraph(species, edges, ddeg, prob)
    if largest_component and edges:
        g = dg.to_nx()
        comp = max(nx.connected_components(g), key=len)
        idx = sorted(comp); remap = {o: n for n, o in enumerate(idx)}
        dg = DecodedGraph([species[o] for o in idx],
                          [(remap[i], remap[j], t) for i, j, t in edges if i in remap and j in remap],
                          [ddeg[o] for o in idx])
    return dg


def embed_coordinates(dg: DecodedGraph, seed: int = 0) -> np.ndarray:
    g = dg.to_nx()
    n = g.number_of_nodes()
    if n == 0:
        return np.zeros((0, 3))
    pos = np.zeros((n, 3))
    rng = np.random.default_rng(seed)
    offset = 0.0
    for comp in sorted(nx.connected_components(g), key=len, reverse=True):
        sub = g.subgraph(comp)
        if len(comp) == 1:
            p = {next(iter(comp)): np.zeros(3)}
        elif len(comp) <= 400:
            init = {k: rng.normal(size=3) for k in comp}
            p = nx.kamada_kawai_layout(sub, dim=3, pos=init)
        else:
            p = nx.spring_layout(sub, dim=3, seed=seed)
        P = np.array([p[k] for k in sorted(comp)])
        lens = [np.linalg.norm(p[i] - p[j]) for i, j in sub.edges()]
        if lens:
            P *= CC_BOND / max(np.mean(lens), 1e-6)
        P -= P.min(0)
        P[:, 0] += offset
        offset = P[:, 0].max() + 3.0
        for k, row in zip(sorted(comp), P):
            pos[k] = row
    return pos


def pre_relax(pos: np.ndarray, edges, steps: int = 600, r0: float = CC_BOND, r_rep: float = 2.3,
              k_rep: float = 0.5, lr: float = 0.05) -> np.ndarray:
    """Cheap clean-up: harmonic bonds at r0 + soft repulsion between non-bonded atoms closer than r_rep.
    Gradient descent with step clipping. Removes atom overlaps; it is NOT a force field."""
    n = len(pos)
    if n < 2 or not edges:
        return pos
    x = pos.copy()
    bi = np.array([e[0] for e in edges]); bj = np.array([e[1] for e in edges])
    bonded = np.zeros((n, n), bool); bonded[bi, bj] = bonded[bj, bi] = True
    np.fill_diagonal(bonded, True)
    for _ in range(steps):
        g = np.zeros_like(x)
        d = x[bi] - x[bj]; r = np.linalg.norm(d, axis=1) + 1e-9
        f = (2 * (r - r0) / r)[:, None] * d
        np.add.at(g, bi, f); np.add.at(g, bj, -f)
        D = x[:, None] - x[None]; R = np.linalg.norm(D, axis=-1) + 1e-9
        m = (~bonded) & (R < r_rep)
        if m.any():
            w = np.where(m, -2 * k_rep * (r_rep - R) / R, 0.0)
            g += (w[..., None] * D).sum(1)
        step = np.clip(lr * g, -0.2, 0.2)
        x -= step
        if np.abs(step).max() < 1e-4:
            break
    return x


def write_structure_cif(path, name, dg: DecodedGraph, target_density=None, vacuum=3.0, seed=0, comment="",
                        relax: bool = True):
    """Write a CIF; returns dict with cell edge, min non-bonded distance etc."""
    cart = embed_coordinates(dg, seed)
    if relax:
        cart = pre_relax(cart, dg.edges)
        cart -= cart.min(0)
    mass = sum(ATOMIC_MASS.get(s, 12.011) for s in dg.species)
    extent = (cart.max(0) - cart.min(0)).max() if len(cart) else 1.0
    a_min = extent + 2 * vacuum
    a_rho = None
    if target_density and target_density > 0:          # density in kg/m3 as in Table 1
        vol_A3 = mass / 6.02214076e23 / (target_density / 1000.0) * 1e24
        a_rho = vol_A3 ** (1 / 3)
    a = max(a_min, a_rho or 0.0)
    cart = cart - cart.mean(0) + a / 2
    lat = np.eye(3) * a
    frac = cart / a
    bonds = [(i, j, float(np.linalg.norm(cart[i] - cart[j]))) for i, j, _ in dg.edges]
    d = np.linalg.norm(cart[:, None] - cart[None], axis=-1) if len(cart) > 1 else np.zeros((1, 1))
    bonded = np.zeros_like(d, bool)
    for i, j, _ in dg.edges:
        bonded[i, j] = bonded[j, i] = True
    np.fill_diagonal(bonded, True)
    min_nb = float(d[~bonded].min()) if (~bonded).any() else float("nan")
    hdr = (f"Generated ZTC graph decoded by ztc_gen (placeholder geometry - relax before use)\n"
           f"atoms={len(dg.species)} bonds={len(dg.edges)} cubic a={a:.3f} A "
           f"(density-based a={a_rho if a_rho else float('nan'):.3f} A)\n{comment}")
    write_cif(path, name, lat, dg.species, frac, bonds=bonds, comment=hdr)
    return dict(cell_a=a, cell_a_from_density=a_rho, min_nonbonded=min_nb, mass=mass)
