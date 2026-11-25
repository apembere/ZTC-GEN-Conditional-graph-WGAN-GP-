"""Graph-level quality metrics for generated ZTC graphs, always reported next to the same metrics
on the real training graphs (fragments have open boundaries, so 'real' is the right reference)."""
from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np

from .decode import DecodedGraph, MAX_VALENCE, tensors_to_graph


def wl_hash(dg: DecodedGraph) -> str:
    g = dg.to_nx()
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="el", edge_attr="bond", iterations=4)


def smallest_ring_per_bond(g: nx.Graph, max_ring: int = 12) -> Counter:
    """Histogram of the smallest ring containing each bond (BFS with that bond removed).
    ZTCs are dominated by 6-rings with 5-/7-rings giving curvature; 'open' = bond in no ring <= max_ring."""
    adj = {u: set(g.neighbors(u)) for u in g.nodes}
    out = Counter()
    for u, v in g.edges():
        seen, frontier, dist, found = {u}, [u], 0, None
        while frontier and dist < max_ring - 1 and found is None:
            dist += 1
            nxt = []
            for a in frontier:
                for b in adj[a]:
                    if (a == u and b == v) or b in seen:
                        continue
                    if b == v:
                        found = dist + 1
                        break
                    seen.add(b); nxt.append(b)
                if found:
                    break
            frontier = nxt
        out[found if found else "open"] += 1
    return out


def graph_stats(dg: DecodedGraph) -> dict:
    g = dg.to_nx()
    n = g.number_of_nodes()
    if n == 0:
        return dict(n_atoms=0, n_bonds=0, valid=False, frac_valence_ok=0, frac_consistent=0, frac_deg3=0,
                    largest_comp=0, rings=Counter(), ztc_like=False, frac_bonds_ring5to7=0)
    deg = np.array([d for _, d in g.degree()])
    vmax = np.array([MAX_VALENCE.get(dg.species[k], 4) for k in range(n)])
    ddeg = np.array(dg.declared_degree)
    rings = smallest_ring_per_bond(g)
    comps = list(nx.connected_components(g))
    valence_ok = deg <= vmax
    out = dict(
        n_atoms=n, n_bonds=g.number_of_edges(),
        frac_valence_ok=float(valence_ok.mean()),
        # in-graph degree never exceeds the hybridisation the node claims (open fragment boundary allowed)
        frac_consistent=float((deg <= np.maximum(ddeg, 1)).mean()),
        frac_deg3=float((deg == 3).mean()),
        largest_comp=max(len(c) for c in comps) / n,
        valid=bool(valence_ok.all() and len(comps) == 1 and n >= 4),
        rings=rings,
    )
    ring57 = sum(v for k, v in rings.items() if k in (5, 6, 7)) / max(sum(rings.values()), 1)
    out["frac_bonds_ring5to7"] = float(ring57)
    # stricter, ZTC-specific: mostly sp2 carbon whose bonds sit in 5-/6-/7-membered rings
    # (the 68 training fragments: 82 % deg-3 atoms, 89 % of bonds in 5-7 rings)
    out["ztc_like"] = bool(out["valid"] and out["frac_deg3"] >= 0.6 and ring57 >= 0.7)
    return out


def summarize(graphs, train_hashes=None, label="generated") -> dict:
    stats = [graph_stats(g) for g in graphs]
    hashes = [wl_hash(g) for g in graphs]
    valid = [s["valid"] for s in stats]
    rings = Counter()
    for s in stats:
        rings.update(s["rings"])
    tot = sum(rings.values()) or 1
    out = dict(
        label=label, n=len(graphs),
        validity=float(np.mean(valid)),
        ztc_like=float(np.mean([s["ztc_like"] for s in stats])),
        uniqueness=len(set(h for h, v in zip(hashes, valid) if v)) / max(sum(valid), 1),
        mean_atoms=float(np.mean([s["n_atoms"] for s in stats])),
        mean_bonds_per_atom=float(np.mean([s["n_bonds"] / max(s["n_atoms"], 1) for s in stats])),
        frac_atoms_valence_ok=float(np.mean([s["frac_valence_ok"] for s in stats])),
        frac_atoms_consistent=float(np.mean([s["frac_consistent"] for s in stats])),
        frac_atoms_deg3=float(np.mean([s["frac_deg3"] for s in stats])),
        mean_largest_component=float(np.mean([s["largest_comp"] for s in stats])),
        ring_size_distribution={str(k): round(v / tot, 3) for k, v in
                                sorted(rings.items(), key=lambda kv: (isinstance(kv[0], str), kv[0]))},
    )
    if train_hashes is not None:
        vh = [h for h, v in zip(hashes, valid) if v]
        out["novelty"] = float(np.mean([h not in train_hashes for h in vh])) if vh else 0.0
    return out, hashes, stats


def reference_from_dataset(d, max_graphs=None):
    X, A = d["X"], d["A"]
    idx = range(len(X)) if max_graphs is None else range(min(max_graphs, len(X)))
    return [tensors_to_graph(X[i], A[i], repair=False, largest_component=False) for i in idx]
