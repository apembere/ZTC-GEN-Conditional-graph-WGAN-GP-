"""
Lightweight CIF reader / writer for periodic carbon frameworks (no pymatgen needed).

The 68 relaxed ZTC files were written by Open Babel in space group P1 with
fractional coordinates, which is what `read_cif` handles. If pymatgen is
installed it is used as a fallback for files with symmetry operators.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ATOMIC_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "B": 10.81, "S": 32.06}
ATOMIC_NUMBER = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "S": 16}


@dataclass
class Crystal:
    name: str
    lattice: np.ndarray            # (3, 3) rows = a, b, c vectors in Angstrom
    species: list[str]
    frac: np.ndarray               # (N, 3) fractional coordinates
    meta: dict = field(default_factory=dict)

    @property
    def cart(self) -> np.ndarray:
        return self.frac @ self.lattice

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def mass(self) -> float:
        return float(sum(ATOMIC_MASS.get(s, 12.011) for s in self.species))


def lattice_from_parameters(a, b, c, alpha, beta, gamma) -> np.ndarray:
    al, be, ga = np.radians([alpha, beta, gamma])
    ax = np.array([a, 0.0, 0.0])
    bx = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx_ = c * np.cos(be)
    cy_ = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz_ = np.sqrt(max(c * c - cx_ * cx_ - cy_ * cy_, 1e-12))
    return np.vstack([ax, bx, np.array([cx_, cy_, cz_])])


def _num(tok: str) -> float:
    return float(re.sub(r"\(.*\)", "", tok))


def read_cif(path: str | Path) -> Crystal:
    path = Path(path)
    text = path.read_text(errors="ignore").splitlines()
    cell = {}
    sym_ops = []
    atoms_header, atom_rows = [], []
    i = 0
    while i < len(text):
        line = text[i].strip()
        if line.startswith("_cell_length_") or line.startswith("_cell_angle_"):
            k, v = line.split()[:2]
            cell[k] = _num(v)
        if line == "loop_":
            headers = []
            i += 1
            while i < len(text) and text[i].strip().startswith("_"):
                headers.append(text[i].strip().split()[0])
                i += 1
            rows = []
            while i < len(text):
                s = text[i].strip()
                if not s or s.startswith("_") or s == "loop_" or s.startswith("data_") or s.startswith("#"):
                    break
                rows.append(shlex.split(s))
                i += 1
            if any(h.startswith("_atom_site_fract") for h in headers):
                atoms_header, atom_rows = headers, rows
            elif any("equiv_pos_as_xyz" in h or "symop_operation_xyz" in h for h in headers):
                col = [j for j, h in enumerate(headers) if "xyz" in h][0]
                sym_ops = [r[col] if len(r) > col else r[-1] for r in rows]
            continue
        i += 1

    if sym_ops and any(op.replace(" ", "").lower() != "x,y,z" for op in sym_ops):
        try:  # symmetry expansion -> defer to pymatgen when available
            from pymatgen.core import Structure
            st = Structure.from_file(str(path))
            return Crystal(path.stem, st.lattice.matrix.copy(),
                           [sp.symbol for sp in st.species], st.frac_coords.copy())
        except ImportError as e:
            raise RuntimeError(f"{path.name}: non-P1 CIF needs pymatgen") from e

    lat = lattice_from_parameters(cell["_cell_length_a"], cell["_cell_length_b"], cell["_cell_length_c"],
                                  cell["_cell_angle_alpha"], cell["_cell_angle_beta"], cell["_cell_angle_gamma"])
    h = {k: j for j, k in enumerate(atoms_header)}
    sym_col = h.get("_atom_site_type_symbol", h.get("_atom_site_label"))
    species, frac = [], []
    for r in atom_rows:
        sym = re.match(r"[A-Z][a-z]?", r[sym_col]).group(0)
        species.append(sym)
        frac.append([_num(r[h["_atom_site_fract_x"]]), _num(r[h["_atom_site_fract_y"]]),
                     _num(r[h["_atom_site_fract_z"]])])
    frac = np.asarray(frac, dtype=float) % 1.0
    return Crystal(path.stem, lat, species, frac)


def write_cif(path: str | Path, name: str, lattice: np.ndarray, species: list[str], frac: np.ndarray,
              bonds: list[tuple[int, int, float]] | None = None, comment: str = "") -> None:
    a, b, c = np.linalg.norm(lattice, axis=1)
    def ang(u, v):
        return np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
    alpha, beta, gamma = ang(lattice[1], lattice[2]), ang(lattice[0], lattice[2]), ang(lattice[0], lattice[1])
    labels = [f"{s}{k + 1}" for k, s in enumerate(species)]
    out = []
    if comment:
        out += [f"# {ln}" for ln in comment.splitlines()]
    out += [f"data_{name}",
            "_symmetry_space_group_name_H-M   'P 1'",
            "_symmetry_Int_Tables_number      1",
            "loop_", "_symmetry_equiv_pos_as_xyz", "  'x, y, z'",
            f"_cell_length_a    {a:.6f}", f"_cell_length_b    {b:.6f}", f"_cell_length_c    {c:.6f}",
            f"_cell_angle_alpha {alpha:.4f}", f"_cell_angle_beta  {beta:.4f}", f"_cell_angle_gamma {gamma:.4f}",
            "loop_", "  _atom_site_label", "  _atom_site_type_symbol",
            "  _atom_site_fract_x", "  _atom_site_fract_y", "  _atom_site_fract_z", "  _atom_site_occupancy"]
    for lab, s, f in zip(labels, species, frac):
        out.append(f"  {lab:<6} {s:<3} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f} 1.0")
    if bonds:
        out += ["loop_", "  _geom_bond_atom_site_label_1", "  _geom_bond_atom_site_label_2",
                "  _geom_bond_distance"]
        for i, j, d in bonds:
            out.append(f"  {labels[i]:<6} {labels[j]:<6} {d:.4f}")
    Path(path).write_text("\n".join(out) + "\n")
