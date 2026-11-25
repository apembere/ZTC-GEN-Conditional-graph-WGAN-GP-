"""Extract Table 1 (68 ZTCs: Mass, Density, HVF, APV, SSA, CO2 uptake) from the .docx into a CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["sno", "name", "Mass", "Density", "HVF", "APV", "SSA", "CO2_173K", "CO2_298K", "CO2_673K"]
# Mass: g/mol per cell, Density: kg/m3, HVF: helium void fraction, APV: accessible pore volume (cm3/g),
# SSA: surface area (m2/g), CO2_*: calculated CO2 loading (wt%)


def table_docx_to_csv(docx_path: str | Path, csv_path: str | Path) -> pd.DataFrame:
    import docx
    doc = docx.Document(str(docx_path))
    rows = []
    for t in doc.tables:
        for r in t.rows:
            cells = [c.text.strip() for c in r.cells]
            if cells and cells[0].isdigit():
                rows.append(cells[:10])
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in COLUMNS[2:]:
        df[c] = pd.to_numeric(df[c].replace({"N/A": np.nan, "-": np.nan}), errors="coerce")
    # CHA has SSA = N/A: its HVF (0.001) and APV (0.001) show it is non-porous, so SSA = 0
    df["SSA"] = df["SSA"].fillna(0.0)
    df["sno"] = df["sno"].astype(int)
    df.to_csv(csv_path, index=False)
    return df


def sanity_flags(df: pd.DataFrame) -> list[str]:
    """CO2 uptake should fall monotonically with temperature; flag rows where it does not."""
    out = []
    for _, r in df.iterrows():
        a, b, c = r["CO2_173K"], r["CO2_298K"], r["CO2_673K"]
        if b < c:
            out.append(f"{r['name']}: CO2 at 298 K ({b}) < 673 K ({c})")
        elif a > 0 and (a == b or b == c):
            out.append(f"{r['name']}: identical uptake at two temperatures ({a}, {b}, {c})")
    return out
