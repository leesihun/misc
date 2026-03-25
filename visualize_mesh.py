"""
visualize_mesh.py
=================
Generate images from:
  1. An ANSYS APDL .inp file mesh (nodes + edges, colored by part)
  2. A generated .h5 dataset file (B8_REV04_SEC_BOT.h5) with mesh
     colored by z-displacement (feature index 5 per DATASET_FORMAT.md)

Both files are chosen via a tkinter GUI file-picker dialog.
CLI flags (--inp, --h5) can pre-fill the paths and skip the dialog.

Usage
-----
python visualize_mesh.py                          # GUI for both files
python visualize_mesh.py --inp mesh.inp           # GUI only for .h5
python visualize_mesh.py --inp mesh.inp --h5 x.h5 # no GUI needed

Dependencies: numpy, h5py, matplotlib, tkinter (stdlib)
"""

from __future__ import annotations

import argparse
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Minimal ANSYS APDL .inp parser (nodes + edges only)
# ---------------------------------------------------------------------------

def _parse_nblock(lines: list[str], start: int) -> tuple[dict[int, np.ndarray], int]:
    """Parse a NBLOCK section. Returns {node_id: xyz_array} and next line index."""
    # Line after NBLOCK header is a format descriptor, skip it
    i = start + 1
    nodes: dict[int, np.ndarray] = {}
    while i < len(lines):
        line = lines[i]
        i += 1
        stripped = line.strip()
        if stripped.upper().startswith("N,R5.3"):
            continue  # format line
        if stripped == "" or stripped.startswith("!") or stripped.upper() == "-1":
            break
        # Fixed-width NBLOCK record: field widths depend on format declaration,
        # but ANSYS typically writes: node_id(8) ?(8) ?(8) x y z (each 16 or 20 chars)
        # We try a robust approach: split after the 3 leading integers
        try:
            # Each NBLOCK line: nid solid_model kp x y z  (fixed width)
            # Try parsing with known ANSYS widths: nid(8) + 2×int(8) + 3×float(20)
            nid = int(line[0:8])
            x = float(line[24:44])
            y = float(line[44:64])
            z = float(line[64:84]) if len(line) > 64 else 0.0
            nodes[nid] = np.array([x, y, z], dtype=np.float64)
        except (ValueError, IndexError):
            # Fallback: split by whitespace
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    nid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[nid] = np.array([x, y, z], dtype=np.float64)
                except ValueError:
                    pass
    return nodes, i


def _parse_eblock(lines: list[str], start: int) -> tuple[list[list[int]], int]:
    """Parse an EBLOCK section. Returns list of element node-id lists."""
    # EBLOCK header: EBLOCK,19,SOLID, nelem
    # Next line: format descriptor, skip it
    i = start + 1
    elems: list[list[int]] = []
    while i < len(lines):
        line = lines[i]
        i += 1
        stripped = line.strip()
        if stripped == "" or stripped.startswith("!") or stripped == "-1":
            break
        # EBLOCK record (solid): 19 fixed integers, first 8 are metadata
        # Columns: mat type real sec cs death ndiv nn  e_id  n1 n2 ... nN
        # We care about nn (num nodes, col index 8) and node ids starting col 9
        parts = stripped.split()
        if len(parts) < 11:
            continue
        try:
            nn = int(parts[8])    # number of nodes in element
            e_id = int(parts[9])  # element id (unused here)
            node_ids = [int(v) for v in parts[10:10 + nn]]
            if len(node_ids) == nn and nn > 0:
                elems.append(node_ids)
        except (ValueError, IndexError):
            pass
    return elems, i


def parse_inp_for_viz(inp_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse an ANSYS APDL .inp file for visualization.

    Returns
    -------
    coords  : (N, 3) float32   – node coordinates ordered by compact index
    edges   : (E, 2) int64     – unique edges as compact node indices
    part_ids: (N,) int32       – part index per node (0-based), always 0 if unknown
    """
    with open(inp_path, "r", errors="replace") as fh:
        lines = fh.readlines()

    nodes: dict[int, np.ndarray] = {}
    elems: list[list[int]] = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        upper = raw.strip().upper()

        if upper.startswith("NBLOCK"):
            new_nodes, i = _parse_nblock(lines, i)
            nodes.update(new_nodes)
            continue

        if upper.startswith("EBLOCK"):
            new_elems, i = _parse_eblock(lines, i)
            elems.extend(new_elems)
            continue

        # Individual node: N,nid,x,y,z
        m = re.match(r"^\s*N\s*,\s*(\d+)\s*,\s*([Ee0-9.+\-]+)\s*,\s*([Ee0-9.+\-]+)(?:\s*,\s*([Ee0-9.+\-]+))?", raw, re.IGNORECASE)
        if m:
            nid = int(m.group(1))
            x, y = float(m.group(2)), float(m.group(3))
            z = float(m.group(4)) if m.group(4) else 0.0
            nodes[nid] = np.array([x, y, z], dtype=np.float64)
            i += 1
            continue

        i += 1

    if not nodes:
        raise ValueError(f"No nodes found in {inp_path}")

    # Build compact mapping: global node id → compact index
    sorted_nids = sorted(nodes.keys())
    nid_to_idx: dict[int, int] = {nid: idx for idx, nid in enumerate(sorted_nids)}

    coords = np.array([nodes[nid] for nid in sorted_nids], dtype=np.float32)

    # Extract unique edges from element connectivity
    edge_set: set[tuple[int, int]] = set()
    for elem_nodes in elems:
        valid = [nid_to_idx[n] for n in elem_nodes if n in nid_to_idx]
        n = len(valid)
        for k in range(n):
            a, b = valid[k], valid[(k + 1) % n]
            edge_set.add((min(a, b), max(a, b)))

    edges = np.array(sorted(edge_set), dtype=np.int64) if edge_set else np.zeros((0, 2), dtype=np.int64)
    part_ids = np.zeros(len(coords), dtype=np.int32)

    print(f"[INP]  Nodes: {len(coords):,}   Edges: {len(edges):,}")
    return coords, edges, part_ids


# ---------------------------------------------------------------------------
# HDF5 loader (DATASET_FORMAT.md)
# ---------------------------------------------------------------------------

# Feature indices per spec
FEAT_X      = 0
FEAT_Y      = 1
FEAT_Z      = 2
FEAT_XDISP  = 3
FEAT_YDISP  = 4
FEAT_ZDISP  = 5
FEAT_STRESS = 6
FEAT_PART   = 7


def load_h5_sample(h5_path: Path, sample_id: int = 1):
    """
    Load one sample from the dataset following DATASET_FORMAT.md.

    Returns
    -------
    coords   : (N, 3) float32  – [x, y, z] per node
    edges    : (E, 2) int64    – undirected edges
    z_disp   : (N,)  float32  – z-displacement per node
    meta     : dict            – sample metadata attributes
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[f"data/{sample_id}"]
        nd  = grp["nodal_data"][:]        # (8, timesteps, N)
        ei  = grp["mesh_edge"][:]         # (2, E)
        meta = dict(grp["metadata"].attrs)

    # timestep 0, transpose to (N, 8)
    x = nd[:, 0, :].T.astype(np.float32)

    coords = x[:, FEAT_X:FEAT_Z+1]       # (N, 3)
    z_disp = x[:, FEAT_ZDISP]            # (N,)
    edges  = ei.T.astype(np.int64)        # (E, 2)

    print(f"[H5]   Nodes: {len(coords):,}   Edges: {len(edges):,}"
          f"   z_disp range: [{z_disp.min():.3f}, {z_disp.max():.3f}] mm")
    return coords, edges, z_disp, meta


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _make_edge_segments_2d(coords: np.ndarray, edges: np.ndarray,
                            xi: int = 0, yi: int = 1) -> np.ndarray:
    """Build (E, 2, 2) segment array for LineCollection from 2-D projection."""
    if len(edges) == 0:
        return np.zeros((0, 2, 2))
    src = edges[:, 0]
    dst = edges[:, 1]
    segs = np.stack([coords[src][:, [xi, yi]],
                     coords[dst][:, [xi, yi]]], axis=1)  # (E, 2, 2)
    return segs


def _axis_labels(view: str) -> tuple[str, str, int, int]:
    mapping = {
        "xy": ("X (mm)", "Y (mm)", 0, 1),
        "xz": ("X (mm)", "Z (mm)", 0, 2),
        "yz": ("Y (mm)", "Z (mm)", 1, 2),
    }
    return mapping.get(view, ("X (mm)", "Y (mm)", 0, 1))


MAX_EDGE_DRAW = 300_000   # cap edges drawn to keep rendering fast


def plot_inp_mesh(coords: np.ndarray, edges: np.ndarray,
                  part_ids: np.ndarray, view: str = "xy",
                  dpi: int = 150) -> plt.Figure:
    """Figure 1: .inp mesh colored by part number."""
    xlabel, ylabel, xi, yi = _axis_labels(view)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    # Draw edges (subsample if huge)
    if len(edges) > 0:
        draw_edges = edges if len(edges) <= MAX_EDGE_DRAW else edges[
            np.random.choice(len(edges), MAX_EDGE_DRAW, replace=False)]
        segs = _make_edge_segments_2d(coords, draw_edges, xi, yi)
        lc = LineCollection(segs, linewidths=0.3, colors="steelblue", alpha=0.4)
        ax.add_collection(lc)

    # Scatter nodes colored by part
    sc = ax.scatter(coords[:, xi], coords[:, yi],
                    c=part_ids, cmap="tab10", s=0.5, alpha=0.6,
                    linewidths=0, rasterized=True)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("INP Mesh — Nodes & Edges")
    plt.colorbar(sc, ax=ax, label="Part index", fraction=0.03, pad=0.02,
                 ticks=np.unique(part_ids))
    fig.tight_layout()
    return fig


def plot_h5_zdisp(coords: np.ndarray, edges: np.ndarray,
                  z_disp: np.ndarray, meta: dict,
                  view: str = "xy", dpi: int = 150) -> plt.Figure:
    """Figure 2: H5 mesh colored by z-displacement."""
    xlabel, ylabel, xi, yi = _axis_labels(view)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=dpi)

    # Draw edges
    if len(edges) > 0:
        draw_edges = edges if len(edges) <= MAX_EDGE_DRAW else edges[
            np.random.choice(len(edges), MAX_EDGE_DRAW, replace=False)]
        segs = _make_edge_segments_2d(coords, draw_edges, xi, yi)
        lc = LineCollection(segs, linewidths=0.25, colors="black", alpha=0.25)
        ax.add_collection(lc)

    # Scatter nodes colored by z_disp
    vmin, vmax = np.percentile(z_disp, [1, 99])   # robust color range
    sc = ax.scatter(coords[:, xi], coords[:, yi],
                    c=z_disp, cmap="coolwarm",
                    vmin=vmin, vmax=vmax,
                    s=1.0, alpha=0.8, linewidths=0, rasterized=True)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    n_nodes = meta.get("num_nodes", len(coords))
    n_edges = meta.get("num_edges", len(edges))
    ax.set_title(
        f"H5 Mesh — Z-Displacement  "
        f"(nodes={n_nodes:,}, edges={n_edges:,})"
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("z_disp (mm)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self, init_inp: Path | None, init_h5: Path | None,
                 init_sample: int, init_view: str,
                 init_dpi: int, init_outdir: Path):
        super().__init__()
        self.title("Mesh Visualizer")
        self.resizable(False, False)

        PAD = dict(padx=8, pady=4)

        # ── INP file row ──────────────────────────────────────────────────
        tk.Label(self, text="INP mesh file:", anchor="w").grid(
            row=0, column=0, sticky="w", **PAD)
        self._inp_var = tk.StringVar(value=str(init_inp) if init_inp else "")
        tk.Entry(self, textvariable=self._inp_var, width=55).grid(
            row=0, column=1, **PAD)
        tk.Button(self, text="Browse…", command=self._browse_inp).grid(
            row=0, column=2, **PAD)
        tk.Button(self, text="Clear", command=lambda: self._inp_var.set("")).grid(
            row=0, column=3, **PAD)

        # ── H5 file row ───────────────────────────────────────────────────
        tk.Label(self, text="H5 dataset file:", anchor="w").grid(
            row=1, column=0, sticky="w", **PAD)
        self._h5_var = tk.StringVar(value=str(init_h5) if init_h5 else "")
        tk.Entry(self, textvariable=self._h5_var, width=55).grid(
            row=1, column=1, **PAD)
        tk.Button(self, text="Browse…", command=self._browse_h5).grid(
            row=1, column=2, **PAD)
        tk.Button(self, text="Clear", command=lambda: self._h5_var.set("")).grid(
            row=1, column=3, **PAD)

        # ── Options row ───────────────────────────────────────────────────
        opt_frame = tk.Frame(self)
        opt_frame.grid(row=2, column=0, columnspan=4, sticky="w", **PAD)

        tk.Label(opt_frame, text="Sample ID:").pack(side="left")
        self._sample_var = tk.IntVar(value=init_sample)
        tk.Spinbox(opt_frame, textvariable=self._sample_var,
                   from_=1, to=99999, width=6).pack(side="left", padx=(2, 12))

        tk.Label(opt_frame, text="View:").pack(side="left")
        self._view_var = tk.StringVar(value=init_view)
        ttk.Combobox(opt_frame, textvariable=self._view_var,
                     values=["xy", "xz", "yz"], width=4,
                     state="readonly").pack(side="left", padx=(2, 12))

        tk.Label(opt_frame, text="DPI:").pack(side="left")
        self._dpi_var = tk.IntVar(value=init_dpi)
        tk.Spinbox(opt_frame, textvariable=self._dpi_var,
                   from_=72, to=600, increment=25, width=5).pack(side="left", padx=(2, 12))

        # ── Output dir row ────────────────────────────────────────────────
        tk.Label(self, text="Output folder:", anchor="w").grid(
            row=3, column=0, sticky="w", **PAD)
        self._outdir_var = tk.StringVar(value=str(init_outdir))
        tk.Entry(self, textvariable=self._outdir_var, width=55).grid(
            row=3, column=1, **PAD)
        tk.Button(self, text="Browse…", command=self._browse_outdir).grid(
            row=3, column=2, **PAD)

        # ── Status / progress ─────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self._status_var, anchor="w",
                 fg="navy").grid(row=4, column=0, columnspan=4,
                                 sticky="w", **PAD)
        self._pbar = ttk.Progressbar(self, mode="indeterminate", length=400)
        self._pbar.grid(row=5, column=0, columnspan=4, **PAD)

        # ── Generate button ───────────────────────────────────────────────
        self._gen_btn = tk.Button(self, text="Generate images",
                                  command=self._on_generate,
                                  bg="#1a73e8", fg="white",
                                  font=("sans-serif", 11, "bold"),
                                  padx=16, pady=6)
        self._gen_btn.grid(row=6, column=0, columnspan=4, pady=10)

    # ── File dialogs ──────────────────────────────────────────────────────

    def _browse_inp(self):
        path = filedialog.askopenfilename(
            title="Select ANSYS APDL .inp file",
            filetypes=[("APDL input files", "*.inp *.cdb *.dat"), ("All files", "*")],
        )
        if path:
            self._inp_var.set(path)

    def _browse_h5(self):
        path = filedialog.askopenfilename(
            title="Select HDF5 dataset file",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*")],
        )
        if path:
            self._h5_var.set(path)

    def _browse_outdir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._outdir_var.set(path)

    # ── Generate ──────────────────────────────────────────────────────────

    def _on_generate(self):
        inp_str  = self._inp_var.get().strip()
        h5_str   = self._h5_var.get().strip()

        if not inp_str and not h5_str:
            messagebox.showerror("No files selected",
                                 "Please select at least one file (INP or H5).")
            return

        inp_path  = Path(inp_str)  if inp_str  else None
        h5_path   = Path(h5_str)   if h5_str   else None
        out_dir   = Path(self._outdir_var.get().strip() or "figures")
        sample_id = self._sample_var.get()
        view      = self._view_var.get()
        dpi       = self._dpi_var.get()

        # Validate
        if inp_path and not inp_path.is_file():
            messagebox.showerror("File not found", f"INP not found:\n{inp_path}")
            return
        if h5_path and not h5_path.is_file():
            messagebox.showerror("File not found", f"H5 not found:\n{h5_path}")
            return

        # Run in background thread so the GUI stays responsive
        self._gen_btn.config(state="disabled")
        self._pbar.start(12)
        thread = threading.Thread(
            target=self._generate,
            args=(inp_path, h5_path, sample_id, view, dpi, out_dir),
            daemon=True,
        )
        thread.start()

    def _generate(self, inp_path, h5_path, sample_id, view, dpi, out_dir):
        outputs = []
        try:
            out_dir.mkdir(parents=True, exist_ok=True)

            if inp_path:
                self._set_status(f"Parsing {inp_path.name}…")
                coords, edges, part_ids = parse_inp_for_viz(inp_path)
                self._set_status("Rendering INP mesh…")
                fig = plot_inp_mesh(coords, edges, part_ids, view=view, dpi=dpi)
                out = out_dir / f"inp_mesh_{view}.png"
                fig.savefig(out, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                outputs.append(str(out))

            if h5_path:
                self._set_status(f"Loading {h5_path.name} sample {sample_id}…")
                coords, edges, z_disp, meta = load_h5_sample(h5_path, sample_id)
                self._set_status("Rendering H5 z-disp mesh…")
                fig = plot_h5_zdisp(coords, edges, z_disp, meta, view=view, dpi=dpi)
                out = out_dir / f"{h5_path.stem}_zdisp_{view}.png"
                fig.savefig(out, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                outputs.append(str(out))

            msg = "Saved:\n" + "\n".join(outputs)
            self._set_status("Done. " + "  |  ".join(outputs))
            self.after(0, lambda: messagebox.showinfo("Done", msg))

        except Exception as exc:
            self._set_status(f"Error: {exc}")
            self.after(0, lambda: messagebox.showerror("Error", str(exc)))

        finally:
            self.after(0, self._pbar.stop)
            self.after(0, lambda: self._gen_btn.config(state="normal"))

    def _set_status(self, msg: str):
        self.after(0, lambda: self._status_var.set(msg))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize INP mesh and H5 dataset sample (GUI file picker).")
    parser.add_argument("--inp",        type=Path, default=None,
                        help="Pre-fill INP path (skips that file dialog)")
    parser.add_argument("--h5",         type=Path, default=None,
                        help="Pre-fill H5 path (skips that file dialog)")
    parser.add_argument("--sample-id",  type=int,  default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi",        type=int,  default=150)
    parser.add_argument("--view",       choices=["xy", "xz", "yz"], default="xy")
    args = parser.parse_args()

    app = App(
        init_inp=args.inp,
        init_h5=args.h5,
        init_sample=args.sample_id,
        init_view=args.view,
        init_dpi=args.dpi,
        init_outdir=args.output_dir,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
