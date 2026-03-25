"""
visualize_mesh.py
=================
Generate images from:
  1. An ANSYS APDL .inp file mesh (nodes + edges, colored by part)
  2. A generated .h5 dataset file (B8_REV04_SEC_BOT.h5) with mesh
     colored by z-displacement (feature index 5 per DATASET_FORMAT.md)

Files are chosen via a PyQt5 GUI file-picker dialog.
CLI flags (--inp, --h5) can pre-fill the paths.

Usage
-----
python visualize_mesh.py                          # GUI for both files
python visualize_mesh.py --inp mesh.inp           # pre-fill INP path
python visualize_mesh.py --inp mesh.inp --h5 x.h5 # pre-fill both

Dependencies: numpy, h5py, matplotlib, PyQt5
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox,
)


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
# GUI — worker thread
# ---------------------------------------------------------------------------

class Worker(QThread):
    status  = pyqtSignal(str)
    done    = pyqtSignal(list)   # list of saved paths
    error   = pyqtSignal(str)

    def __init__(self, inp_path, h5_path, sample_id, view, dpi, out_dir):
        super().__init__()
        self.inp_path  = inp_path
        self.h5_path   = h5_path
        self.sample_id = sample_id
        self.view      = view
        self.dpi       = dpi
        self.out_dir   = out_dir

    def run(self):
        outputs = []
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)

            if self.inp_path:
                self.status.emit(f"Parsing {self.inp_path.name}…")
                coords, edges, part_ids = parse_inp_for_viz(self.inp_path)
                self.status.emit("Rendering INP mesh…")
                fig = plot_inp_mesh(coords, edges, part_ids,
                                    view=self.view, dpi=self.dpi)
                out = self.out_dir / f"inp_mesh_{self.view}.png"
                fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
                plt.close(fig)
                outputs.append(str(out))

            if self.h5_path:
                self.status.emit(f"Loading {self.h5_path.name} sample {self.sample_id}…")
                coords, edges, z_disp, meta = load_h5_sample(self.h5_path, self.sample_id)
                self.status.emit("Rendering H5 z-disp mesh…")
                fig = plot_h5_zdisp(coords, edges, z_disp, meta,
                                    view=self.view, dpi=self.dpi)
                out = self.out_dir / f"{self.h5_path.stem}_zdisp_{self.view}.png"
                fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
                plt.close(fig)
                outputs.append(str(out))

            self.done.emit(outputs)

        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# GUI — main window
# ---------------------------------------------------------------------------

class App(QWidget):
    def __init__(self, init_inp: Path | None, init_h5: Path | None,
                 init_sample: int, init_view: str,
                 init_dpi: int, init_outdir: Path):
        super().__init__()
        self.setWindowTitle("Mesh Visualizer")
        self.setMinimumWidth(620)
        self._worker = None

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # ── INP file ──────────────────────────────────────────────────────
        inp_box = QGroupBox("INP mesh file")
        inp_lay = QHBoxLayout(inp_box)
        self._inp_edit = QLineEdit(str(init_inp) if init_inp else "")
        self._inp_edit.setPlaceholderText("(none selected)")
        inp_lay.addWidget(self._inp_edit)
        btn_inp = QPushButton("Browse…")
        btn_inp.clicked.connect(self._browse_inp)
        inp_lay.addWidget(btn_inp)
        btn_inp_clr = QPushButton("Clear")
        btn_inp_clr.clicked.connect(lambda: self._inp_edit.clear())
        inp_lay.addWidget(btn_inp_clr)
        root.addWidget(inp_box)

        # ── H5 file ───────────────────────────────────────────────────────
        h5_box = QGroupBox("H5 dataset file")
        h5_lay = QHBoxLayout(h5_box)
        self._h5_edit = QLineEdit(str(init_h5) if init_h5 else "")
        self._h5_edit.setPlaceholderText("(none selected)")
        h5_lay.addWidget(self._h5_edit)
        btn_h5 = QPushButton("Browse…")
        btn_h5.clicked.connect(self._browse_h5)
        h5_lay.addWidget(btn_h5)
        btn_h5_clr = QPushButton("Clear")
        btn_h5_clr.clicked.connect(lambda: self._h5_edit.clear())
        h5_lay.addWidget(btn_h5_clr)
        root.addWidget(h5_box)

        # ── Options ───────────────────────────────────────────────────────
        opt_box = QGroupBox("Options")
        opt_lay = QHBoxLayout(opt_box)

        opt_lay.addWidget(QLabel("Sample ID:"))
        self._sample_spin = QSpinBox()
        self._sample_spin.setRange(1, 99999)
        self._sample_spin.setValue(init_sample)
        opt_lay.addWidget(self._sample_spin)

        opt_lay.addSpacing(16)
        opt_lay.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["xy", "xz", "yz"])
        self._view_combo.setCurrentText(init_view)
        opt_lay.addWidget(self._view_combo)

        opt_lay.addSpacing(16)
        opt_lay.addWidget(QLabel("DPI:"))
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(72, 600)
        self._dpi_spin.setSingleStep(25)
        self._dpi_spin.setValue(init_dpi)
        opt_lay.addWidget(self._dpi_spin)

        opt_lay.addStretch()
        root.addWidget(opt_box)

        # ── Output folder ─────────────────────────────────────────────────
        out_box = QGroupBox("Output folder")
        out_lay = QHBoxLayout(out_box)
        self._outdir_edit = QLineEdit(str(init_outdir))
        out_lay.addWidget(self._outdir_edit)
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_outdir)
        out_lay.addWidget(btn_out)
        root.addWidget(out_box)

        # ── Progress + status ─────────────────────────────────────────────
        self._pbar = QProgressBar()
        self._pbar.setRange(0, 0)   # indeterminate
        self._pbar.setVisible(False)
        root.addWidget(self._pbar)

        self._status_lbl = QLabel("Ready.")
        self._status_lbl.setAlignment(Qt.AlignLeft)
        root.addWidget(self._status_lbl)

        # ── Generate button ───────────────────────────────────────────────
        self._gen_btn = QPushButton("Generate images")
        self._gen_btn.setStyleSheet(
            "QPushButton { background:#1a73e8; color:white; font-weight:bold;"
            " font-size:13px; padding:8px 20px; border-radius:4px; }"
            "QPushButton:disabled { background:#aaa; }"
        )
        self._gen_btn.clicked.connect(self._on_generate)
        root.addWidget(self._gen_btn, alignment=Qt.AlignHCenter)

    # ── Browse dialogs ────────────────────────────────────────────────────

    def _browse_inp(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ANSYS APDL .inp file", "",
            "APDL input files (*.inp *.cdb *.dat);;All files (*)"
        )
        if path:
            self._inp_edit.setText(path)

    def _browse_h5(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select HDF5 dataset file", "",
            "HDF5 files (*.h5 *.hdf5);;All files (*)"
        )
        if path:
            self._h5_edit.setText(path)

    def _browse_outdir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._outdir_edit.setText(path)

    # ── Generate ──────────────────────────────────────────────────────────

    def _on_generate(self):
        inp_str = self._inp_edit.text().strip()
        h5_str  = self._h5_edit.text().strip()

        if not inp_str and not h5_str:
            QMessageBox.warning(self, "No files selected",
                                "Please select at least one file (INP or H5).")
            return

        inp_path = Path(inp_str) if inp_str else None
        h5_path  = Path(h5_str)  if h5_str  else None
        out_dir  = Path(self._outdir_edit.text().strip() or "figures")

        if inp_path and not inp_path.is_file():
            QMessageBox.critical(self, "File not found", f"INP not found:\n{inp_path}")
            return
        if h5_path and not h5_path.is_file():
            QMessageBox.critical(self, "File not found", f"H5 not found:\n{h5_path}")
            return

        self._gen_btn.setEnabled(False)
        self._pbar.setVisible(True)
        self._status_lbl.setText("Starting…")

        self._worker = Worker(
            inp_path=inp_path,
            h5_path=h5_path,
            sample_id=self._sample_spin.value(),
            view=self._view_combo.currentText(),
            dpi=self._dpi_spin.value(),
            out_dir=out_dir,
        )
        self._worker.status.connect(self._status_lbl.setText)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, outputs: list):
        self._pbar.setVisible(False)
        self._gen_btn.setEnabled(True)
        self._status_lbl.setText("Done.")
        QMessageBox.information(self, "Done",
                                "Saved:\n" + "\n".join(outputs))

    def _on_error(self, tb: str):
        self._pbar.setVisible(False)
        self._gen_btn.setEnabled(True)
        self._status_lbl.setText("Error — see details.")
        QMessageBox.critical(self, "Error", tb)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize INP mesh and H5 dataset sample (PyQt5 GUI).")
    parser.add_argument("--inp",        type=Path, default=None,
                        help="Pre-fill INP path")
    parser.add_argument("--h5",         type=Path, default=None,
                        help="Pre-fill H5 path")
    parser.add_argument("--sample-id",  type=int,  default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi",        type=int,  default=150)
    parser.add_argument("--view",       choices=["xy", "xz", "yz"], default="xy")
    args = parser.parse_args()

    qapp = QApplication(sys.argv)
    win = App(
        init_inp=args.inp,
        init_h5=args.h5,
        init_sample=args.sample_id,
        init_view=args.view,
        init_dpi=args.dpi,
        init_outdir=args.output_dir,
    )
    win.show()
    sys.exit(qapp.exec_())


if __name__ == "__main__":
    main()
