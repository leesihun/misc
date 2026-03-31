"""
warpage_to_hdf5.py
==================
Convert ANSYS APDL .inp mesh + warpage .txt raster files → HDF5 dataset
following DATASET_FORMAT.md.

Usage
-----
python warpage_to_hdf5.py \
    --inp mesh.inp \
    --warpage-dir ./txt_files/ \
    --output dataset.h5 \
    [--unit-scale 1.0] \
    [--flip-y | --no-flip-y] \
    [--void-threshold 9998.0] \
    [--train-ratio 0.7] \
    [--val-ratio 0.15] \
    [--seed 42] \
    [--workers 8]

Dependencies: numpy, scipy, h5py, tqdm (optional)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _progress(iterable, total: int, desc: str = ""):
    """Wrap iterable with tqdm if available, otherwise plain iteration."""
    if _HAS_TQDM:
        return _tqdm(iterable, total=total, desc=desc)
    return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    inp_path: Path
    warpage_dir: Path
    output_path: Path
    unit_scale: float = 1.0
    flip_y: bool = True
    void_threshold: float = 9998.0
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    num_workers: int = field(default_factory=lambda: os.cpu_count() or 1)

    FEATURE_NAMES: tuple = (
        "x_coord", "y_coord", "z_coord",
        "x_disp(mm)", "y_disp(mm)", "z_disp(mm)",
        "stress(MPa)", "Part No.",
    )
    NUM_FEATURES: int = 8
    NUM_TIMESTEPS: int = 1


# ---------------------------------------------------------------------------
# ANSYS APDL element face tables
# ---------------------------------------------------------------------------

# Maps ANSYS element name → face definitions as tuples of LOCAL corner indices.
# For quadratic elements only the first N corner nodes are used.
ELEMENT_FACES: dict[str, list[tuple[int, ...]]] = {
    # 4-node tet (SOLID72, SOLID285)
    "SOLID72":  [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)],
    "SOLID285": [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)],
    # 10-node tet — corners 0-3 (SOLID87, SOLID92, SOLID187)
    "SOLID87":  [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)],
    "SOLID92":  [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)],
    "SOLID187": [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)],
    # 8-node hex (SOLID45, SOLID185)
    "SOLID45":  [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (1, 2, 6, 5),
        (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    "SOLID185": [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (1, 2, 6, 5),
        (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    # 20-node hex — corners 0-7 (SOLID95, SOLID186)
    "SOLID95":  [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (1, 2, 6, 5),
        (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    "SOLID186": [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (1, 2, 6, 5),
        (2, 3, 7, 6), (3, 0, 4, 7),
    ],

    # 6-node wedge/prism (SOLID90)
    "SOLID90": [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (0, 2, 5, 3)],
    # Shell elements (always exterior)
    "SHELL41":  [(0, 1, 2, 3)],
    "SHELL43":  [(0, 1, 2, 3)],
    "SHELL63":  [(0, 1, 2, 3)],
    "SHELL93":  [(0, 1, 2, 3)],
    "SHELL181": [(0, 1, 2, 3)],
    "SHELL281": [(0, 1, 2, 3)],
}

# How many leading nodes to keep as corners per element type
CORNER_COUNT: dict[str, int] = {
    "SOLID72": 4, "SOLID285": 4,
    "SOLID87": 4, "SOLID92": 4, "SOLID187": 4,
    "SOLID45": 8, "SOLID185": 8,
    "SOLID95": 8, "SOLID186": 8,
    "SOLID90": 6,
    "SHELL41": 4, "SHELL43": 4, "SHELL63": 4,
    "SHELL93": 4, "SHELL181": 4, "SHELL281": 4,
}

# ANSYS element type number → name (for ET,itype,number syntax)
APDL_TYPE_BY_NUMBER: dict[int, str] = {
    45: "SOLID45", 72: "SOLID72", 87: "SOLID87", 90: "SOLID90",
    92: "SOLID92", 95: "SOLID95",
    185: "SOLID185", 186: "SOLID186", 187: "SOLID187", 285: "SOLID285",
    41: "SHELL41", 43: "SHELL43", 63: "SHELL63",
    93: "SHELL93", 181: "SHELL181", 281: "SHELL281",
}


def _get_face_table(etype: str) -> tuple[list[tuple[int, ...]], int]:
    """Return (face_definitions, num_corners) for an element type string."""
    etype_upper = etype.upper().strip()
    for key in ELEMENT_FACES:
        if etype_upper.startswith(key) or etype_upper == key:
            return ELEMENT_FACES[key], CORNER_COUNT[key]
    # Unknown type — treat all nodes as corners, no face info
    return [], 0


# ---------------------------------------------------------------------------
# Section 1: ANSYS APDL Parser
# ---------------------------------------------------------------------------

@dataclass
class AbaqusMesh:
    """Parsed mesh data (node/element topology)."""
    # global_node_id -> [x, y, z]
    nodes: dict[int, np.ndarray] = field(default_factory=dict)
    # global_element_id -> list of global node ids (corner nodes only)
    elements: dict[int, list[int]] = field(default_factory=dict)
    # global_element_id -> element type string
    elem_types: dict[int, str] = field(default_factory=dict)
    # global_node_id -> part_index (0-based)
    node_part: dict[int, int] = field(default_factory=dict)
    # ordered list of part names seen
    part_names: list[str] = field(default_factory=list)


def parse_apdl(inp_path: Path) -> AbaqusMesh:
    """
    Parse an ANSYS APDL (.inp / .cdb / .dat) file.

    Handles:
    - ET,itype,ename: element type definitions
    - TYPE,itype: set current element type
    - NBLOCK: bulk node definitions (fixed-width FORTRAN format)
    - EBLOCK: bulk element definitions
    - N,nid,x,y,z: individual node commands
    - E,n1,...: individual element commands (auto-incremented ID)
    - EN,eid,n1,...: individual element commands with explicit ID
    """
    mesh = AbaqusMesh()

    # ET command map: type_number → element name string
    et_map: dict[int, str] = {}
    current_type_num: int = 1
    auto_eid: int = 0  # counter for E, command (no explicit ID)

    def _resolve_etype(type_num: int) -> str:
        return et_map.get(type_num, "SOLID185").upper()

    STATE_IDLE    = "IDLE"
    STATE_NBLOCK  = "NBLOCK"
    STATE_EBLOCK  = "EBLOCK"
    STATE_CMBLOCK = "CMBLOCK"

    state = STATE_IDLE
    nblock_n_int  = 3   # number of leading integer fields per NBLOCK data line
    nblock_i_width = 8  # character width of each integer field
    nblock_f_width = 16 # character width of each float field

    # EBLOCK per-element accumulation
    eblock_pending_eid:  int       = -1
    eblock_pending_type: int       = 1
    eblock_need_nodes:   int       = 0
    eblock_nodes_so_far: list[int] = []
    eblock_i_width:      int       = 8  # integer field width from format line

    face_tables: dict[int, list[tuple[int, ...]]] = {}

    def commit_element(eid: int, type_num: int, node_ids: list[int]) -> None:
        ename = _resolve_etype(type_num)
        faces, n_corners = _get_face_table(ename)
        if n_corners > 0:
            node_ids = node_ids[:n_corners]
        mesh.elements[eid] = node_ids
        mesh.elem_types[eid] = ename
        if faces:
            face_tables[eid] = faces
        for nid in node_ids:
            if nid not in mesh.node_part:
                mesh.node_part[nid] = 0

    with open(inp_path, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            # Strip trailing newline only; preserve leading spaces for detection
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                continue

            # Strip inline APDL comment character
            content = stripped.split("!")[0].strip()
            if not content:
                continue

            # ----------------------------------------------------------------
            # NBLOCK data mode
            # ----------------------------------------------------------------
            if state == STATE_NBLOCK:
                if content.startswith("-1"):
                    state = STATE_IDLE
                    continue
                # Format line: e.g. (3i8,6e16.9) or (3i9,6e21.13e3)
                # Parse integer field count and widths for fixed-width slicing.
                if content.startswith("("):
                    m = re.match(r"\(\s*(\d+)\s*[iI]\s*(\d+)", content)
                    if m:
                        nblock_n_int  = int(m.group(1))
                        nblock_i_width = int(m.group(2))
                    fm = re.search(r"[eEfFdD]\s*(\d+)", content)
                    if fm:
                        nblock_f_width = int(fm.group(1))
                    continue
                # Use fixed-width slicing so adjacent fields (no space) parse correctly.
                try:
                    int_end = nblock_n_int * nblock_i_width
                    nid = int(raw_line[:nblock_i_width])
                    float_start = int_end
                    x = float(raw_line[float_start                : float_start +     nblock_f_width])
                    y = float(raw_line[float_start +   nblock_f_width : float_start + 2*nblock_f_width])
                    z = float(raw_line[float_start + 2*nblock_f_width : float_start + 3*nblock_f_width])
                except (ValueError, IndexError):
                    continue
                mesh.nodes[nid] = np.array([x, y, z], dtype=np.float64)
                if nid not in mesh.node_part:
                    mesh.node_part[nid] = 0
                continue

            # ----------------------------------------------------------------
            # EBLOCK data mode
            # ----------------------------------------------------------------
            if state == STATE_EBLOCK:
                if content.startswith("-1"):
                    state = STATE_IDLE
                    continue
                # Format line: (19i8) — capture integer field width
                if content.startswith("("):
                    m = re.match(r"\(\s*\d+\s*[iI]\s*(\d+)", content)
                    if m:
                        eblock_i_width = int(m.group(1))
                    continue
                # Fixed-width parse: split raw_line into eblock_i_width chunks
                try:
                    line_stripped = raw_line.rstrip("\n\r")
                    tokens = []
                    pos = 0
                    while pos + eblock_i_width <= len(line_stripped):
                        field = line_stripped[pos:pos + eblock_i_width].strip()
                        if field:
                            tokens.append(int(field))
                        pos += eblock_i_width
                except ValueError:
                    continue

                if eblock_need_nodes == 0:
                    # First line of a new element record
                    # Layout: mat type real secn esys death solidm shape nodes user eid n1 n2 ...
                    if len(tokens) < 11:
                        continue
                    type_num  = tokens[1]
                    n_nodes   = tokens[8]
                    eid       = tokens[10]
                    node_ids  = tokens[11:]
                    eblock_pending_eid  = eid
                    eblock_pending_type = type_num
                    eblock_nodes_so_far = node_ids
                    eblock_need_nodes   = n_nodes
                    if len(eblock_nodes_so_far) >= eblock_need_nodes:
                        commit_element(eblock_pending_eid, eblock_pending_type,
                                       eblock_nodes_so_far[:eblock_need_nodes])
                        eblock_need_nodes = 0
                else:
                    # Continuation line with remaining node IDs
                    eblock_nodes_so_far.extend(tokens)
                    if len(eblock_nodes_so_far) >= eblock_need_nodes:
                        commit_element(eblock_pending_eid, eblock_pending_type,
                                       eblock_nodes_so_far[:eblock_need_nodes])
                        eblock_need_nodes = 0
                continue

            # ----------------------------------------------------------------
            # CMBLOCK — skip until terminator
            # ----------------------------------------------------------------
            if state == STATE_CMBLOCK:
                if content.startswith("-1"):
                    state = STATE_IDLE
                continue

            # ----------------------------------------------------------------
            # Command parsing (IDLE state)
            # ----------------------------------------------------------------
            # Split on comma; first token is the command
            parts = [p.strip() for p in content.split(",")]
            cmd   = parts[0].upper()
            args  = parts[1:]

            # Skip section markers, macro lines, and pure comments
            if not cmd or cmd.startswith("/") or cmd.startswith("*"):
                continue

            if cmd == "ET":
                # ET,itype,ename_or_number[,kop...]
                if len(args) >= 2:
                    try:
                        itype    = int(args[0])
                        raw_name = args[1].strip().upper()
                        try:
                            # Numeric form: ET,1,185
                            et_map[itype] = APDL_TYPE_BY_NUMBER.get(
                                int(raw_name), f"TYPE{raw_name}"
                            )
                        except ValueError:
                            # Named form: ET,1,SOLID185
                            et_map[itype] = raw_name
                    except (ValueError, IndexError):
                        pass

            elif cmd == "TYPE":
                if args:
                    try:
                        current_type_num = int(args[0])
                    except ValueError:
                        pass

            elif cmd == "NBLOCK":
                state        = STATE_NBLOCK
                nblock_n_int = 3  # reset; format line will update

            elif cmd == "EBLOCK":
                state               = STATE_EBLOCK
                eblock_need_nodes   = 0
                eblock_nodes_so_far = []

            elif cmd == "CMBLOCK":
                state = STATE_CMBLOCK

            elif cmd == "N":
                # N,nid,x,y,z[,rx,ry,rz]
                if len(args) >= 4:
                    try:
                        nid = int(args[0])
                        x, y, z = float(args[1]), float(args[2]), float(args[3])
                        mesh.nodes[nid] = np.array([x, y, z], dtype=np.float64)
                        if nid not in mesh.node_part:
                            mesh.node_part[nid] = 0
                    except (ValueError, IndexError):
                        pass

            elif cmd in ("E", "EN"):
                # EN,eid,n1,n2,...  or  E,n1,n2,... (auto EID)
                try:
                    if cmd == "EN":
                        eid      = int(args[0])
                        node_ids = [int(a) for a in args[1:] if a]
                    else:
                        auto_eid += 1
                        eid      = auto_eid
                        node_ids = [int(a) for a in args if a]
                    commit_element(eid, current_type_num, node_ids)
                except (ValueError, IndexError):
                    pass

    if not mesh.part_names:
        mesh.part_names.append("Part-0")

    mesh._face_tables = face_tables  # type: ignore[attr-defined]
    log.info(
        "Parsed %d nodes, %d elements from %s",
        len(mesh.nodes), len(mesh.elements), inp_path.name,
    )
    return mesh


# ---------------------------------------------------------------------------
# Section 2: Mesh Topology (exterior faces + edge extraction)
# ---------------------------------------------------------------------------

def _triangulate_quad(quad: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    """Fan-triangulate a quad face."""
    a, b, c, d = quad
    return [(a, b, c), (a, c, d)]


def extract_surface_edges(mesh: AbaqusMesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract:
      - corner_node_ids: sorted array of global node IDs referenced in exterior faces
      - mesh_edge: (2, E) int64 array of compact-indexed edges

    Exterior face detection: a face (as a sorted node tuple) that appears
    exactly once across all elements is a boundary face.
    Shell elements are always exterior.
    """
    face_tables: dict[int, list[tuple[int, ...]]] = getattr(
        mesh, "_face_tables", {}
    )

    # Count face occurrences using canonical (sorted) tuple as key
    face_count: dict[tuple[int, ...], int] = defaultdict(int)

    for eid, local_faces in face_tables.items():
        global_nodes = mesh.elements[eid]
        etype = mesh.elem_types[eid]
        is_shell = etype.upper().startswith("SHELL")

        for local_face in local_faces:
            global_face = tuple(sorted(global_nodes[i] for i in local_face
                                       if i < len(global_nodes)))
            if is_shell:
                # Shell faces are always exterior; -1 sentinel is never dropped.
                face_count[global_face] = -1
            else:
                # Don't overwrite a shell-sentinel with a solid face count.
                if face_count.get(global_face, 0) != -1:
                    face_count[global_face] += 1

    # Exterior = count == 1, or -1 (shell)
    exterior_faces: list[tuple[int, ...]] = [
        face for face, cnt in face_count.items() if cnt == 1 or cnt == -1
    ]

    if not exterior_faces:
        log.warning(
            "No exterior faces detected — falling back to ALL element faces. "
            "Check element types in the .inp file."
        )
        exterior_faces = list(face_count.keys())

    # Triangulate all faces
    tri_faces: list[tuple[int, int, int]] = []
    for face in exterior_faces:
        if len(face) == 3:
            tri_faces.append(face)  # type: ignore[arg-type]
        elif len(face) == 4:
            tri_faces.extend(_triangulate_quad(face))  # type: ignore[arg-type]
        # Ignore degenerate faces

    # Collect unique corner nodes referenced in triangular faces
    corner_node_ids = np.array(sorted({n for tri in tri_faces for n in tri}),
                               dtype=np.int64)

    # Build compact index map: global_id -> compact 0-based index
    compact_idx = {gid: i for i, gid in enumerate(corner_node_ids)}

    # Extract unique edges (compact indices)
    edge_set: set[tuple[int, int]] = set()
    for tri in tri_faces:
        a, b, c = (compact_idx[n] for n in tri)
        edge_set.add((min(a, b), max(a, b)))
        edge_set.add((min(b, c), max(b, c)))
        edge_set.add((min(a, c), max(a, c)))

    if not edge_set:
        mesh_edge = np.zeros((2, 0), dtype=np.int64)
    else:
        edges_arr = np.array(sorted(edge_set), dtype=np.int64)
        non_self_mask = edges_arr[:, 0] != edges_arr[:, 1]
        edges_arr = edges_arr[non_self_mask]
        mesh_edge = edges_arr.T  # (2, E)

    log.info(
        "Surface topology: %d corner nodes, %d unique edges from %d exterior faces",
        len(corner_node_ids), mesh_edge.shape[1], len(exterior_faces),
    )
    return corner_node_ids, mesh_edge


def get_bounding_box(
    mesh: AbaqusMesh,
) -> tuple[float, float, float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max, z_min, z_max)."""
    if not mesh.nodes:
        raise ValueError("Mesh has no nodes.")
    coords = np.array(list(mesh.nodes.values()))
    return (
        float(coords[:, 0].min()), float(coords[:, 0].max()),
        float(coords[:, 1].min()), float(coords[:, 1].max()),
        float(coords[:, 2].min()), float(coords[:, 2].max()),
    )


def compute_part_nos(
    node_coords: np.ndarray,  # (N, 3)
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    tol_frac: float = 1e-8,
) -> np.ndarray:
    """
    Assign Part No. based on XY bounding box boundary.

    Part No. = 1  if the node lies on any of the four outer edges of the
                   bounding box (x ≈ x_min, x ≈ x_max, y ≈ y_min, y ≈ y_max).
    Part No. = 0  everywhere else.

    tol_frac: fraction of the bounding box span used as snap tolerance.
    """
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    tol_x = tol_frac * x_span
    tol_y = tol_frac * y_span

    nx = node_coords[:, 0]
    ny = node_coords[:, 1]

    on_boundary = (
        (nx <= x_min + tol_x) |
        (nx >= x_max - tol_x) |
        (ny <= y_min + tol_y) |
        (ny >= y_max - tol_y)
    )

    part_nos = on_boundary.astype(np.float32)  # 1.0 on boundary, 0.0 elsewhere
    log.info(
        "Part No.: %d boundary nodes (1), %d interior nodes (0)",
        int(on_boundary.sum()), int((~on_boundary).sum()),
    )
    return part_nos


# ---------------------------------------------------------------------------
# Section 3: Warpage .txt Reader
# ---------------------------------------------------------------------------

def read_warpage_grid(
    txt_path: Path,
    void_threshold: float = 9998.0,
    unit_scale: float = 1.0,
) -> np.ndarray:
    """
    Load a tab-separated warpage grid.

    Returns float32 ndarray shape (R, C).
    Values > void_threshold are replaced with np.nan.
    Valid values are multiplied by unit_scale.
    """
    grid = np.loadtxt(str(txt_path), delimiter="\t", dtype=np.float32)
    if grid.ndim == 1:
        grid = grid.reshape(1, -1)

    void_mask = grid > void_threshold
    grid[void_mask] = np.nan
    if unit_scale != 1.0:
        valid = ~void_mask
        grid[valid] *= unit_scale

    log.debug(
        "%s: grid shape %s, %.1f%% void",
        txt_path.name, grid.shape,
        100.0 * void_mask.sum() / grid.size,
    )
    return grid


# ---------------------------------------------------------------------------
# Section 4: Interpolator
# ---------------------------------------------------------------------------

def _fill_nan_nearest(grid: np.ndarray) -> np.ndarray:
    """
    Replace NaN cells with the value of the nearest non-NaN cell.
    Uses scipy.ndimage.distance_transform_edt — O(R*C), much faster
    than building a KDTree over all valid points.

    Returns a copy; the original grid is not modified.
    """
    nan_mask = np.isnan(grid)
    if not nan_mask.any():
        return grid
    if nan_mask.all():
        return np.zeros_like(grid)
    # distance_transform_edt with return_indices gives, for every True cell
    # in the mask, the row/col of the nearest False cell.
    _, indices = distance_transform_edt(nan_mask, return_distances=True,
                                        return_indices=True)
    filled = grid[tuple(indices)]
    return filled


def interpolate_warpage_to_nodes(
    grid: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    node_xy: np.ndarray,  # (N, 2) — [x, y] per node
    flip_y: bool = True,
) -> np.ndarray:
    """
    Map a 2-D warpage raster to node positions via bilinear interpolation.

    Returns float32 array (N,) of interpolated z_disp values.

    Strategy:
      1. Fill NaN (void) gaps in the grid with nearest-valid-cell values
         using distance_transform_edt (O(R*C)).
      2. Single-pass RegularGridInterpolator on the gap-filled grid.

    AXIS ORDER: RegularGridInterpolator expects (y, x) because the grid
    has shape (rows, cols) = (y_axis, x_axis).
    """
    R, C = grid.shape

    # Fill void gaps before interpolation — avoids slow KDTree fallback
    grid_filled = _fill_nan_nearest(grid)

    # Build physical coordinate arrays for each axis
    x_coords = np.linspace(x_min, x_max, C)
    if flip_y:
        y_coords = np.linspace(y_max, y_min, R)  # row 0 = y_max
    else:
        y_coords = np.linspace(y_min, y_max, R)  # row 0 = y_min

    # Clamp node coordinates to grid extent to avoid edge issues
    node_x = np.clip(node_xy[:, 0], x_min, x_max)
    node_y = np.clip(node_xy[:, 1], y_min, y_max)

    # Sort y_coords for RegularGridInterpolator (must be monotonically increasing)
    if y_coords[0] > y_coords[-1]:
        y_coords_sorted = y_coords[::-1]
        grid_sorted = grid_filled[::-1, :]
    else:
        y_coords_sorted = y_coords
        grid_sorted = grid_filled

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interp = RegularGridInterpolator(
            (y_coords_sorted, x_coords),
            grid_sorted,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
    query_pts = np.column_stack([node_y, node_x])  # (N, 2) in (y, x) order
    z_disp = interp(query_pts).astype(np.float32)

    # Safety: if any node still ended up NaN (should not happen with filled grid)
    nan_count = np.isnan(z_disp).sum()
    if nan_count > 0:
        log.warning("  %d nodes still NaN after interpolation — set to 0.0", nan_count)
        z_disp = np.nan_to_num(z_disp, nan=0.0)

    return z_disp


# ---------------------------------------------------------------------------
# Section 5a: Nodal data assembly (used inside workers)
# ---------------------------------------------------------------------------

def build_nodal_data(
    node_coords: np.ndarray,   # (N, 3) — x, y, z per corner node
    z_disp: np.ndarray,        # (N,)
    part_nos: np.ndarray,      # (N,) float
) -> np.ndarray:
    """
    Assemble (8, 1, N) float32 nodal data tensor.

    Feature layout:
      0: x_coord
      1: y_coord
      2: z_coord
      3: x_disp = 0
      4: y_disp = 0
      5: z_disp (interpolated)
      6: stress = 0
      7: part_no
    """
    N = node_coords.shape[0]
    data = np.zeros((8, 1, N), dtype=np.float32)
    data[0, 0, :] = node_coords[:, 0]
    data[1, 0, :] = node_coords[:, 1]
    data[2, 0, :] = node_coords[:, 2]
    # [3] x_disp = 0 (already zero)
    # [4] y_disp = 0
    data[5, 0, :] = z_disp
    # [6] stress = 0
    data[7, 0, :] = part_nos
    return data


# ---------------------------------------------------------------------------
# Section 5b: Worker function (top-level for pickling)
# ---------------------------------------------------------------------------

# Shared data set once per worker process via initializer — avoids pickling
# the large node_coords/part_nos arrays with every single task.
_shared: dict = {}


def _init_worker(
    node_coords: np.ndarray,
    part_nos: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    void_threshold: float, unit_scale: float, flip_y: bool,
) -> None:
    """Called once per worker process to store shared data."""
    _shared["node_coords"] = node_coords
    _shared["part_nos"] = part_nos
    _shared["bbox"] = (x_min, x_max, y_min, y_max)
    _shared["void_threshold"] = void_threshold
    _shared["unit_scale"] = unit_scale
    _shared["flip_y"] = flip_y


def _worker(txt_path_str: str) -> tuple[np.ndarray, str]:
    """
    Process one warpage .txt file and return (nodal_data, filename_id).
    Runs in a subprocess — reads shared data from _shared dict.
    """
    node_coords = _shared["node_coords"]
    part_nos = _shared["part_nos"]
    x_min, x_max, y_min, y_max = _shared["bbox"]
    void_threshold = _shared["void_threshold"]
    unit_scale = _shared["unit_scale"]
    flip_y = _shared["flip_y"]

    txt_path = Path(txt_path_str)

    try:
        grid = read_warpage_grid(txt_path, void_threshold, unit_scale)
    except Exception as exc:
        log.error("Failed to read %s: %s", txt_path.name, exc)
        z_disp = np.zeros(node_coords.shape[0], dtype=np.float32)
    else:
        z_disp = interpolate_warpage_to_nodes(
            grid, x_min, x_max, y_min, y_max,
            node_coords[:, :2], flip_y,
        )

    nodal_data = build_nodal_data(node_coords, z_disp, part_nos)
    return nodal_data, txt_path.stem


# ---------------------------------------------------------------------------
# Section 6: HDF5 Writer
# ---------------------------------------------------------------------------

def write_sample(
    h5file: h5py.File,
    sample_idx: int,
    nodal_data: np.ndarray,   # (8, 1, N)
    mesh_edge: np.ndarray,    # (2, E)
    filename_id: str,
) -> None:
    grp = h5file.create_group(f"data/{sample_idx}")
    grp.create_dataset("nodal_data", data=nodal_data, dtype="float32")
    grp.create_dataset("mesh_edge", data=mesh_edge, dtype="int64")
    meta = grp.create_group("metadata")
    meta.attrs["filename_id"] = filename_id
    meta.attrs["num_nodes"] = nodal_data.shape[2]
    meta.attrs["num_edges"] = mesh_edge.shape[1]
    meta.attrs["num_corner_nodes"] = nodal_data.shape[2]


def compute_normalization_stats(
    h5file: h5py.File,
    num_samples: int,
    num_features: int,
) -> dict[str, np.ndarray]:
    """
    Streaming Welford pass over all samples for per-feature mean/std.
    Also tracks global min/max. Never loads all data into RAM simultaneously.
    """
    gmin = np.full(num_features, np.inf, dtype=np.float64)
    gmax = np.full(num_features, -np.inf, dtype=np.float64)
    count_f = np.zeros(num_features, dtype=np.int64)
    mean_f = np.zeros(num_features, dtype=np.float64)
    M2_f = np.zeros(num_features, dtype=np.float64)

    for i in range(1, num_samples + 1):
        nd = h5file[f"data/{i}/nodal_data"][:, 0, :].astype(np.float64)  # (F, N)
        gmin = np.minimum(gmin, nd.min(axis=1))
        gmax = np.maximum(gmax, nd.max(axis=1))
        # Welford batch update per feature
        N = nd.shape[1]
        for f in range(num_features):
            fdata = nd[f]
            new_count = count_f[f] + N
            new_mean = mean_f[f] + (fdata.sum() - N * mean_f[f]) / new_count
            M2_f[f] += np.sum((fdata - mean_f[f]) * (fdata - new_mean))
            mean_f[f] = new_mean
            count_f[f] = new_count

    std_f = np.where(count_f > 1, np.sqrt(M2_f / (count_f - 1)), 0.0)

    # Zero-std guard: constant features (x_disp, y_disp, stress) → set std=1
    std_f = np.where(std_f < 1e-12, 1.0, std_f)

    return {
        "min": gmin.astype(np.float32),
        "max": gmax.astype(np.float32),
        "mean": mean_f.astype(np.float32),
        "std": std_f.astype(np.float32),
    }


def write_global_metadata(
    h5file: h5py.File,
    config: Config,
    num_samples: int,
    stats: dict[str, np.ndarray],
    sample_indices: list[int],
) -> None:
    h5file.attrs["num_samples"] = num_samples
    h5file.attrs["num_features"] = config.NUM_FEATURES
    h5file.attrs["num_timesteps"] = config.NUM_TIMESTEPS

    meta = h5file.create_group("metadata")

    # Feature names as variable-length byte strings
    dt = h5py.special_dtype(vlen=bytes)
    feat_ds = meta.create_dataset("feature_names", (config.NUM_FEATURES,), dtype=dt)
    for i, name in enumerate(config.FEATURE_NAMES):
        feat_ds[i] = name.encode()

    # Normalization params
    norm = meta.create_group("normalization_params")
    for key in ("min", "max", "mean", "std"):
        norm.create_dataset(key, data=stats[key], dtype="float32")

    # Train/val/test splits
    rng = np.random.default_rng(config.seed)
    ids = np.array(sample_indices, dtype=np.int64)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)

    splits = meta.create_group("splits")
    splits.create_dataset("train", data=ids[:n_train], dtype="int64")
    splits.create_dataset("val", data=ids[n_train:n_train + n_val], dtype="int64")
    splits.create_dataset("test", data=ids[n_train + n_val:], dtype="int64")

    log.info(
        "Splits: %d train / %d val / %d test",
        n_train, n_val, n - n_train - n_val,
    )


# ---------------------------------------------------------------------------
# Section 7: Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(config: Config) -> None:
    # 1. Parse mesh
    log.info("Parsing ANSYS APDL mesh: %s", config.inp_path)
    mesh = parse_apdl(config.inp_path)

    # 2. Extract surface topology
    log.info("Extracting surface topology…")
    corner_ids, mesh_edge = extract_surface_edges(mesh)

    if len(corner_ids) == 0:
        raise RuntimeError(
            "No corner nodes found. Check element types in the .inp file."
        )

    # 3. Bounding box
    x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(mesh)
    log.info(
        "Mesh bounding box: X [%.3f, %.3f], Y [%.3f, %.3f], Z [%.3f, %.3f]",
        x_min, x_max, y_min, y_max, z_min, z_max,
    )

    # 4. Build per-node arrays (corner nodes only, sorted by original ID)
    node_coords = np.array(
        [mesh.nodes[nid] for nid in corner_ids], dtype=np.float32
    )  # (N, 3)
    part_nos = compute_part_nos(node_coords, x_min, x_max, y_min, y_max)  # (N,)

    # 5. Discover warpage files
    txt_files = sorted(config.warpage_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {config.warpage_dir}"
        )
    log.info("Found %d warpage files in %s", len(txt_files), config.warpage_dir)

    # 6. Process samples in parallel
    #    Shared arrays (node_coords, part_nos, bbox, config) are sent once
    #    per worker via initializer — NOT pickled per task.
    log.info("Processing %d samples with %d workers…", len(txt_files), config.num_workers)

    txt_paths_str = [str(p) for p in txt_files]
    results: list[tuple[np.ndarray, str]] = [None] * len(txt_files)  # type: ignore

    with ProcessPoolExecutor(
        max_workers=config.num_workers,
        initializer=_init_worker,
        initargs=(
            node_coords, part_nos,
            x_min, x_max, y_min, y_max,
            config.void_threshold, config.unit_scale, config.flip_y,
        ),
    ) as pool:
        futures = {
            pool.submit(_worker, path_str): idx
            for idx, path_str in enumerate(txt_paths_str)
        }
        for future in _progress(as_completed(futures), total=len(txt_files),
                                desc="Processing samples"):
            orig_idx = futures[future]
            try:
                results[orig_idx] = future.result()
            except Exception as exc:
                log.error(
                    "Worker failed for %s: %s",
                    txt_files[orig_idx].name, exc,
                )
                # Fill with zeros so indexing stays contiguous
                results[orig_idx] = (
                    build_nodal_data(
                        node_coords,
                        np.zeros(node_coords.shape[0], dtype=np.float32),
                        part_nos,
                    ),
                    txt_files[orig_idx].stem,
                )

    # 7. Write HDF5 serially (h5py is not multiprocess-safe)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing HDF5: %s", config.output_path)
    sample_indices = list(range(1, len(txt_files) + 1))

    with h5py.File(str(config.output_path), "w") as h5f:
        for sample_idx, (nodal_data, filename_id) in enumerate(results, start=1):
            write_sample(h5f, sample_idx, nodal_data, mesh_edge, filename_id)

        # 9. Compute global normalization stats (streaming pass)
        log.info("Computing normalization statistics…")
        stats = compute_normalization_stats(h5f, len(results), config.NUM_FEATURES)
        log.info(
            "z_disp stats — min: %.4f, max: %.4f, mean: %.4f, std: %.4f",
            stats["min"][5], stats["max"][5], stats["mean"][5], stats["std"][5],
        )

        # 10. Write global metadata
        write_global_metadata(h5f, config, len(results), stats, sample_indices)

    log.info("Done. Output: %s", config.output_path)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ANSYS APDL .inp mesh + warpage .txt files → HDF5 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inp", required=True, type=Path,
                        help="Path to ANSYS APDL .inp mesh file")
    parser.add_argument("--warpage-dir", required=True, type=Path,
                        help="Directory containing warpage .txt files")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output HDF5 file path")
    parser.add_argument("--unit-scale", type=float, default=1.0,
                        help="Multiply warpage values by this factor (e.g. 0.001 for μm→mm)")
    parser.add_argument("--flip-y", action=argparse.BooleanOptionalAction, default=True,
                        help="Row 0 = y_max (image convention). Use --no-flip-y to invert.")
    parser.add_argument("--void-threshold", type=float, default=9998.0,
                        help="Values above this are treated as void (NaN)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Fraction of samples for training split")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of samples for validation split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val/test split")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes")

    args = parser.parse_args()

    if not args.inp.exists():
        parser.error(f"Mesh file not found: {args.inp}")
    if not args.warpage_dir.is_dir():
        parser.error(f"Warpage directory not found: {args.warpage_dir}")
    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("train_ratio + val_ratio must be < 1.0")

    config = Config(
        inp_path=args.inp,
        warpage_dir=args.warpage_dir,
        output_path=args.output,
        unit_scale=args.unit_scale,
        flip_y=args.flip_y,
        void_threshold=args.void_threshold,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.workers,
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
