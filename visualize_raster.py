"""
Visualization script for tab-separated raster grid files.
Usage: python visualize_raster.py <filename> [--nodata 9999.0] [--cmap terrain]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
# Use non-interactive Agg when there's no display (headless Linux, SSH, etc.)
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def load_raster(filepath, nodata=9999.0):
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split('\t')]
            data.append(vals)
    arr = np.array(data, dtype=np.float32)
    arr[arr == nodata] = np.nan
    return np.ma.masked_invalid(arr)


def visualize(filepath, nodata=9999.0, cmap='terrain', output=None, hillshade=False):
    print(f"Loading {filepath}...")
    arr = load_raster(filepath, nodata=nodata)
    print(f"  Shape: {arr.shape[0]} rows x {arr.shape[1]} cols")

    valid = arr.compressed()
    print(f"  Valid cells: {len(valid):,} / {arr.size:,} ({100*len(valid)/arr.size:.1f}%)")
    print(f"  Value range: {valid.min():.2f} – {valid.max():.2f}")
    print(f"  Mean: {valid.mean():.2f}  Std: {valid.std():.2f}")

    # --- figure layout ---
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(Path(filepath).name, fontsize=11, y=0.98)

    if hillshade:
        # 2-panel: hillshade + colored elevation
        ax1 = fig.add_axes([0.05, 0.12, 0.40, 0.80])
        ax2 = fig.add_axes([0.52, 0.12, 0.40, 0.80])
        cax = fig.add_axes([0.93, 0.12, 0.015, 0.80])
        _plot_hillshade(ax1, arr)
        im = _plot_elevation(ax2, arr, cmap)
        ax1.set_title("Hillshade", fontsize=9)
        ax2.set_title("Elevation", fontsize=9)
    else:
        ax2 = fig.add_axes([0.05, 0.12, 0.86, 0.80])
        cax = fig.add_axes([0.93, 0.12, 0.015, 0.80])
        im = _plot_elevation(ax2, arr, cmap)
        ax2.set_title("Elevation", fontsize=9)

    plt.colorbar(im, cax=cax, label="Value")

    # Stats box
    stats = (
        f"min  {valid.min():.1f}\n"
        f"max  {valid.max():.1f}\n"
        f"mean {valid.mean():.1f}\n"
        f"std  {valid.std():.1f}\n"
        f"NaN  {arr.mask.sum():,}"
    )
    fig.text(0.005, 0.50, stats, va='center', fontsize=8,
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    if not output and matplotlib.get_backend().lower() == 'agg':
        output = Path(filepath).stem + '.png'
        print("No display detected — saving to file instead.")

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved to {output}")
    else:
        plt.show()


def _plot_elevation(ax, arr, cmap):
    vmin, vmax = arr.min(), arr.max()
    # Use diverging colormap if data spans negative values
    if vmin < 0 and cmap == 'terrain':
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
    ax.set_xlabel("Column", fontsize=8)
    ax.set_ylabel("Row", fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def _plot_hillshade(ax, arr):
    from matplotlib.colors import LightSource
    data = arr.filled(np.nan)
    # fill NaN with mean so shading doesn't break at borders
    mean_val = np.nanmean(data)
    data = np.where(np.isnan(data), mean_val, data)
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(data, vert_exag=0.05)
    mask = arr.mask if np.ma.is_masked(arr) else False
    shade_masked = np.ma.array(shade, mask=mask)
    ax.imshow(shade_masked, cmap='gray', interpolation='nearest', aspect='auto')
    ax.set_xlabel("Column", fontsize=8)
    ax.set_ylabel("Row", fontsize=8)
    ax.tick_params(labelsize=7)


def main():
    parser = argparse.ArgumentParser(description="Visualize tab-separated raster grid")
    parser.add_argument("filepath", help="Path to the .txt raster file")
    parser.add_argument("--nodata", type=float, default=9999.0,
                        help="Nodata sentinel value (default: 9999.0)")
    parser.add_argument("--cmap", default="terrain",
                        help="Matplotlib colormap (default: terrain)")
    parser.add_argument("--hillshade", action="store_true",
                        help="Add hillshade panel alongside elevation")
    parser.add_argument("--output", "-o", default=None,
                        help="Save figure to this path instead of showing it")
    args = parser.parse_args()

    visualize(
        filepath=args.filepath,
        nodata=args.nodata,
        cmap=args.cmap,
        output=args.output,
        hillshade=args.hillshade,
    )


if __name__ == "__main__":
    main()
