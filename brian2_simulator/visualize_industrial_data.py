"""
visualize_industrial_data.py
─────────────────────────────
Visualization toolkit for industrial sensor data.

Expected CSV format
  • Column 0 : 'date'  — timestamps like "00:00.0", "00:01.2", …
  • Columns 1-35 : signal columns (any names)

The 35 signals are arranged in a 5×7 grid (rows × cols).
If your data has a different number of signals, pass --rows / --cols.

Plots produced
  1. grid_signals      — grid of all signal time series
  2. mean_std_band     — per-row mean ± std band
  3. correlation       — Pearson correlation matrix
  4. psd               — power spectral density per row

Usage
  python visualize_industrial_data.py --csv data.csv
  python visualize_industrial_data.py --csv data.csv --rows 5 --cols 7 --out_dir ./plots
  python visualize_industrial_data.py --csv data.csv --plots grid band
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.signal import welch


# ── Aesthetics (light theme) ───────────────────────────────────────────────────
STYLE = {
    "figure.facecolor":   "#ffffff",
    "axes.facecolor":     "#f7f8fc",
    "axes.edgecolor":     "#c0c8d8",
    "axes.labelcolor":    "#2e3650",
    "axes.titlecolor":    "#1a1f36",
    "xtick.color":        "#6b7799",
    "ytick.color":        "#6b7799",
    "grid.color":         "#dde2ef",
    "grid.linewidth":     0.5,
    "text.color":         "#2e3650",
    "font.family":        "monospace",
    "lines.linewidth":    0.6,
    "figure.dpi":         120,
}

# Color ramp across columns — same palette as brian2 visualizer
COL_COLORS = ["#4ecdc4", "#45b7d1", "#7c83e0", "#a78bfa", "#f472b6",
              "#fb923c", "#34d399"]   # extra colours for wider grids
BAND_ALPHA  = 0.18


# ── I/O helpers ────────────────────────────────────────────────────────────────

def parse_time(series: pd.Series) -> np.ndarray:
    """
    Convert the 'date' column (MM:SS.f strings) to seconds.
    Falls back to a 0-based index if parsing fails.
    """
    def _to_sec(val):
        try:
            parts = str(val).split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(val)
        except (ValueError, AttributeError):
            return np.nan

    t = series.apply(_to_sec).values
    if np.isnan(t).any():
        print("  ⚠  Could not parse all timestamps — using sample index instead.")
        t = np.arange(len(series), dtype=float)
    return t


def load_csv(csv_path: str, rows: int, cols: int):
    """
    Load CSV.  Returns
      t        : 1-D time array (seconds)
      data     : dict {(r, c): np.array}
      sig_names: list of signal column names, row-major order
    """
    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError("CSV must have a 'date' column as the first column.")

    t = parse_time(df["date"])

    sig_cols = [c for c in df.columns if c != "date"]
    n_signals = len(sig_cols)
    expected  = rows * cols

    if n_signals < expected:
        print(f"  ⚠  Found {n_signals} signals but grid is {rows}×{cols}={expected}. "
              f"Padding missing sensors with zeros.")
        for i in range(expected - n_signals):
            df[f"__pad_{i}"] = 0.0
            sig_cols.append(f"__pad_{i}")
    elif n_signals > expected:
        print(f"  ⚠  Found {n_signals} signals; using first {expected} for {rows}×{cols} grid.")
        sig_cols = sig_cols[:expected]

    data = {}
    for idx, col in enumerate(sig_cols):
        r, c = divmod(idx, cols)
        data[(r, c)] = df[col].values.astype(float)

    print(f"  Loaded {n_signals} signals × {len(t)} samples  →  {rows}×{cols} grid")
    return t, data, sig_cols[:expected]


def compute_stats(data, rows, cols):
    """Return per-sensor mean and std (trivially from the single run)."""
    mean_d = {rc: data[rc].copy() for rc in data}
    std_d  = {rc: np.zeros_like(data[rc]) for rc in data}
    return mean_d, std_d


# ── Plot 1: grid of all signals ────────────────────────────────────────────────

def plot_grid_signals(t, data, rows, cols, sig_names=None,
                      title="Industrial sensor signals — grid",
                      save_path=None):
    """Plot all signals in a rows×cols subplot grid."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 0.5 * rows),
                                 sharex=True)
        # Ensure axes is always 2-D
        if rows == 1:
            axes = axes[np.newaxis, :]
        if cols == 1:
            axes = axes[:, np.newaxis]

        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01,
                     color="#1a1f36")

        for r in range(rows):
            for c in range(cols):
                ax    = axes[r][c]
                sig   = data[(r, c)]
                color = COL_COLORS[c % len(COL_COLORS)]
                name  = sig_names[r * cols + c] if sig_names else f"({r},{c})"

                ax.plot(t, sig, color=color, linewidth=0.5, alpha=0.9)
                ax.fill_between(t, sig, alpha=0.10, color=color)

                ax.grid(True, axis="y", linewidth=0.4)
                ax.set_xlim(t[0], t[-1])

                if r == 0:
                    ax.set_title(f"col {c}", fontsize=8, color="#6b7799", pad=3)
                if c == 0:
                    ax.set_ylabel(f"row {r}", fontsize=8, color="#6b7799",
                                  rotation=0, labelpad=28, va="center")

                #ax.text(0.97, 0.92, name, transform=ax.transAxes,
                #        fontsize=6, color="#3a4570", ha="right", va="top")

                if r < rows - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("time (s)", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.yaxis.set_major_locator(MaxNLocator(3))

        fig.tight_layout()
        _save_or_show(fig, save_path)


# ── Plot 2: per-row mean ± std band ───────────────────────────────────────────

def plot_mean_std_band(t, mean_data, std_data, rows, cols,
                       title="Per-row signal overview",
                       save_path=None):
    """For each row, overlay all column signals with a std band."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, 1, figsize=(14, 2.5 * rows), sharex=True)
        if rows == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=12, fontweight="bold", color="#1a1f36")

        for r in range(rows):
            ax = axes[r]
            for c in range(cols):
                mu  = mean_data[(r, c)]
                sig = std_data[(r, c)]
                col = COL_COLORS[c % len(COL_COLORS)]

                ax.fill_between(t, mu - sig, mu + sig,
                                alpha=BAND_ALPHA, color=col)
                ax.plot(t, mu, color=col, linewidth=0.6,
                        label=f"({r},{c})", alpha=0.9)

            ax.set_ylabel(f"row {r}", fontsize=8,
                          color="#6b7799", rotation=0, labelpad=44, va="center")
            ax.grid(True, linewidth=0.4)
            ax.set_xlim(t[0], t[-1])
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_locator(MaxNLocator(4))

            if r == 0:
                ax.legend(loc="upper right", fontsize=6,
                          ncol=min(cols, 7), framealpha=0.4,
                          labelcolor="linecolor")

        axes[-1].set_xlabel("time (s)", fontsize=9)
        fig.tight_layout()
        _save_or_show(fig, save_path)


# ── Plot 3: correlation matrix ─────────────────────────────────────────────────

def plot_correlation_matrix(data, rows, cols, sig_names=None,
                             title="Sensor correlation matrix",
                             save_path=None):
    """Pearson correlation matrix across all signals."""
    sensor_order = [(r, c) for r in range(rows) for c in range(cols)]
    labels = (sig_names if sig_names
              else [f"{r},{c}" for r, c in sensor_order])

    matrix = np.corrcoef(np.stack([data[rc] for rc in sensor_order]))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(8, cols * 0.9),
                                        max(7, rows * 0.9)))
        fig.suptitle(title, fontsize=12, fontweight="bold", color="#1a1f36")

        im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1,
                       aspect="auto", interpolation="nearest")

        # Row-group separator lines
        for i in range(1, rows):
            ax.axhline(i * cols - 0.5, color="#c0c8d8", linewidth=1.0)
            ax.axvline(i * cols - 0.5, color="#c0c8d8", linewidth=1.0)

        n = rows * cols
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=5.5)

        plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
        fig.tight_layout()
        _save_or_show(fig, save_path)


# ── Plot 4: power spectral density per row ────────────────────────────────────

def plot_psd(t, data, rows, cols,
             title="Power spectral density per row",
             save_path=None):
    """Welch PSD for each signal, grouped by row."""
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    fs = 1.0 / dt

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, 1, figsize=(14, 2.5 * rows), sharex=True)
        if rows == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=12, fontweight="bold", color="#1a1f36")

        for r in range(rows):
            ax = axes[r]
            for c in range(cols):
                col = COL_COLORS[c % len(COL_COLORS)]
                f, p = welch(data[(r, c)], fs=fs,
                             nperseg=min(512, max(4, len(t) // 4)))
                ax.semilogy(f, p, color=col, linewidth=0.6,
                            label=f"({r},{c})", alpha=0.85)

            ax.set_ylabel(f"row {r}\nPSD", fontsize=8,
                          color="#6b7799", rotation=0, labelpad=44, va="center")
            ax.grid(True, which="both", linewidth=0.3, alpha=0.6)
            ax.tick_params(labelsize=7)

            if r == 0:
                ax.legend(loc="upper right", fontsize=6,
                          ncol=min(cols, 7), framealpha=0.4,
                          labelcolor="linecolor")

        axes[-1].set_xlabel("frequency (Hz)", fontsize=9)
        axes[-1].set_xlim(0, fs / 2)
        fig.tight_layout()
        _save_or_show(fig, save_path)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_or_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight",
                    facecolor=STYLE["figure.facecolor"])
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize industrial sensor data from a CSV file."
    )
    parser.add_argument(
        "--csv", "-c", required=True,
        help="Path to the CSV file (first column = 'date', rest = signals)."
    )
    parser.add_argument(
        "--rows", "-r", type=int, default=5,
        help="Number of rows in the sensor grid (default: 5)."
    )
    parser.add_argument(
        "--cols", "-C", type=int, default=7,
        help="Number of columns in the sensor grid (default: 7)."
    )
    parser.add_argument(
        "--out_dir", "-o", default=None,
        help="Directory to save plots (PNG). If omitted, plots are shown interactively."
    )
    parser.add_argument(
        "--plots", "-p", nargs="+",
        default=["all"],
        choices=["all", "grid", "band", "corr", "psd"],
        help="Which plots to produce (default: all)."
    )
    parser.add_argument(
        "--title_prefix", default="",
        help="Optional prefix added to every plot title."
    )
    args = parser.parse_args()

    do_all = "all" in args.plots
    rows, cols = args.rows, args.cols
    prefix = (args.title_prefix + "  |  ").lstrip("  |  ") if args.title_prefix else ""

    def out_path(name):
        if args.out_dir is None:
            return None
        return os.path.join(args.out_dir, f"{name}.png")

    print(f"\n📊  Visualizing: {args.csv}")
    print(f"    Grid      : {rows} × {cols}")
    print(f"    Output    : {args.out_dir or 'interactive'}\n")

    # ── Load ──
    t, data, sig_names = load_csv(args.csv, rows, cols)
    mean_data, std_data = compute_stats(data, rows, cols)

    # ── Plot 1: signal grid ──
    if do_all or "grid" in args.plots:
        print("1/4  signal grid...")
        plot_grid_signals(
            t, data, rows, cols, sig_names=sig_names,
            title=f"{prefix}Sensor signals — {rows}×{cols} grid",
            save_path=out_path("01_grid_signals"),
        )

    # ── Plot 2: per-row mean ± std band ──
    if do_all or "band" in args.plots:
        print("2/4  per-row band overview...")
        plot_mean_std_band(
            t, mean_data, std_data, rows, cols,
            title=f"{prefix}Per-row signal overview",
            save_path=out_path("02_mean_std_band"),
        )

    # ── Plot 3: correlation matrix ──
    if do_all or "corr" in args.plots:
        print("3/4  correlation matrix...")
        plot_correlation_matrix(
            data, rows, cols, sig_names=sig_names,
            title=f"{prefix}Sensor correlation matrix",
            save_path=out_path("03_correlation_matrix"),
        )

    # ── Plot 4: PSD per row ──
    if do_all or "psd" in args.plots:
        print("4/4  power spectral density...")
        plot_psd(
            t, data, rows, cols,
            title=f"{prefix}Power spectral density per row",
            save_path=out_path("04_psd"),
        )

    print("\n✅  Done!")


if __name__ == "__main__":
    main()
