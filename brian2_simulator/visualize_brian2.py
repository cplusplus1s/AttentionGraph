"""
visualize_brian2.py
────────────────────
Visualization toolkit for Brian2 5×5 SNN simulation data.

Plots produced:
  1. grid_signals_healthy   — 5×5 grid of firing-rate time series (one healthy run)
  2. grid_signals_unhealthy — 5×5 grid of firing-rate time series (one unhealthy run)
  3. healthy_vs_fault       — side-by-side 5×5 grids: healthy mean vs fault run
  4. mean_std_band          — mean ± std band across N healthy runs per sensor
  5. difference_heatmap     — per-sensor mean-rate difference (healthy − fault)
  6. correlation_matrix     — Pearson correlation of sensor signals (healthy vs fault)
  7. psd                    — power spectral density per sensor row (healthy vs fault)

Usage:
  python visualize_brian2.py --topology hourglass --data_dir ./data/raw/brian2
  python visualize_brian2.py --topology highway --plots grid_h grid_u
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.signal import welch

# ── Aesthetics ─────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor":   "#ffffff",   # was "#0f1117"
    "axes.facecolor":     "#f7f8fc",   # was "#161b27"
    "axes.edgecolor":     "#c0c8d8",   # was "#2e3650"
    "axes.labelcolor":    "#2e3650",   # was "#c8d0e8"
    "axes.titlecolor":    "#1a1f36",   # was "#e8eaf6"
    "xtick.color":        "#6b7799",
    "ytick.color":        "#6b7799",
    "grid.color":         "#dde2ef",   # was "#1e2540"
    "grid.linewidth":     0.5,
    "text.color":         "#2e3650",   # was "#c8d0e8"
    "font.family":        "monospace",
    "lines.linewidth":    0.6,         # was 1.0  → global default thinner
    "figure.dpi":         120,
}

# Color ramp: teal for drivers (col0), purple for hub-like nodes, otherwise
# gradient by column so structure is visually encoded in color.
COL_COLORS = ["#4ecdc4", "#45b7d1", "#7c83e0", "#a78bfa", "#f472b6"]
HEALTHY_COLOR = "#4ecdc4"
FAULT_COLOR   = "#f87171"
BAND_ALPHA    = 0.18
ROWS, COLS    = 5, 5


# ── I/O helpers ────────────────────────────────────────────────────────────────

def sensor_col(r, c):
    return f"brian2_sensor_{r}_{c}"

def load_run(csv_path):
    """Load one CSV. Returns (time_array, data_dict {(r,c): np.array})."""
    df = pd.read_csv(csv_path)
    t  = df["time_sec"].values
    data = {}
    for r in range(ROWS):
        for c in range(COLS):
            col = sensor_col(r, c)
            data[(r, c)] = df[col].values if col in df.columns else np.zeros_like(t)
    return t, data

def load_healthy_runs(data_dir, topology, n=30):
    """Load up to n healthy runs. Returns list of (t, data) tuples."""
    runs = []
    for i in range(1, n + 1):
        path = os.path.join(data_dir, topology, f"healthy_{i}", "brian2_data.csv")
        if os.path.exists(path):
            runs.append(load_run(path))
    if not runs:
        raise FileNotFoundError(
            f"No healthy runs found at {data_dir}/{topology}/healthy_*/brian2_data.csv"
        )
    print(f"Loaded {len(runs)} healthy runs for topology '{topology}'")
    return runs

def load_unhealthy_runs(data_dir, topology, n=30):
    """Load up to n unhealthy runs. Returns list of (t, data) tuples."""
    runs = []
    for i in range(1, n + 1):
        path = os.path.join(data_dir, topology, f"unhealthy_{i}", "brian2_data.csv")
        if os.path.exists(path):
            runs.append(load_run(path))
    if not runs:
        raise FileNotFoundError(
            f"No unhealthy runs found at {data_dir}/{topology}/unhealthy_*/brian2_data.csv"
        )
    print(f"Loaded {len(runs)} unhealthy runs for topology '{topology}'")
    return runs

def load_fault_run(data_dir, topology, idx=1):
    path = os.path.join(data_dir, topology, f"unhealthy_{idx}", "brian2_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fault run not found: {path}")
    return load_run(path)

def healthy_stats(runs):
    """
    Compute per-sensor mean and std across all healthy runs.
    Returns dicts {(r,c): mean_array} and {(r,c): std_array}.
    Assumes all runs share the same time axis.
    """
    t = runs[0][0]
    mean_data, std_data = {}, {}
    for r in range(ROWS):
        for c in range(COLS):
            stack = np.stack([run[1][(r, c)] for run in runs], axis=0)  # (n_runs, T)
            mean_data[(r, c)] = stack.mean(axis=0)
            std_data[(r, c)]  = stack.std(axis=0)
    return t, mean_data, std_data


# ── Plot 1/2: 5×5 grid of signals (single run, healthy or unhealthy) ───────────

def plot_grid_signals(t, data, title="Firing rate signals — 5×5 grid",
                      save_path=None):
    """Plot all 25 sensor signals in a 5×5 subplot grid."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(ROWS, COLS, figsize=(16, 10), sharex=True)
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01,
                     color="#e8eaf6")

        for r in range(ROWS):
            for c in range(COLS):
                ax   = axes[r][c]
                sig  = data[(r, c)]
                color = COL_COLORS[c]

                ax.plot(t, sig, color=color, linewidth=0.5, alpha=0.9)
                ax.fill_between(t, sig, alpha=0.12, color=color)

                # Subtle grid
                ax.grid(True, axis="y", linewidth=0.4)
                ax.set_xlim(t[0], t[-1])

                # Row/col labels only on edges
                if r == 0:
                    ax.set_title(f"col {c}", fontsize=8, color="#6b7799", pad=3)
                if c == 0:
                    ax.set_ylabel(f"row {r}", fontsize=8, color="#6b7799",
                                  rotation=0, labelpad=28, va="center")

                # Node label inside each panel
                ax.text(0.97, 0.92, f"({r},{c})", transform=ax.transAxes,
                        fontsize=7, color="#4a5580", ha="right", va="top")

                # Hide inner tick labels
                if r < ROWS - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("time (s)", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.yaxis.set_major_locator(MaxNLocator(3))

        fig.tight_layout()
        _save_or_show(fig, save_path)


# ── Plot 3: Healthy mean vs Fault — side-by-side 5×5 grids ───────────────────

def plot_healthy_vs_fault(t, mean_data, std_data, fault_data,
                          topology="", save_path=None):
    """Two 5×5 grids side by side: healthy mean±std (left) vs fault (right)."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(22, 10))
        fig.patch.set_facecolor(STYLE["figure.facecolor"])

        outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08,
                                  left=0.04, right=0.97, top=0.93, bottom=0.07)

        titles = ["Healthy baseline  (mean ± std, 30 runs)",
                  "Fault run  (one synapse cut)"]
        datasets = [(mean_data, std_data, False), (fault_data, None, True)]

        for side, (data, std, is_fault) in enumerate(datasets):
            inner = gridspec.GridSpecFromSubplotSpec(
                ROWS, COLS, subplot_spec=outer[side], hspace=0.08, wspace=0.08
            )
            color_main = FAULT_COLOR if is_fault else HEALTHY_COLOR

            for r in range(ROWS):
                for c in range(COLS):
                    ax  = fig.add_subplot(inner[r, c])
                    sig = data[(r, c)]
                    col_color = FAULT_COLOR if is_fault else COL_COLORS[c]

                    ax.plot(t, sig, color=col_color, linewidth=0.5, alpha=0.95)

                    if std is not None:
                        ax.fill_between(t,
                                        sig - std[(r, c)],
                                        sig + std[(r, c)],
                                        color=col_color, alpha=BAND_ALPHA)

                    ax.set_xlim(t[0], t[-1])
                    ax.grid(True, axis="y", linewidth=0.3)

                    # Thin border accent
                    for spine in ax.spines.values():
                        spine.set_edgecolor(col_color)
                        spine.set_linewidth(0.4 if not is_fault else 0.7)

                    ax.tick_params(labelsize=5, length=2)
                    ax.yaxis.set_major_locator(MaxNLocator(2))

                    if r < ROWS - 1: ax.tick_params(labelbottom=False)
                    if c > 0:        ax.tick_params(labelleft=False)

                    ax.text(0.96, 0.90, f"({r},{c})", transform=ax.transAxes,
                            fontsize=6, color="#3a4570", ha="right", va="top")

            # Section title
            fig.text(0.25 + side * 0.5, 0.965, titles[side],
                     ha="center", va="center", fontsize=11,
                     color=color_main, fontweight="bold")

        fig.suptitle(f"Topology: {topology}", fontsize=10, color="#6b7799", y=0.998)
        _save_or_show(fig, save_path)


# ── Plot 4: Mean ± std band — per row ─────────────────────────────────────────

def plot_mean_std_band(t, mean_data, std_data, fault_data=None,
                       topology="", save_path=None):
    """
    For each row (0..4), plot all 5 sensors' mean±std bands.
    Optionally overlay the fault run's signals as dashed lines.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(ROWS, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f"Per-row healthy baseline ± std  |  topology: {topology}",
                     fontsize=12, fontweight="bold", color="#e8eaf6")

        for r in range(ROWS):
            ax = axes[r]
            for c in range(COLS):
                mu  = mean_data[(r, c)]
                sig = std_data[(r, c)]
                col = COL_COLORS[c]
                label = f"({r},{c})"

                ax.fill_between(t, mu - sig, mu + sig,
                                alpha=BAND_ALPHA, color=col)
                ax.plot(t, mu, color=col, linewidth=0.9,
                        label=label, alpha=0.9)

                if fault_data is not None:
                    ax.plot(t, fault_data[(r, c)], color=col,
                            linewidth=0.8, linestyle="--", alpha=0.6)

            ax.set_ylabel(f"row {r}\nrate (Hz)", fontsize=8,
                          color="#6b7799", rotation=0, labelpad=44, va="center")
            ax.grid(True, linewidth=0.4)
            ax.set_xlim(t[0], t[-1])
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_locator(MaxNLocator(4))

            if r == 0:
                ax.legend(loc="upper right", fontsize=6, ncol=5,
                          framealpha=0.2, labelcolor="linecolor")

        axes[-1].set_xlabel("time (s)", fontsize=9)

        if fault_data is not None:
            fig.text(0.97, 0.01,
                     "— solid: healthy mean  |  -- dashed: fault run",
                     ha="right", fontsize=8, color="#6b7799", style="italic")

        fig.tight_layout()
        _save_or_show(fig, save_path)


# ── Plot 5: Difference heatmap (healthy mean − fault) ─────────────────────────

def plot_difference_heatmap(mean_data, fault_data, topology="", save_path=None):
    """
    5×5 heatmap of (healthy_mean_rate − fault_mean_rate) per sensor.
    """
    diff = np.zeros((ROWS, COLS))
    for r in range(ROWS):
        for c in range(COLS):
            diff[r, c] = mean_data[(r, c)].mean() - fault_data[(r, c)].mean()

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                                 gridspec_kw={"width_ratios": [1, 1, 0.05]})
        fig.suptitle(
            f"Mean firing-rate difference  (healthy − fault)  |  topology: {topology}",
            fontsize=12, fontweight="bold", color="#e8eaf6"
        )

        vmax = np.abs(diff).max() + 1e-6

        ax = axes[0]
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_title("Δ rate  (Hz)", fontsize=9, color="#c8d0e8")
        _annotate_heatmap(ax, diff, fmt="{:.1f}")

        ax2 = axes[1]
        im2 = ax2.imshow(np.abs(diff), cmap="YlOrRd", vmin=0, vmax=vmax,
                         aspect="auto", interpolation="nearest")
        ax2.set_title("|Δ rate|  (Hz)", fontsize=9, color="#c8d0e8")
        _annotate_heatmap(ax2, np.abs(diff), fmt="{:.1f}")

        for ax in axes[:2]:
            ax.set_xticks(range(COLS))
            ax.set_xticklabels([f"c{c}" for c in range(COLS)], fontsize=8)
            ax.set_yticks(range(ROWS))
            ax.set_yticklabels([f"r{r}" for r in range(ROWS)], fontsize=8)

        plt.colorbar(im2, cax=axes[2], label="Hz")

        fig.tight_layout()
        _save_or_show(fig, save_path)


# ── Plot 6: Correlation matrices (healthy vs fault) ────────────────────────────

def plot_correlation_matrices(mean_data, fault_data, healthy_runs,
                               topology="", save_path=None):
    """Pearson correlation matrices across all 25 sensors."""
    sensor_order = [(r, c) for r in range(ROWS) for c in range(COLS)]
    labels = [f"{r},{c}" for r, c in sensor_order]

    healthy_stack = np.hstack(
        [np.stack([run[1][rc] for rc in sensor_order]) for run in healthy_runs]
    )
    corr_healthy = np.corrcoef(healthy_stack)

    fault_stack  = np.stack([fault_data[rc] for rc in sensor_order])
    corr_fault   = np.corrcoef(fault_stack)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"Sensor correlation matrices  |  topology: {topology}",
                     fontsize=12, fontweight="bold", color="#e8eaf6")

        for ax, corr, title in zip(
            axes,
            [corr_healthy, corr_fault],
            ["Healthy baseline (all runs)", "Fault run"]
        ):
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1,
                           aspect="auto", interpolation="nearest")
            ax.set_title(title, fontsize=10, color="#c8d0e8", pad=8)

            for i in range(1, ROWS):
                ax.axhline(i * COLS - 0.5, color="#2e3650", linewidth=1.2)
                ax.axvline(i * COLS - 0.5, color="#2e3650", linewidth=1.2)

            ax.set_xticks(range(25))
            ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
            ax.set_yticks(range(25))
            ax.set_yticklabels(labels, fontsize=5.5)

        plt.colorbar(im, ax=axes[1], label="Pearson r", shrink=0.8)
        fig.tight_layout()
        _save_or_show(fig, save_path)


# ── Plot 7: Power spectral density per row ─────────────────────────────────────

def plot_psd(t, mean_data, fault_data, topology="", save_path=None):
    """PSD (Welch) for each sensor, grouped by row."""
    dt = float(t[1] - t[0])
    fs = 1.0 / dt

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(ROWS, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(
            f"Power spectral density per row  |  topology: {topology}",
            fontsize=12, fontweight="bold", color="#e8eaf6"
        )

        for r in range(ROWS):
            ax = axes[r]
            for c in range(COLS):
                col = COL_COLORS[c]

                f_h, p_h = welch(mean_data[(r, c)],  fs=fs, nperseg=min(512, len(t)//4))
                f_f, p_f = welch(fault_data[(r, c)], fs=fs, nperseg=min(512, len(t)//4))

                ax.semilogy(f_h, p_h, color=col, linewidth=0.6,
                            label=f"({r},{c})", alpha=0.85)
                ax.semilogy(f_f, p_f, color=col, linewidth=0.5,
                            linestyle="--", alpha=0.55)

            ax.set_ylabel(f"row {r}\nPSD", fontsize=8,
                          color="#6b7799", rotation=0, labelpad=44, va="center")
            ax.grid(True, which="both", linewidth=0.3, alpha=0.6)
            ax.tick_params(labelsize=7)

            if r == 0:
                ax.legend(loc="upper right", fontsize=6, ncol=5,
                          framealpha=0.2, labelcolor="linecolor")

        axes[-1].set_xlabel("frequency (Hz)", fontsize=9)
        axes[-1].set_xlim(0, fs / 2)

        fig.text(0.97, 0.01,
                 "— solid: healthy  |  -- dashed: fault",
                 ha="right", fontsize=8, color="#6b7799", style="italic")
        fig.tight_layout()
        _save_or_show(fig, save_path)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _annotate_heatmap(ax, data, fmt="{:.1f}"):
    """Annotate each cell with its value."""
    vmax = np.abs(data).max()
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val   = data[r, c]
            color = "white" if abs(val) > 0.5 * vmax else "#8899bb"
            ax.text(c, r, fmt.format(val), ha="center", va="center",
                    fontsize=7, color=color)

def _save_or_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        description="Visualize Brian2 5×5 simulation data."
    )
    parser.add_argument(
        "--topology", "-t",
        default="highway",
        choices=["hourglass", "chain_of_chains", "funnel", "highway", "binary_tree"],
        help="Which topology's data to visualize."
    )
    parser.add_argument(
        "--data_dir", "-d",
        default="./data/raw/brian2",
        help="Root data directory (contains <topology>/healthy_N/ sub-dirs)."
    )
    parser.add_argument(
        "--out_dir", "-o",
        default=None,
        help="Directory to save plots (PNG). If omitted, plots are shown interactively."
    )
    parser.add_argument(
        "--n_healthy", "-n",
        type=int, default=30,
        help="Max number of healthy runs to load."
    )
    parser.add_argument(
        "--n_unhealthy", "-u",
        type=int, default=30,
        help="Max number of unhealthy runs to load."
    )
    parser.add_argument(
        "--plots", "-p",
        nargs="+",
        default=["all"],
        choices=["all", "grid_h", "grid_u", "compare", "band", "diff", "corr", "psd"],
        help="Which plots to produce."
    )
    args = parser.parse_args()

    do_all = "all" in args.plots
    topo   = args.topology
    out    = args.out_dir

    def out_path(name):
        if out is None:
            return None
        return os.path.join(out, topo, f"{name}.png")

    print(f"\n📊 Visualizing topology: {topo}")
    print(f"   Data dir : {args.data_dir}")
    print(f"   Output   : {out or 'interactive'}\n")

    # Load data
    healthy_runs = load_healthy_runs(args.data_dir, topo, n=args.n_healthy)
    t_h, mean_data, std_data = healthy_stats(healthy_runs)

    t_f, fault_data = load_fault_run(args.data_dir, topo)

    # Use time axis from healthy (both should be equal length; trim if needed)
    T = min(len(t_h), len(t_f))
    t = t_h[:T]
    for rc in [(r, c) for r in range(ROWS) for c in range(COLS)]:
        mean_data[rc] = mean_data[rc][:T]
        std_data[rc]  = std_data[rc][:T]
        fault_data[rc] = fault_data[rc][:T]

    # ── Plot 1: single healthy run grid ──
    if do_all or "grid_h" in args.plots:
        print("1/7  grid signals (first healthy run)...")
        t1, data1 = healthy_runs[0]
        plot_grid_signals(
            t1[:T], {rc: data1[rc][:T] for rc in data1},
            title=f"Firing-rate signals — {topo} — healthy run 1",
            save_path=out_path("01_grid_signals_healthy")
        )

    # ── Plot 2: single unhealthy run grid ──
    if do_all or "grid_u" in args.plots:
        print("2/7  grid signals (first unhealthy run)...")
        unhealthy_runs = load_unhealthy_runs(args.data_dir, topo, n=args.n_unhealthy)
        t2, data2 = unhealthy_runs[0]
        T_u = min(len(t2), T)
        plot_grid_signals(
            t2[:T_u], {rc: data2[rc][:T_u] for rc in data2},
            title=f"Firing-rate signals — {topo} — unhealthy run 1",
            save_path=out_path("02_grid_signals_unhealthy")
        )

    # # ── Plot 3: healthy mean vs fault ──
    # if do_all or "compare" in args.plots:
    #     print("3/7  healthy vs fault comparison grids...")
    #     plot_healthy_vs_fault(
    #         t, mean_data, std_data, fault_data,
    #         topology=topo,
    #         save_path=out_path("03_healthy_vs_fault")
    #     )

    # # ── Plot 4: mean ± std bands ──
    # if do_all or "band" in args.plots:
    #     print("4/7  mean ± std bands per row...")
    #     plot_mean_std_band(
    #         t, mean_data, std_data, fault_data,
    #         topology=topo,
    #         save_path=out_path("04_mean_std_band")
    #     )

    # # ── Plot 5: difference heatmap ──
    # if do_all or "diff" in args.plots:
    #     print("5/7  difference heatmap...")
    #     plot_difference_heatmap(
    #         mean_data, fault_data,
    #         topology=topo,
    #         save_path=out_path("05_difference_heatmap")
    #     )

    # # ── Plot 6: correlation matrices ──
    # if do_all or "corr" in args.plots:
    #     print("6/7  correlation matrices...")
    #     plot_correlation_matrices(
    #         mean_data, fault_data, healthy_runs,
    #         topology=topo,
    #         save_path=out_path("06_correlation_matrices")
    #     )

    # # ── Plot 7: PSD per row ──
    # if do_all or "psd" in args.plots:
    #     print("7/7  power spectral density...")
    #     plot_psd(
    #         t, mean_data, fault_data,
    #         topology=topo,
    #         save_path=out_path("07_psd")
    #     )

    print("\n✅ Done!")


if __name__ == "__main__":
    main()